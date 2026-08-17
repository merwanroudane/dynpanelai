"""AC-GATE: entity-conditioned heterogeneous lag discovery.

Implements

    Xu, A. (2026). Discovering entity-conditioned lag heterogeneity: a
    lag-gated neural audit framework for panel time series.

The question is not "what is the forecast?" but "how far back does each unit
look?"  Countries, firms and regions absorb shocks at different speeds, and
pooling them into one lag structure hides that.  AC-GATE makes the effective
lag a *structural output* of the model rather than a post-hoc explanation:

.. math::
    p_i \\;\\to\\; z_i = f_\\phi(p_i) \\;\\to\\;
    \\omega_{i,k} = \\mathrm{softmax}_k\\!
      \\Bigl(\\frac{g_\\theta(z_i)_k - \\lambda k/K}{\\tau}\\Bigr)
    \\;\\to\\; c_{i,t} = \\sum_k \\omega_{i,k}\\tilde X_{i,t-k}
    \\;\\to\\; \\hat Y_{i,t+1},

with the auditable per-entity effective lag
:math:`k^\\star_i = \\sum_k k\\,\\omega_{i,k}`.

Two design choices carry the argument:

- the position bias :math:`\\lambda k/K` is normalised by ``K``, so the
  penalty on distant lags does not change meaning when the horizon changes;
- the contemporaneous :math:`X_{i,t}` **never** enters the backbone.  Only
  lagged values reach the network through :math:`c_{i,t}`.  Without this the
  recurrent layer could ignore the gate entirely and read the answer off
  today's covariates -- the classic leakage failure in panel ML.

Requires PyTorch: ``pip install dynpanelai[neural]``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..core.panel import PanelData

__all__ = ["ACGate", "ACGateResult", "build_lag_tensors"]


def build_lag_tensors(
    panel: PanelData,
    y: str,
    features: Sequence[str],
    proxies: Sequence[str],
    K: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assemble the lag tensor for AC-GATE.

    Parameters
    ----------
    panel : PanelData
    y : str
        Target column.
    features : sequence of str
        Time-varying covariates.
    proxies : sequence of str
        Entity-level (time-invariant) conditioning variables.
    K : int
        Maximum lag horizon.

    Returns
    -------
    X : ndarray of shape (n_samples, K, n_features)
        ``X[s, k]`` is the covariate vector at lag ``k+1`` for sample ``s``.
    Y : ndarray of shape (n_samples,)
    P : ndarray of shape (n_entities, n_proxies)
    unit_idx : ndarray of shape (n_samples,)
        Entity index of each sample.
    times : ndarray of shape (n_samples,)

    Notes
    -----
    Only observations with a complete lag window of length ``K`` are kept, so
    each unit loses its first ``K`` periods.
    """
    df = panel.df
    n_feat = len(features)
    uniq = np.arange(panel.N)

    P = np.zeros((panel.N, len(proxies)))
    for j, pcol in enumerate(proxies):
        P[:, j] = df.groupby("_i")[pcol].first().reindex(uniq).to_numpy()

    Xs, Ys, us, ts = [], [], [], []
    for i, idx in df.groupby("_i").groups.items():
        rows = df.loc[idx].sort_values("_t")
        vals = rows[list(features)].to_numpy(float)
        yv = rows[y].to_numpy(float)
        tv = rows["_t"].to_numpy()
        for pos in range(K, len(rows)):
            window = vals[pos - K : pos][::-1]  # lag 1 first
            if not np.isfinite(window).all() or not np.isfinite(yv[pos]):
                continue
            Xs.append(window)
            Ys.append(yv[pos])
            us.append(i)
            ts.append(tv[pos])

    if not Xs:
        raise ValueError(
            f"no samples with a complete lag window of K={K}; "
            "reduce K or check for gaps in the panel"
        )
    return (
        np.stack(Xs).astype(np.float32),
        np.asarray(Ys, dtype=np.float32),
        P.astype(np.float32),
        np.asarray(us),
        np.asarray(ts),
    )


@dataclass
class ACGateResult:
    """Output of an AC-GATE fit.

    Attributes
    ----------
    effective_lag : pandas.Series
        :math:`k^\\star_i` for each entity.
    lag_weights : pandas.DataFrame
        The full :math:`\\omega_{i,k}` distribution, entities by lags.
    metrics : dict
        Test-set MSE, MAE and :math:`R^2` (audit layer L0).
    history : list of float
        Validation loss by epoch.
    """

    effective_lag: pd.Series
    lag_weights: pd.DataFrame
    metrics: dict
    history: list = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"ACGateResult(k* mean={self.effective_lag.mean():.2f}, "
            f"sd={self.effective_lag.std():.3f}, "
            f"test R2={self.metrics.get('r2', float('nan')):.3f})"
        )


class ACGate:
    """Adaptive-conditioning encoder with a scale-invariant lag gate.

    Parameters
    ----------
    y : str
        Target column.
    features : sequence of str
        Time-varying covariates entering only through lags.
    proxies : sequence of str
        Time-invariant entity characteristics that condition the lag gate.
    K : int, default 8
        Maximum lag horizon.
    hidden : int, default 32
        LSTM hidden size.
    layers : int, default 2
        LSTM depth.
    tau : float, default 1.0
        Softmax temperature; lower values sharpen the lag distribution.
    lam_pos : float, default 0.5
        Coefficient on the normalised position bias :math:`\\lambda k/K`.
    lam_recon : float, default 0.1
        Weight on the proxy-reconstruction auxiliary loss.
    epochs : int, default 60
    lr : float, default 1e-3
    batch_size : int, default 256
    seed : int, default 0

    Examples
    --------
    >>> from dynpanelai.neural import ACGate
    >>> est = ACGate(y="y", features=["f0", "f1"],
    ...              proxies=["proxy0", "proxy1"], K=8)
    >>> res = est.fit(panel)                        # doctest: +SKIP
    >>> res.effective_lag.head()                    # doctest: +SKIP
    """

    def __init__(
        self,
        y: str,
        features: Sequence[str],
        proxies: Sequence[str],
        *,
        K: int = 8,
        hidden: int = 32,
        layers: int = 2,
        tau: float = 1.0,
        lam_pos: float = 0.5,
        lam_recon: float = 0.1,
        epochs: int = 60,
        lr: float = 1e-3,
        batch_size: int = 256,
        val_fraction: float = 0.15,
        test_fraction: float = 0.15,
        seed: int = 0,
    ) -> None:
        self.y = y
        self.features = list(features)
        self.proxies = list(proxies)
        self.K = K
        self.hidden = hidden
        self.layers = layers
        self.tau = tau
        self.lam_pos = lam_pos
        self.lam_recon = lam_recon
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.seed = seed
        self.result_: ACGateResult | None = None

    # ------------------------------------------------------------------
    def _build(self, n_features: int, n_proxies: int):
        import torch
        from torch import nn

        K, H, L = self.K, self.hidden, self.layers
        tau, lam_pos = self.tau, self.lam_pos

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                # adaptive-conditioning encoder: proxies -> scalar score
                self.enc = nn.Sequential(
                    nn.Linear(n_proxies, 16), nn.Tanh(), nn.Linear(16, 1)
                )
                # reconstruction head, for the auxiliary objective
                self.dec = nn.Sequential(
                    nn.Linear(1, 16), nn.Tanh(), nn.Linear(16, n_proxies)
                )
                # scalar score -> K lag logits
                self.gate = nn.Sequential(
                    nn.Linear(1, 16), nn.Tanh(), nn.Linear(16, K)
                )
                self.lstm = nn.LSTM(
                    n_features, H, num_layers=L, batch_first=True
                )
                self.head = nn.Sequential(nn.Linear(H + 1, 32), nn.ReLU(), nn.Linear(32, 1))
                pos = torch.arange(1, K + 1, dtype=torch.float32) / K
                self.register_buffer("pos_bias", pos)

            def lag_weights(self, p):
                z = self.enc(p)
                logits = (self.gate(z) - lam_pos * self.pos_bias) / tau
                return torch.softmax(logits, dim=-1), z

            def forward(self, x, p):
                # x: (B, K, F)   p: (B, n_proxies)
                w, z = self.lag_weights(p)
                # lag-weighted context; the contemporaneous X never appears
                ctx = (x * w.unsqueeze(-1)).sum(dim=1, keepdim=True)
                ctx = ctx.repeat(1, 2, 1)
                out, _ = self.lstm(ctx)
                h = out[:, -1, :]
                pred = self.head(torch.cat([h, z], dim=-1)).squeeze(-1)
                return pred, w, z, self.dec(z)

        return Net()

    # ------------------------------------------------------------------
    def fit(self, panel: PanelData) -> ACGateResult:
        """Train the model and extract the effective lags.

        Parameters
        ----------
        panel : PanelData

        Returns
        -------
        ACGateResult

        Raises
        ------
        ImportError
            If PyTorch is not installed.

        Notes
        -----
        Splits are **chronological**, never random: validation and test
        periods follow the training periods, so no future information can
        leak into the fitted lag gate.
        """
        try:
            import torch
            from torch import nn
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "AC-GATE needs PyTorch. Install it with "
                "`pip install dynpanelai[neural]` or `pip install torch`."
            ) from exc

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        X, Y, P, units, times = build_lag_tensors(
            panel, self.y, self.features, self.proxies, self.K
        )

        # chronological split
        order = np.argsort(times)
        n = len(order)
        n_test = int(self.test_fraction * n)
        n_val = int(self.val_fraction * n)
        test_idx = order[n - n_test :]
        val_idx = order[n - n_test - n_val : n - n_test]
        train_idx = order[: n - n_test - n_val]

        # standardise using training statistics only
        mu_x = X[train_idx].reshape(-1, X.shape[2]).mean(axis=0)
        sd_x = X[train_idx].reshape(-1, X.shape[2]).std(axis=0) + 1e-8
        mu_y, sd_y = Y[train_idx].mean(), Y[train_idx].std() + 1e-8
        mu_p, sd_p = P.mean(axis=0), P.std(axis=0) + 1e-8

        Xn = (X - mu_x) / sd_x
        Yn = (Y - mu_y) / sd_y
        Pn = (P - mu_p) / sd_p

        def t(a):
            return torch.tensor(np.asarray(a, dtype=np.float32))

        Xt, Yt, Pt = t(Xn), t(Yn), t(Pn)
        Pobs = Pt[units]

        net = self._build(X.shape[2], P.shape[1])
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        mse = nn.MSELoss()

        best_state, best_val, history = None, np.inf, []
        for _ in range(self.epochs):
            net.train()
            perm = np.random.permutation(train_idx)
            for s in range(0, len(perm), self.batch_size):
                b = perm[s : s + self.batch_size]
                opt.zero_grad()
                pred, _, _, precon = net(Xt[b], Pobs[b])
                loss = mse(pred, Yt[b]) + self.lam_recon * mse(precon, Pobs[b])
                loss.backward()
                opt.step()

            net.eval()
            with torch.no_grad():
                pv, _, _, _ = net(Xt[val_idx], Pobs[val_idx])
                v = float(mse(pv, Yt[val_idx]))
            history.append(v)
            if v < best_val:
                best_val = v
                best_state = {k: q.clone() for k, q in net.state_dict().items()}

        if best_state is not None:
            net.load_state_dict(best_state)

        net.eval()
        with torch.no_grad():
            pt, _, _, _ = net(Xt[test_idx], Pobs[test_idx])
            pred_test = pt.numpy() * sd_y + mu_y
            w_all, _ = net.lag_weights(Pt)
            W = w_all.numpy()

        y_test = Y[test_idx]
        err = y_test - pred_test
        ss_res = float(np.sum(err**2))
        ss_tot = float(np.sum((y_test - y_test.mean()) ** 2))
        metrics = {
            "mse": float(np.mean(err**2)),
            "mae": float(np.mean(np.abs(err))),
            "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
            "n_test": int(len(test_idx)),
        }

        lags = np.arange(1, self.K + 1)
        k_star = W @ lags
        idx = panel.units
        result = ACGateResult(
            effective_lag=pd.Series(k_star, index=idx, name="k_star"),
            lag_weights=pd.DataFrame(
                W, index=idx, columns=[f"lag{k}" for k in lags]
            ),
            metrics=metrics,
            history=history,
        )
        self.result_ = result
        return result
