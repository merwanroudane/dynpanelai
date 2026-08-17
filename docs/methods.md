# Methods

The econometrics behind each module, with the equations and the reason each
method exists. For code, see the [user guide](user_guide.md).

---

## The core problem

A dynamic panel with unit effects,

$$y_{it} = \rho y_{i,t-1} + \theta d_{it} + \alpha_i + u_{it},$$

cannot be estimated consistently by the within estimator for fixed $T$. Demeaning
subtracts $\bar y_{i,-1}$, which contains $u_{it}$, so the transformed regressor
and error are correlated. The resulting **Nickell (1981) bias** is $O(1/T)$ and
biases $\hat\rho$ *downward*.

Every method here is a different answer to that problem, plus the complications
that arise when the controls are high-dimensional.

---

## 1. Moment-based estimation (`gmm`)

First-differencing removes $\alpha_i$:

$$\Delta y_{it} = \rho\,\Delta y_{i,t-1} + \theta\,\Delta d_{it} + \Delta u_{it}.$$

The transformed error is MA(1), so $y_{i,t-2}$ and deeper lags are valid
instruments. Stacking them over all $t$ gives Arellano and Bond's GMM estimator.

**Forward orthogonal deviations** (Arellano and Bover, 1995) are the alternative:

$$\Delta^{\perp} z_{it} = c_t\Bigl(z_{it} - \tfrac{1}{T-t}\textstyle\sum_{s>t} z_{is}\Bigr),
\qquad c_t = \sqrt{\tfrac{T-t}{T-t+1}}.$$

FOD annihilates $\alpha_i$ *and* leaves the transformed error serially
uncorrelated — unlike first differences. That is why it is more efficient, and
why cross-fitting does not disturb its asymptotics.

**The catch.** The number of moment conditions grows like $T^2$. Overfitting in
the first-stage projection produces a bias of order $m/n = T/N$, and valid
inference needs

$$m^2/n = T^3/N \to 0.$$

In the COVID application ($m = 3{,}375$, $NT \approx 68{,}000$) this quantity is
about **168**. The estimator is badly biased.

---

## 2. Bias correction (`biascorr`)

Three ways to remove the leading $O(1/T)$ term without instruments.

**Analytical.** Estimate the score's non-zero mean directly:

$$\hat\theta_{\text{DFE}} = \hat\theta_{\text{FE}}
+ \Bigl(\tfrac1n X'X\Bigr)^{-1}\Bigl(\tfrac1n\sum_{i,t} X_{it}\hat u_{i,t-1}\Bigr)\cdot\tfrac{N}{n}.$$

The $N/n = 1/T$ factor is essential: it is what makes the correction the same
order as the bias.

**Split-panel jackknife.** If the bias is $B/T$, halving the panel doubles it, so

$$\hat\theta_{\text{SPJ}} = 2\hat\theta_{\text{full}} - \tfrac12(\hat\theta_A + \hat\theta_B)$$

cancels the leading term. Splitting on *time* is Dhaene and Jochmans (2015);
splitting on the *cross-section* is what Chen, Chernozhukov and Fernández-Val
apply to Arellano–Bond, giving the DAB estimator.

**Kiviet/Bruno.** A closed-form expansion of the bias, evaluated at a consistent
initial estimator.

> **Implementation note.** System GMM is disabled in `dynpanelai` 0.1.1: the
> level-equation instrument block does not validate against a design where the
> extra moments hold by construction. Use `diff_gmm(collapse=True)`.

---

## 3. Arellano–Bond LASSO (`ablasso`)

The moment conditions are *approximately sparse*: the effective dimension of the
informative instruments at period $t$ is $\min(\log N, t)$, far below $t$. So
select them.

**Step 1.** For each $t$ separately, LASSO-regress each transformed regressor on
the instrument history $X_{i1},\dots,X_{it}$:

$$\hat\Pi_t \in \arg\min \sum_i\Bigl(W_{it} - \pi_{t0} - \sum_{s\le t} X_{is}'\pi_{ts}\Bigr)^2
+ \lambda_t\sum_s \omega_{ts}|\pi_{ts}|_1.$$

**Step 2.** Estimate by **IV** using the fitted values as instruments:

$$\hat\theta = \Bigl(\sum_{i,t}\widehat{\Delta X}_{it}\Delta X_{it}'\Bigr)^{-1}
\sum_{i,t}\widehat{\Delta X}_{it}\Delta Y_{it}.$$

> **Why IV and not 2SLS?** Only the IV form has a moment function that is
> Neyman-orthogonal with respect to the first-stage coefficients $\Pi_t$. The
> 2SLS version, which also replaces $\Delta X_{it}$ on the right, does not — its
> first-stage estimation error enters at first order.

**AB-LASSO-SS** adds cross-sectional splitting with cross-fitting, aggregated by
the median over many random splits. This removes the residual overfitting bias
and makes the estimate invariant to the (arbitrary) ordering of units.

The small-bias condition improves from $T^3/N \to 0$ to
$\max_t \sqrt{s^*_t/N} \to 0$.

---

## 4. Double machine learning (`dml`)

For a partially linear dynamic panel with a nonparametric control function
$g_0(X_{it})$, residualise both sides on the information set

$$W_{it} = (\ddot Y_{i,t-1:t-L_y},\; \ddot D_{i,t-1:t-L_d},\; \ddot X_{i,t:t-L_x})$$

and solve the **partialling-out** moment

$$\mathbb E\bigl[\tilde D_{it}(\tilde Y_{it} - \theta_0\tilde D_{it})\bigr] = 0,
\qquad \tilde Y = \ddot Y - \ell_0(W),\;\; \tilde D = \ddot D - m_0(W).$$

Orthogonality means nuisance error enters only at second order, so the product
rate $\delta_Y\delta_D = o(n^{-1/2})$ suffices — each nuisance may converge
slowly.

**Fold design is the panel-specific contribution.** Random i.i.d. folds leak: the
observation adjacent to a held-out one is strongly correlated with it. Blocked-time
folds hold out contiguous periods and purge an additional buffer

$$B_* = B + L_*,\qquad B = \lceil\log T\rceil,$$

which satisfies $B_T\to\infty$, $B_T/T\to 0$ — enough for train–test dependence
to vanish under $\alpha$-mixing.

**The limitation to respect.** Because $W_{it}$ contains within-demeaned lagged
outcomes, Nickell contamination enters at $O(1/T)$, so $\sqrt N$ inference needs
$\sqrt N/T \to 0$. In short panels, use GMM.

---

## 5. Orthogonal and debiased Lasso (`ortho`)

When the treatment effect is itself high-dimensional — a base treatment $P_{it}$
interacted with a rich dictionary $K(X_{it})$ — three stages:

1. **Residualise** with **neighbours-left-out** cross-fitting: block $k$ is scored
   using nuisances fit on all blocks except $k$ and its two immediate neighbours.
   Strassen's coupling shows that under $\beta$-mixing the block and its
   quasi-complement can be replaced by independent copies with vanishing error.

2. **Orthogonal Lasso** on the residuals. When the CATE function is simpler than
   the control function, this attains $\sqrt{s\log d/NT}$ — faster than any
   single-stage regression.

3. **Debias** with a CLIME approximate inverse:

   $$\hat\beta_{DL} = \hat\beta_L + \hat\Omega\,\tfrac{1}{NT}\sum_{i,t}\hat V_{it}
     (\hat{\tilde Y}_{it} - \hat V_{it}'\hat\beta_L),$$

   $$\hat\Omega = \arg\min\|\Omega\|_1 \text{ s.t. } \|\hat Q\Omega - I\|_\infty \le \lambda_Q.$$

   CLIME needs only *approximate* sparsity of $Q^{-1}$; nodewise regression needs
   exact sparsity.

**Simultaneous inference.** Testing many coefficients pointwise produces false
positives. The Gaussian multiplier bootstrap of Chernozhukov, Chetverikov and
Kato gives a critical value for $\|Z\|_\infty$ that is valid even when $d \gg n$.
Use simultaneous bands whenever the claim is about *which* coefficients differ.

---

## 6. Uniform inference with weakly sparse effects (`hdpanel`)

Kock and Tang penalise the slopes and the fixed effects at different rates,
because $NT$ observations identify each $\alpha_j$ but only $T$ identify each
$\eta_i$:

$$\mathcal L(\gamma) = \|y - \Pi\gamma\|^2 + 2\lambda_N\|\alpha\|_1
+ \frac{2\lambda_N}{\sqrt N}\|\eta\|_1,
\qquad \lambda_N = \sqrt{4M\,NT(\log(p\vee N))^3}.$$

The fixed effects need only be **weakly sparse**, $\sum_i|\eta_i|^\nu \le E$ for
$0<\nu<1$ — a genuine middle ground between random effects (no correlation
allowed) and unrestricted fixed effects (not estimable in high dimensions).

Inference uses the **desparsified** estimator with
$\hat\Theta = \operatorname{diag}(\hat\Theta_Z, I_N)$, where $\hat\Theta_Z$ comes
from nodewise regressions. The resulting bands are *honest*: uniformly valid, and
usable for a growing number of coefficients at once.

> **Check the assumption.** If the unit effects are dense, the estimator
> degenerates toward pooled OLS and biases $\hat\rho$ *upward*. `PanelLasso`
> warns when it detects this.

---

## 7. Optimal shrinkage (`shrink`)

Let $y_j$ be the least-squares estimate of unit $j$'s effect, with variance
$M_j$. Shrink toward $\mu$:

$$\hat\theta_j = (I - S_j)\mu + S_j y_j, \qquad S_j = \Lambda(\Lambda + M_j)^{-1}.$$

**EBMLE** chooses $(\mu,\Lambda)$ by the marginal likelihood; **URE** minimises
Stein's unbiased risk estimate

$$\mathrm{URE} = \tfrac1J\sum_j\Bigl[-2\operatorname{tr}\bigl((\Lambda+M_j)^{-1}M_jWM_j\bigr)
+ \bigl\|M_j(\Lambda+M_j)^{-1}(y_j-\mu)\bigr\|_W^2\Bigr],$$

which requires no distributional assumption on the effects.

**For forecasting, penalise only the fixed effects** — leave $\rho$ and the slopes
unpenalised — and tune by rolling-origin blocked CV with the one-standard-error
rule.

The finding worth internalising: Anderson–Hsiao dominates on *bias* and loses
badly on *forecast MSE*. Consistency is not the only thing that matters.

---

## 8. Neural lag discovery (`neural`)

AC-GATE makes the effective lag a structural output rather than a post-hoc
explanation:

$$\omega_{i,k} = \operatorname{softmax}_k\Bigl(\frac{g_\theta(z_i)_k - \lambda k/K}{\tau}\Bigr),
\qquad c_{i,t} = \sum_k \omega_{i,k}\tilde X_{i,t-k},
\qquad k^\star_i = \sum_k k\,\omega_{i,k}.$$

Two design choices carry the argument. The position bias is normalised by $K$, so
it means the same thing at any horizon. And the contemporaneous $X_{i,t}$ never
enters the backbone — otherwise the recurrent layer could ignore the gate and
read the answer off today's covariates.

**The audit protocol matters more than the architecture.** Forecast accuracy does
not establish that a learned lag structure is real:

| Layer | Question |
|---|---|
| L0 | Is the model predictively calibrated? |
| L1 | Is $k^\star$ non-degenerate across entities? |
| L2 | Does $k^\star$ align with pre-specified stratifiers, against a permutation null? |
| L3 | Does it recover known truth (synthetic only)? |

Plus a proxy-shuffle negative control: permute proxies across entities and refit.
If L2 alignment survives, it was an artefact of model capacity.

---

## References

Arellano, M. and Bond, S. (1991). *Review of Economic Studies* 58(2), 277–297.

Arellano, M. and Bover, O. (1995). *Journal of Econometrics* 68(1), 29–51.

Belloni, A., Chen, D., Chernozhukov, V. and Hansen, C. (2012). *Econometrica*
80(6), 2369–2429.

Belloni, A., Chernozhukov, V., Hansen, C. and Kozbur, D. (2016). *JBES* 34(4),
590–605.

Blundell, R. and Bond, S. (1998). *Journal of Econometrics* 87(1), 115–143.

Cai, T. T., Liu, W. and Luo, X. (2011). *JASA* 106(494), 594–607.

Chen, S., Chernozhukov, V. and Fernández-Val, I. (2019). *AEA P&P* 109, 77–82.

Chernozhukov, V. et al. (2018). *Econometrics Journal* 21(1), C1–C68.

Chernozhukov, V., Chetverikov, D. and Kato, K. (2013). *Annals of Statistics*
41(6), 2786–2819.

Chernozhukov, V., Fernández-Val, I., Huang, C. and Wang, W. (2024).
Arellano-Bond LASSO estimator for dynamic linear panel models. arXiv:2402.00584.

Chudik, A., Pesaran, M. H. and Yang, J.-C. (2018). *JAE* 33(6), 816–836.

Cornejo, M. and Sosa-Escudero, W. (2026). Machine learning and shrinkage in
dynamic panel forecasting. UdeSA WP 183.

Dhaene, G. and Jochmans, K. (2015). *Review of Economic Studies* 82(3), 991–1030.

Hahn, J. and Kuersteiner, G. (2002). *Econometrica* 70(4), 1639–1657.

Kiviet, J. F. (1995). *Journal of Econometrics* 68(1), 53–78.

Kock, A. B. and Tang, H. (2019). Uniform inference in high-dimensional dynamic
panel data models. *Econometric Theory*.

Kwon, S. (2026). *Econometrica* 94(2), 663–677.

Liu, L., Moon, H. R. and Schorfheide, F. (2020). *Econometrica* 88(1), 171–201.

Moral-Benito, E. (2013). *JBES* 31(4), 451–472.

Mundlak, Y. (1978). *Econometrica* 46(1), 69–85.

Nickell, S. (1981). *Econometrica* 49(6), 1417–1426.

Semenova, V., Goldman, M., Chernozhukov, V. and Taddy, M. (2023). *Quantitative
Economics*.

Sneller, L. (2026). Double machine learning for dynamic panel data.

van de Geer, S., Bühlmann, P., Ritov, Y. and Dezeure, R. (2014). *Annals of
Statistics* 42(3), 1166–1202.

Windmeijer, F. (2005). *Journal of Econometrics* 126(1), 25–51.

Xu, A. (2026). Discovering entity-conditioned lag heterogeneity. arXiv:2605.21542.
