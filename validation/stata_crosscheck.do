*==============================================================================
* dynpanelai — cross-validation against xtabond2
*
* Purpose: check whether dynpanelai's diff_gmm reproduces xtabond2 on the
* canonical Arellano-Bond employment panel, for exactly matched specifications.
*
* Run:  do validation/stata_crosscheck.do
*
* If xtabond2 is not installed:  ssc install xtabond2
*==============================================================================

clear all
set more off

capture which xtabond2
if _rc {
    display as error "xtabond2 not found. Run: ssc install xtabond2"
    exit 111
}

webuse abdata, clear
xtset id year

display _newline(2) "{hline 78}"
display "SAMPLE"
display "{hline 78}"
count
display "firms: "
quietly levelsof id, local(ids)
display `: word count `ids''

*------------------------------------------------------------------------------
* SPEC A — the headline specification in the dynpanelai README
*
*   dp.diff_gmm(panel, y="n", lags=2,
*               predetermined=["w"], exogenous=["k"],
*               gmm_lags=(2, 4), collapse=True, steps=2)
*
* dynpanelai reports:
*   L1.n  0.0769 (0.1461)      instruments: 7
*   L2.n  0.0171 (0.0576)      Hansen chi2(3) = 2.089, p = 0.554
*   w    -1.6588 (0.6870)      AR(1) z = -1.977, p = 0.048
*   k     0.4051 (0.1173)      AR(2) z = -0.529, p = 0.597
*------------------------------------------------------------------------------
display _newline(2) "{hline 78}"
display "SPEC A: difference GMM, gmm(n w, lag(2 4) collapse), iv(k), twostep"
display "{hline 78}"

xtabond2 n L(1/2).n w k, ///
    gmm(n w, lag(2 4) collapse) ///
    iv(k) ///
    noleveleq twostep robust

*------------------------------------------------------------------------------
* SPEC B — same, uncollapsed
*   dynpanelai reports: L1.n 0.0559, 35 instruments, Hansen p = 0.037
*------------------------------------------------------------------------------
display _newline(2) "{hline 78}"
display "SPEC B: same but UNCOLLAPSED"
display "{hline 78}"

xtabond2 n L(1/2).n w k, ///
    gmm(n w, lag(2 4)) ///
    iv(k) ///
    noleveleq twostep robust

*------------------------------------------------------------------------------
* SPEC C — system GMM, which dynpanelai currently disables.
* This is the reference we need in order to fix it.
*------------------------------------------------------------------------------
display _newline(2) "{hline 78}"
display "SPEC C: SYSTEM GMM (disabled in dynpanelai; this is the target)"
display "{hline 78}"

xtabond2 n L(1/2).n w k, ///
    gmm(n w, lag(2 4) collapse) ///
    iv(k) ///
    twostep robust

*------------------------------------------------------------------------------
* SPEC D — native xtabond, as a second opinion on the difference estimator
*------------------------------------------------------------------------------
display _newline(2) "{hline 78}"
display "SPEC D: native xtabond, twostep"
display "{hline 78}"

xtabond n w k, lags(2) maxldep(3) twostep vce(robust)

*------------------------------------------------------------------------------
* SPEC E — one-step, to isolate whether any gap is in the weight matrix
*------------------------------------------------------------------------------
display _newline(2) "{hline 78}"
display "SPEC E: SPEC A but one-step"
display "{hline 78}"

xtabond2 n L(1/2).n w k, ///
    gmm(n w, lag(2 4) collapse) ///
    iv(k) ///
    noleveleq robust

display _newline(2) "{hline 78}"
display "DONE — compare SPEC A against the dynpanelai numbers in the header."
display "{hline 78}"
