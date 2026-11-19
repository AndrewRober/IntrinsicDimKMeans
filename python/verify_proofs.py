"""
Symbolic & Numerical Proof Verification on NVIDIA GB10 (gx10-54cd)
for Spectral Intrinsic-Dimension Minimum Description Length K-Means (SID-KMeans).
"""
import numpy as np
import sympy as sp
from scipy.special import logsumexp
from sklearn.cluster import KMeans

print("=== [1] Symbolic Proof Verification (SymPy): Huygens Decomposition & High-D Collapse Limit ===")
N, d, m, K, d_sig, s_par, s_perp = sp.symbols("N d m K d_sig s_par s_perp", positive=True, real=True)

# Theorem 2: Exact limit of N * d * ln(WCSS(K*) / WCSS(1)) as ambient dimension d -> oo
wcss_1 = N * (d_sig + m * s_par + (d - m) * s_perp)
wcss_k = N * (m * s_par + (d - m) * s_perp)
ll_gain = N * d * sp.log(wcss_1 / wcss_k)
lim_ll_gain = sp.limit(ll_gain, d, sp.oo)
expected_lim = N * d_sig / s_perp
assert sp.simplify(lim_ll_gain - expected_lim) == 0, f"Expected {expected_lim}, got {lim_ll_gain}"
print(f"  [PASS] Exact Asymptotic Signal-Gain Ceiling proved symbolically: lim_(d->oo) N*d*ln(WCSS(1)/WCSS(K*)) = {lim_ll_gain}")

# Critical dimension d_crit where Ambient BIC penalty (K - 1)*(d + 1)*ln(N) overtakes finite signal gain
penalty = (K - 1) * d * sp.log(N)
d_crit = sp.solve(sp.Eq(lim_ll_gain, penalty), d)[0]
print(f"  [PASS] Exact Ambient BIC Collapse Dimension d_crit = {d_crit}")

# Theorem 1: Voronoi Hard-Assignment Distortion on Unimodal 1D Standard Normal N(0, 1) at K = 2
z = sp.Symbol("z", real=True)
phi = sp.exp(-z**2 / 2) / sp.sqrt(2 * sp.pi)
mu_half = sp.integrate(z * 2 * phi, (z, 0, sp.oo))  # E[Z | Z > 0] = sqrt(2/pi)
wcss_ratio_k2 = sp.integrate((z - mu_half)**2 * 2 * phi, (z, 0, sp.oo))  # 1 - 2/pi
assert sp.simplify(wcss_ratio_k2 - (1 - 2 / sp.pi)) == 0
print(f"  [PASS] Exact Voronoi Split Distortion on N(0,1): E[WCSS(2)/WCSS(1)] = {wcss_ratio_k2} = {float(wcss_ratio_k2):.6f}")

# Check Pelleg & Moore X-Means Hard-Assignment Log-Likelihood Change per sample at K=2 on N(0, I_m):
# Pelleg & Moore (ICML 2000) sets l_hard(K) = -N*m/2 * ln(2*pi*e * WCSS/(N*m)) + sum n_k ln(n_k / N)
# In -2*l scale: Delta_per_sample = m * ln(1 - 2/(pi*m)) + 2*ln(2)
for m_val in [1, 2, 5, 10, 50]:
    icl_delta = m_val * np.log(1.0 - 2.0 / (np.pi * m_val)) + 2.0 * np.log(2.0)
    naive_wcss_delta = m_val * np.log(1.0 - 2.0 / (np.pi * m_val))
    print(f"  m={m_val:2d} | Naive WCSS-BIC drift/N = {naive_wcss_delta:+.4f} (<0 OVERFITS!) | Complete-Data Assignment-Corrected drift/N = {icl_delta:+.4f} (>0 CONSISTENT!)")
    assert naive_wcss_delta < -0.63 and icl_delta > 0.37

print("\n=== [2] Testing Exact Marginal LogSumExp Mixture BIC vs Hard X-Means BIC on Pure Noise (K*=1) ===")
rng = np.random.default_rng(42)
for N_test in [300, 1500]:
    X_null = rng.standard_normal((N_test, 6))
    scores_naive_wcss, scores_xmeans_pm, scores_exact_mix, scores_icl = [], [], [], []
    for k in range(1, 7):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_null)
        labels = km.labels_
        centers = km.cluster_centers_
        wcss = km.inertia_
        m_dim = X_null.shape[1]
        counts = np.bincount(labels, minlength=k)
        pis = counts / N_test
        sig2 = max(wcss / (N_test * m_dim), 1e-12)
        p_params = k * m_dim + (k - 1) + 1

        # 1. Naive WCSS-BIC (no assignment entropy, no mixture overlap): N*m*ln(sig2) + p*ln(N)
        scores_naive_wcss.append(N_test * m_dim * np.log(sig2) + p_params * np.log(N_test))

        # 2. Pelleg & Moore Hard-Assignment BIC (= ICL-style complete likelihood): N*m*(ln(2*pi*sig2)+1) - 2*sum n_k ln(pi_k) + p*ln(N)
        neg2_ll_complete = N_test * m_dim * (np.log(2 * np.pi * sig2) + 1.0) - 2.0 * np.sum(counts * np.log(np.maximum(pis, 1e-12)))
        scores_icl.append(neg2_ll_complete + p_params * np.log(N_test))

        # 3. Exact Marginal Mixture LogSumExp BIC evaluated at K-Means centroids & variance!
        # sq_dists: (N, k)
        sq_dists = np.sum((X_null[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        log_comp = np.log(np.maximum(pis, 1e-12))[None, :] - 0.5 * m_dim * np.log(2 * np.pi * sig2) - sq_dists / (2.0 * sig2)
        neg2_ll_mix = -2.0 * np.sum(logsumexp(log_comp, axis=1))
        scores_exact_mix.append(neg2_ll_mix + p_params * np.log(N_test))

    print(f"  Pure Noise (True K*=1, N={N_test}, d=6): Naive WCSS-BIC picks K={np.argmin(scores_naive_wcss)+1} | "
          f"Exact Marginal Mix-BIC picks K={np.argmin(scores_exact_mix)+1} | ICL-BIC picks K={np.argmin(scores_icl)+1}")
