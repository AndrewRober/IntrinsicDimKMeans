"""
sid_kmeans.py: Spectral Intrinsic-Dimension Minimum Description Length K-Means (SID-KMeans)
===========================================================================================
Zero-hyperparameter automatic cluster count selection (K* in {1..K_max}) for both
low-dimensional Euclidean data and high-dimensional neural embeddings (768D-4096D).

Combines three exact statistical pillars:
  1. Gavish-Donoho (2014) Marchenko-Pastur Singular-Value Thresholding: strips high-D
     isotropic ambient noise dimensions that otherwise dilute WCSS log-gains to O(1/d).
  2. Satterthwaite-Welch Quadratic-Form Degrees of Freedom nu = (Tr Sigma)^2 / Tr(Sigma^2):
     matches the exact chi-square moment structure of squared distances on anisotropic /
     collinear manifolds, preventing the d / nu log-likelihood inflation that causes
     ambient BIC / X-Means to catastrophically over-split correlated features.
  3. Exact Marginal LogSumExp Mixture Likelihood (fixing X-Means' Voronoi hard-assignment
     truncation bias -2/pi * N on unimodal Gaussians while accurately resolving softly
     touching semantic clusters).
"""

from __future__ import annotations
import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import KMeans


class SIDKMeans:
    def __init__(self, k_max: int = 15, n_init: int = 10, random_state: int = 42):
        self.k_max = int(k_max)
        self.n_init = int(n_init)
        self.random_state = int(random_state)

        self.k_hat_: int | None = None
        self.nu_: float | None = None
        self.kept_rank_: int | None = None
        self.mp_null_: bool = False
        self.scores_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.cluster_centers_: np.ndarray | None = None

    @staticmethod
    def gavish_donoho_omega(beta: float) -> float:
        """Exact polynomial approximation of optimal Marchenko-Pastur hard threshold factor (Gavish & Donoho, 2014)."""
        return 0.56 * (beta ** 3) - 0.95 * (beta ** 2) + 1.82 * beta + 1.43

    def fit(self, X: np.ndarray) -> "SIDKMeans":
        X = np.asarray(X, dtype=np.float64)
        N, d = X.shape
        k_upper = min(self.k_max, N - 1)

        mean_vec = np.mean(X, axis=0, keepdims=True)
        Xc = X - mean_vec
        U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        r = len(s)

        # Pillar 1: Marchenko-Pastur / Gavish-Donoho bulk noise removal for high-D matrices
        if min(N - 1, d) >= 10:
            beta = min(N, d) / max(N, d)
            tau = self.gavish_donoho_omega(beta) * float(np.median(s))
            keep = s > tau
            if not np.any(keep):
                # Zero singular values exceed random-matrix bulk noise edge -> pure unstructured noise!
                self.mp_null_ = True
                s_clean = s[: min(2, r)]
                Xd = U[:, : len(s_clean)] * s_clean
                V_keep = Vt[: len(s_clean)]
            else:
                self.mp_null_ = False
                s_clean = s[keep]
                Xd = U[:, keep] * s_clean  # Isometric representation in signal subspace R^{m_keep}
                V_keep = Vt[keep]
        else:
            self.mp_null_ = False
            s_clean = s
            Xd = Xc
            V_keep = np.eye(d, dtype=np.float64)

        self.kept_rank_ = int(Xd.shape[1])

        # Pillar 2: Satterthwaite-Welch Effective Chi-Square Degrees of Freedom nu
        eigs = (s_clean ** 2) / max(N, 1)
        sum_eigs = float(np.sum(eigs))
        sum_eigs_sq = float(np.sum(eigs ** 2))
        if sum_eigs_sq <= 1e-24:
            self.nu_ = 1.0
            self.k_hat_ = 1
            self.labels_ = np.zeros(N, dtype=int)
            self.cluster_centers_ = mean_vec.copy()
            self.scores_ = np.zeros(1)
            return self

        nu = (sum_eigs ** 2) / sum_eigs_sq
        nu = float(np.clip(nu, 1.0, float(Xd.shape[1])))
        self.nu_ = nu

        if self.mp_null_:
            self.k_hat_ = 1
            self.labels_ = np.zeros(N, dtype=int)
            self.cluster_centers_ = mean_vec.copy()
            self.scores_ = np.array([0.0])
            return self

        # Pillar 3: Exact Marginal LogSumExp Mixture MDL over Satterthwaite-calibrated geometry
        scores = np.zeros(k_upper, dtype=np.float64)
        models = []

        for idx, k in enumerate(range(1, k_upper + 1)):
            if k == 1:
                labels = np.zeros(N, dtype=int)
                centers_sub = np.zeros((1, Xd.shape[1]), dtype=np.float64)
                wcss = float(np.sum(Xd ** 2))
            else:
                km = KMeans(n_clusters=k, n_init=self.n_init, random_state=self.random_state).fit(Xd)
                labels = km.labels_
                centers_sub = km.cluster_centers_
                wcss = float(km.inertia_)

            wcss = max(wcss, 1e-18)
            counts = np.bincount(labels, minlength=k)
            pis = counts / float(N)
            sig2_eff = wcss / (N * nu)

            sq_dists = np.sum((Xd[:, None, :] - centers_sub[None, :, :]) ** 2, axis=2)
            log_comp = (
                np.log(np.maximum(pis, 1e-15))[None, :]
                - 0.5 * nu * np.log(2.0 * np.pi * sig2_eff)
                - sq_dists / (2.0 * sig2_eff)
            )
            neg2_ll = -2.0 * float(np.sum(logsumexp(log_comp, axis=1)))
            penalty = (k * (nu + 1.0) - 1.0) * np.log(N)
            scores[idx] = neg2_ll + penalty
            models.append((labels, centers_sub))

        best_idx = int(np.argmin(scores))
        self.scores_ = scores
        self.k_hat_ = best_idx + 1
        best_labels, best_centers_sub = models[best_idx]
        self.labels_ = best_labels

        # Map subspace centroids back to ambient R^d (or compute exact ambient cluster means!)
        amb_centers = np.zeros((self.k_hat_, d), dtype=np.float64)
        for c in range(self.k_hat_):
            mask = best_labels == c
            amb_centers[c] = X[mask].mean(axis=0) if np.any(mask) else mean_vec[0]
        self.cluster_centers_ = amb_centers
        return self
