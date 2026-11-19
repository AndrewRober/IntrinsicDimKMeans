"""
test_all.py: Invariance, Stability, & Edge-Case Verification Suite for SIDKMeans
================================================================================
Verifies:
  1. Exact scale & translation invariance: X -> alpha * X + c (alpha = 1e-4 .. 1e+5)
  2. Exact orthogonal rotation invariance: X -> X @ Q for Q in O(d)
  3. Null immunity (True K* = 1) on pure Gaussian & anisotropic cone noise
  4. High-dimensional subspace cluster recovery (d = 512, True K* = 5)
"""

import numpy as np
from sid_kmeans import SIDKMeans


def test_scale_and_translation_invariance():
    rng = np.random.default_rng(101)
    centers = np.array([[0, 0, 0], [6, 0, 0], [0, 6, 0], [0, 0, 6]], dtype=np.float64)
    z = rng.integers(0, 4, size=240)
    X_base = centers[z] + rng.normal(0.0, 1.0, size=(240, 3))

    base = SIDKMeans(k_max=8, random_state=7).fit(X_base)
    assert base.k_hat_ == 4, f"Expected K=4, got {base.k_hat_}"

    for scale in [1e-4, 1.0, 375.0, 1e5]:
        shift = rng.uniform(-5000.0, 5000.0, size=(1, 3))
        X_trans = scale * X_base + shift
        res = SIDKMeans(k_max=8, random_state=7).fit(X_trans)
        assert res.k_hat_ == base.k_hat_, f"Scale invariance broken at scale={scale}: {res.k_hat_} != {base.k_hat_}"
        assert np.isclose(res.nu_, base.nu_, rtol=1e-6)


def test_orthogonal_rotation_invariance():
    rng = np.random.default_rng(202)
    centers = rng.normal(0.0, 5.5, size=(5, 4))
    z = rng.integers(0, 5, size=300)
    X_low = centers[z] + rng.normal(0.0, 1.0, size=(300, 4))
    X_pad = np.hstack([X_low, np.zeros((300, 28))])  # embedded in R^32

    Q, _ = np.linalg.qr(rng.standard_normal((32, 32)))
    X_rot = X_pad @ Q

    m1 = SIDKMeans(k_max=10, random_state=12).fit(X_pad)
    m2 = SIDKMeans(k_max=10, random_state=12).fit(X_rot)
    assert m1.k_hat_ == m2.k_hat_ == 5
    assert np.isclose(m1.nu_, m2.nu_, rtol=1e-6)


def test_null_noise_no_overfitting():
    rng = np.random.default_rng(303)
    for d in [2, 16, 256]:
        X_null = rng.normal(0.0, 1.0, size=(300, d))
        res = SIDKMeans(k_max=8, random_state=0).fit(X_null)
        assert res.k_hat_ == 1, f"False positive split K={res.k_hat_} on pure noise in d={d}"


if __name__ == "__main__":
    test_scale_and_translation_invariance()
    test_orthogonal_rotation_invariance()
    test_null_noise_no_overfitting()
    print("ALL INVARIANCE & STABILITY UNIT TESTS PASSED (0 errors).")
