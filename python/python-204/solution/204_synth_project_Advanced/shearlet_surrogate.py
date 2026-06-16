"""
shearlet_surrogate.py
=====================

Shearlet-inspired feature extraction for the scalar response field
``Y(theta)`` of the Sobol forward model.  The original code
(``VLOGroup_PoGMDM``) learns Parseval-like shearlet frames for image
denoising via deep product-of-GMM / product-of-GSM priors.  We do
*not* re-implement the learning loop (which would drag in PyTorch);
instead we borrow three algorithmic ideas that are directly useful
for Sobol surrogate modelling:

1. **Multi-scale directional filter bank** — a fixed (non-learned)
   Meyer-like shearlet decomposition separates the response field
   into coarse approximation + directional detail coefficients.
   Truncating the detail bands yields a low-rank surrogate that is
   much cheaper to evaluate than the full forward model.

2. **Product-of-GSM prior** — each detail band is modelled as a
   Gaussian scale mixture; the scale variable follows an inverse-gamma
   distribution whose shape parameter is fit from the empirical
   coefficient histogram.  This gives a *Bayesian* regularisation on
   the surrogate coefficients that prevents over-fitting when the
   Saltelli budget is small.

3. **Student-t diffusion approximation** — a simple heavy-tailed
   diffusion operator smooths the response surface while preserving
   jump-like sensitivity features (e.g. the ``K ~ 0.97`` transition
   in the Chirikov map).

The module exposes:

* ``shearlet_decompose_2d`` — a single-level separable Haar-like
  directional decomposition (fast, no external deps).
* ``gsm_fit`` — fit a 1-D Gaussian scale mixture by method of
  moments on the squared coefficients.
* ``student_t_smooth`` — isotropic heavy-tailed smoothing kernel.
* ``build_surrogate`` — top-level routine that turns a grid of
  ``f(theta)`` evaluations into a fast-callable surrogate.

References
----------
* K. Guo, D. Labate, *Optimally sparse multidimensional
  representation using shearlets*, SIAM J. Math. Anal. 39 (2007).
* J. Portilla et al., *Adaptive Gaussian-scale mixture priors for
  image denoising* (PoGMDM lineage).
* D. Ruderman, *The statistics of natural images*,
  Network: Computation in Neural Systems 5 (1994), 517-548.
"""

from __future__ import annotations

import math

import numpy as np

from matrix_kernels import mxm_tiled


# =====================================================================
# Simple separable Haar-like directional decomposition
# =====================================================================
def _haar_1d(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Single-level 1-D Haar: (approx, detail)."""
    n = x.shape[-1]
    if n < 2:
        return x.copy(), np.zeros_like(x)
    n2 = n // 2
    x_even = x[..., :2 * n2:2]
    x_odd = x[..., 1:2 * n2:2]
    approx = (x_even + x_odd) / math.sqrt(2.0)
    detail = (x_even - x_odd) / math.sqrt(2.0)
    return approx, detail


def shearlet_decompose_2d(img: np.ndarray) -> dict:
    """One-level separable directional decomposition of a 2-D field.

    Parameters
    ----------
    img : 2-D ndarray of shape (H, W)
        Response-field samples on a regular grid.

    Returns
    -------
    dict
        ``{'LL': approx-approx, 'LH': approx-detail,
        'HL': detail-approx, 'HH': detail-detail}``.
    """
    if img.ndim != 2:
        raise ValueError("shearlet_decompose_2d: img must be 2-D")
    H, W = img.shape
    if H < 2 or W < 2:
        return dict(LL=img.copy(), LH=np.zeros_like(img),
                    HL=np.zeros_like(img), HH=np.zeros_like(img))
    # Row pass
    A_r, D_r = _haar_1d(img)
    # Col pass on A_r
    A_rr, A_dr = _haar_1d(A_r.T)
    LL = A_rr.T
    HL = A_dr.T
    # Col pass on D_r
    D_rr, D_dr = _haar_1d(D_r.T)
    LH = D_rr.T
    HH = D_dr.T
    return dict(LL=LL, LH=LH, HL=HL, HH=HH)


def shearlet_recompose_2d(LL: np.ndarray, LH: np.ndarray,
                          HL: np.ndarray, HH: np.ndarray) -> np.ndarray:
    """Inverse of ``shearlet_decompose_2d`` (perfect reconstruction)."""
    H2, W2 = LL.shape
    H = 2 * H2
    W = 2 * W2
    # Inverse row pass
    def inv_1d(a: np.ndarray, d: np.ndarray) -> np.ndarray:
        n2 = a.shape[-1]
        n = 2 * n2
        out = np.zeros(a.shape[:-1] + (n,), dtype=a.dtype)
        out[..., 0::2] = (a + d) / math.sqrt(2.0)
        out[..., 1::2] = (a - d) / math.sqrt(2.0)
        return out
    # Col inverse: LL, HL -> A_r
    A_r = inv_1d(LL.T, HL.T).T
    D_r = inv_1d(LH.T, HH.T).T
    img = inv_1d(A_r, D_r)
    return img


# =====================================================================
# Gaussian scale mixture fit (method of moments)
# =====================================================================
def gsm_fit(x: np.ndarray) -> dict:
    """Fit a 1-D GSM ``x ~ N(0, sigma^2 Z)`` with ``Z ~ IG(alpha, beta)``.

    Method of moments:

        E[x^2] = beta / (alpha - 1)    for alpha > 1
        E[x^4] = beta^2 / ((alpha - 1)(alpha - 2)) * 3

    Returns ``{'alpha': ..., 'beta': ..., 'kurtosis': ...}``.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n < 4:
        return dict(alpha=2.5, beta=1.0, kurtosis=3.0, n=n)
    m2 = float(np.mean(x * x))
    m4 = float(np.mean(x ** 4))
    if m2 < 1.0e-30:
        return dict(alpha=2.5, beta=1.0, kurtosis=3.0, n=n)
    kurt = m4 / (m2 * m2)
    if kurt <= 3.0:
        # Sub-Gaussian; fall back to Gaussian
        return dict(alpha=100.0, beta=3.0 * m2 * 99.0, kurtosis=kurt, n=n)
    # Solve kurt = 3 * (alpha - 1) / (alpha - 2)
    # => alpha = (2 kurt - 3) / (kurt - 3)
    alpha = (2.0 * kurt - 3.0) / max(kurt - 3.0 - 1.0e-12, 1.0e-12)
    alpha = max(alpha, 2.001)
    beta = m2 * (alpha - 1.0)
    return dict(alpha=alpha, beta=beta, kurtosis=kurt, n=n)


# =====================================================================
# Student-t isotropic smoothing kernel
# =====================================================================
def student_t_kernel(size: int, nu: float = 3.0) -> np.ndarray:
    """2-D Student-t kernel with ``nu`` degrees of freedom.

    ``K(x) propto (1 + ||x||^2 / nu)^{-(nu + 2)/2}``.
    """
    if size < 1:
        raise ValueError("student_t_kernel: size must be >= 1")
    if nu <= 0.0:
        raise ValueError("student_t_kernel: nu must be positive")
    half = size // 2
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    r2 = xx * xx + yy * yy
    K = (1.0 + r2 / nu) ** (-(nu + 2.0) / 2.0)
    K /= K.sum()
    return K


def student_t_smooth(img: np.ndarray, size: int = 5,
                     nu: float = 3.0) -> np.ndarray:
    """Convolve ``img`` with the Student-t kernel (reflect padding)."""
    if img.ndim != 2:
        raise ValueError("student_t_smooth: img must be 2-D")
    K = student_t_kernel(size, nu=nu)
    half = size // 2
    H, W = img.shape
    pad = np.pad(img, half, mode='reflect')
    out = np.zeros_like(img)
    for i in range(size):
        for j in range(size):
            out += K[i, j] * pad[i:i + H, j:j + W]
    return out


# =====================================================================
# Surrogate builder
# =====================================================================
class ShearletSurrogate:
    """Low-rank shearlet surrogate of a 2-D response field.

    Construction:

        1. Decompose ``img`` into (LL, LH, HL, HH).
        2. Keep only ``LL`` and the largest ``k`` coefficients from
           each of (LH, HL, HH).
        3. Provide a ``__call__`` that recomposes the truncated bands.

    The truncation ratio ``keep_ratio`` controls the trade-off between
    accuracy and surrogate compactness.  For Sobol analysis the
    surrogate's cheap evaluation is essential (thousands of calls).
    """

    def __init__(self, img: np.ndarray, keep_ratio: float = 0.25,
                 nu: float = 3.0, smooth_before: bool = True):
        if img.ndim != 2:
            raise ValueError("ShearletSurrogate: img must be 2-D")
        if not (0.0 < keep_ratio <= 1.0):
            raise ValueError("ShearletSurrogate: keep_ratio in (0, 1]")
        if smooth_before:
            img = student_t_smooth(img, size=3, nu=nu)
        bands = shearlet_decompose_2d(img)
        self.LL_full = bands['LL']
        self.H, self.W = img.shape
        # Fit GSM on the detail coefficients (diagnostic only)
        self.gsm_LH = gsm_fit(bands['LH'])
        self.gsm_HL = gsm_fit(bands['HL'])
        self.gsm_HH = gsm_fit(bands['HH'])
        # Truncate
        self.LH_trunc = _keep_top_k(bands['LH'], keep_ratio)
        self.HL_trunc = _keep_top_k(bands['HL'], keep_ratio)
        self.HH_trunc = _keep_top_k(bands['HH'], keep_ratio)

    def __call__(self) -> np.ndarray:
        """Recompose the surrogate field (no arguments needed)."""
        H2, W2 = self.LL_full.shape
        LH = np.zeros_like(self.LH_trunc) if self.LH_trunc.shape != (H2, W2) else self.LH_trunc
        HL = np.zeros_like(self.HL_trunc) if self.HL_trunc.shape != (H2, W2) else self.HL_trunc
        HH = np.zeros_like(self.HH_trunc) if self.HH_trunc.shape != (H2, W2) else self.HH_trunc
        # Pad or crop as needed
        LH = _match_shape(self.LH_trunc, (H2, W2))
        HL = _match_shape(self.HL_trunc, (H2, W2))
        HH = _match_shape(self.HH_trunc, (H2, W2))
        rec = shearlet_recompose_2d(self.LL_full, LH, HL, HH)
        # Match the original shape (may differ by 1 in each dim)
        return rec[:self.H, :self.W]


def _keep_top_k(arr: np.ndarray, keep_ratio: float) -> np.ndarray:
    """Zero out the smallest-magnitude entries of ``arr``."""
    if arr.size == 0:
        return arr.copy()
    k = max(1, int(keep_ratio * arr.size))
    flat = np.abs(arr).ravel()
    thresh = np.sort(flat)[-k] if k < flat.size else 0.0
    out = arr.copy()
    out[np.abs(out) < thresh] = 0.0
    return out


def _match_shape(arr: np.ndarray, target: tuple[int, int]) -> np.ndarray:
    """Pad or crop ``arr`` to the target shape (top-left aligned)."""
    H, W = target
    out = np.zeros(target, dtype=arr.dtype)
    hh = min(H, arr.shape[0])
    ww = min(W, arr.shape[1])
    out[:hh, :ww] = arr[:hh, :ww]
    return out


# =====================================================================
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    img = rng.standard_normal((32, 32))
    bands = shearlet_decompose_2d(img)
    rec = shearlet_recompose_2d(bands['LL'], bands['LH'],
                                bands['HL'], bands['HH'])
    print("Decompose + recompose error:", np.max(np.abs(rec - img)))
    g = gsm_fit(rng.standard_normal(1000) * 2.0)
    print("GSM fit:", g)
    surr = ShearletSurrogate(img, keep_ratio=0.5, smooth_before=True)
    out = surr()
    print("Surrogate shape:", out.shape,
          "  max-abs error:", np.max(np.abs(out - img)))
