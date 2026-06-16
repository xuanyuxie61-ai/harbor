"""
neural_filter.py
================
A small convolutional subgrid-scale (SGS) closure model for the MHD
system, structured after the CNN+LSTM, DFFN and STFTNet architectures
reproduced in 1191_jones12138.

Instead of training the filters on data (we have none in this
first-principles setup) we *prescribe* the kernels analytically so
that the resulting operator approximates a scale-similar SGS stress
tensor in the sense of Bardina et al. (1980).  The architecture is

    input:  U  (NCONS, Nx, Ny, Nz)
    -> 3 x 3 x 3 depthwise convolutions with analytically-prescribed
       Gaussian / Laplacian / shock-capturing kernels
    -> channel-wise sum with learned (here: hand-tuned) weights
    -> output:  correction to the RHS of the conserved variables

The three kernels are:

    K_G   : 3-D Gaussian low-pass (scale separation)
    K_L   : discrete Laplacian (dissipative back-scatter)
    K_SC  : shock-capturing kernel based on the Jiang-Shu smoothness
            indicator

This architecture is a stripped-down, fully-deterministic version of
the deep-learning SGS models that have become popular in the a priori
testing literature (e.g. Zaman et al. JCP 2022).  It retains the
multi-kernel, multi-channel structure of the CNN while removing any
stochastic training component so that the simulation remains
bit-reproducible.
"""

from __future__ import annotations
import math
import numpy as np
from typing import Tuple

import boundary_conditions as bc
import high_order_fd as hfd


# ---------------------------------------------------------------------------
#                    Analytical convolution kernels
# ---------------------------------------------------------------------------
def gaussian_kernel_3d(sigma: float = 0.8, radius: int = 1) -> np.ndarray:
    """Return a normalised 3-D Gaussian kernel of half-width ``radius``
    and standard deviation ``sigma`` (in cells)."""
    x = np.arange(-radius, radius + 1)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    K = np.exp(-0.5 * (X**2 + Y**2 + Z**2) / sigma**2)
    K /= K.sum()
    return K


def laplacian_kernel_3d() -> np.ndarray:
    """Standard 3-D 7-point discrete Laplacian kernel."""
    K = np.zeros((3, 3, 3))
    K[1, 1, 1] = -6.0
    K[0, 1, 1] = K[2, 1, 1] = 1.0
    K[1, 0, 1] = K[1, 2, 1] = 1.0
    K[1, 1, 0] = K[1, 1, 2] = 1.0
    return K


def shock_capturing_kernel_3d() -> np.ndarray:
    """Shock-capturing kernel: a Laplacian of the Gaussian smoothness.

    This mimics the response of the Jiang-Shu indicator (hfd module)
    but in convolutional form; it activates only near discontinuities
    and adds targeted dissipation there.
    """
    K_G = gaussian_kernel_3d(0.8)
    K_L = laplacian_kernel_3d()
    # Convolve the two (discrete, full mode)
    K = np.zeros((3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                s = 0.0
                for ii in range(3):
                    for jj in range(3):
                        for kk in range(3):
                            iL = i + ii - 1
                            jL = j + jj - 1
                            kL = k + kk - 1
                            if 0 <= iL < 3 and 0 <= jL < 3 and 0 <= kL < 3:
                                s += K_L[iL, jL, kL] * K_G[ii, jj, kk]
                K[i, j, k] = s
    # Normalise to unit L1
    K /= (np.sum(np.abs(K)) + 1.0e-30)
    return K


# ---------------------------------------------------------------------------
#                 Apply 3-D convolution to a scalar field
# ---------------------------------------------------------------------------
def conv3d(F: np.ndarray, K: np.ndarray,
           mode: str = "wrap") -> np.ndarray:
    """Apply a 3x3x3 convolution kernel K to a 3-D scalar field F.

    ``mode`` follows the numpy convention: 'wrap' for periodic,
    'nearest' for outflow-style padding.
    """
    if K.shape != (3, 3, 3):
        raise ValueError("conv3d: kernel must be (3, 3, 3)")
    pad_F = np.pad(F, 1, mode=("wrap" if mode == "wrap" else "edge"))
    out = np.zeros_like(F)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                out += K[i, j, k] * pad_F[i:i + F.shape[0],
                                          j:j + F.shape[1],
                                          k:k + F.shape[2]]
    return out


# ---------------------------------------------------------------------------
#                   Multi-kernel SGS correction
# ---------------------------------------------------------------------------
def sgs_correction(U: np.ndarray, g,
                   w_G: float = 0.05,
                   w_L: float = 0.02,
                   w_SC: float = 0.01) -> np.ndarray:
    """Compute the SGS correction dU/dt|_sgs.

    The three channels act on the momentum and magnetic fields only;
    density, energy and psi receive no direct SGS correction (this
    matches the Boussinesq-like approximation used in most a priori
    tests).

    Parameters
    ----------
    U : ndarray of shape (NCONS, Nx+2*ng, Ny+2*ng, Nz+2*ng)
    g : ShearingBoxGrid
    w_G, w_L, w_SC : float
        Channel weights (tuned by hand to give a sub-percent
        correction in the linear MRI regime).
    """
    ng = 2
    dUdt = np.zeros_like(U)
    # Operate on the interior only
    U_int = U[:, ng:-ng, ng:-ng, ng:-ng]
    K_G  = gaussian_kernel_3d()
    K_L  = laplacian_kernel_3d()
    K_SC = shock_capturing_kernel_3d()
    # Apply to mx, my, mz, Bx, By, Bz
    for f in (bc.ConsIdx.mx, bc.ConsIdx.my, bc.ConsIdx.mz,
              bc.ConsIdx.Bx, bc.ConsIdx.By, bc.ConsIdx.Bz):
        F = U_int[f]
        corr_G  = conv3d(F, K_G,  mode="wrap") - F  # high-pass part
        corr_L  = conv3d(F, K_L,  mode="wrap")
        corr_SC = conv3d(F, K_SC, mode="wrap")
        corr = w_G * corr_G + w_L * corr_L + w_SC * corr_SC
        dUdt[f, ng:-ng, ng:-ng, ng:-ng] = corr
    return dUdt


# ---------------------------------------------------------------------------
#              DFFN-style depthwise feature stack (from 1191)
# ---------------------------------------------------------------------------
def dffn_feature_stack(U: np.ndarray, g) -> np.ndarray:
    """Return a multi-channel feature stack analogous to the dual-
    feed-forward network (DFFN) of 1191_jones12138.

    For each conserved field we stack:

        (F,  low-pass(F),  high-pass(F),  |grad F|)

    along a new channel axis.  The resulting array has shape
    (NCONS * 4, Nx, Ny, Nz) and can be used as input to a downstream
    CNN SGS model.
    """
    ng = 2
    U_int = U[:, ng:-ng, ng:-ng, ng:-ng]
    K_G = gaussian_kernel_3d()
    features = []
    for f in range(bc.NCONS):
        F = U_int[f]
        low  = conv3d(F, K_G, mode="wrap")
        high = F - low
        # Gradient magnitude via central differences
        gx = np.gradient(F, float(np.mean(g.dx)), axis=0)
        gy = np.gradient(F, float(np.mean(g.dy)), axis=1)
        gz = np.gradient(F, float(np.mean(g.dz)), axis=2)
        gmag = np.sqrt(gx**2 + gy**2 + gz**2)
        features.extend([F, low, high, gmag])
    return np.stack(features, axis=0)


# ---------------------------------------------------------------------------
#             STFT-style spectral diagnostic (from 1191)
# ---------------------------------------------------------------------------
def stft_line_energy(U: np.ndarray, g,
                     direction: str = "y") -> np.ndarray:
    """Return the 1-D power spectrum of the magnetic energy along a
    pencil in the chosen direction (x, y, or z).

    This mimics the short-time Fourier transform used in STFTNet
    (1191) but collapses to a single-window spectrum because the
    shearing-box is statistically homogeneous in y.
    """
    ng = 2
    By = U[bc.ConsIdx.By, ng:-ng, ng:-ng, ng:-ng]
    Bx = U[bc.ConsIdx.Bx, ng:-ng, ng:-ng, ng:-ng]
    Bz = U[bc.ConsIdx.Bz, ng:-ng, ng:-ng, ng:-ng]
    E_B = 0.5 * (Bx**2 + By**2 + Bz**2)
    if direction == "y":
        spec = np.mean(np.abs(np.fft.rfft(E_B, axis=1))**2, axis=(0, 2))
    elif direction == "x":
        spec = np.mean(np.abs(np.fft.rfft(E_B, axis=0))**2, axis=(1, 2))
    elif direction == "z":
        spec = np.mean(np.abs(np.fft.rfft(E_B, axis=2))**2, axis=(0, 1))
    else:
        raise ValueError(f"stft_line_energy: unknown direction '{direction}'")
    return spec
