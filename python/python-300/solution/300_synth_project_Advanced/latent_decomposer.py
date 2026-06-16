"""
latent_decomposer.py
====================
**Variational Autoencoder (VAE)-inspired** latent-space decomposition of
the multigroup cross-section library for dimensionality reduction and
uncertainty quantification.

The full energy-dependent cross-section surface sigma(E, T) for a given
material is a high-dimensional object.  We compress it to a low-dimensional
latent vector z in R^d (d << G) using a *deterministic* encoder-decoder
pair inspired by the VAE architecture:

    encoder:  sigma -> (mu_z, log_var_z)  in R^d x R^d
    decoder:  z -> sigma_reconstructed    in R^G

The training is replaced by a *physics-informed* projection onto the
leading PCA-like modes of the cross-section library, with the latent
prior p(z) = N(0, I) enforcing smoothness.

Adapted from seed project:
    * 1172_sabrin1997_AccMLBio-esvlsss -> VAE architecture (_semafovae.py)
"""

from __future__ import annotations
import math
from typing import Dict, List, Tuple

import physics_constants as pc
from cross_sections import MultigroupCrossSection


# ---------------------------------------------------------------------------
# Linear encoder / decoder (PCA-like)
# ---------------------------------------------------------------------------
def gram_schmidt(basis: List[List[float]]) -> List[List[float]]:
    """Orthonormalise a list of vectors via modified Gram-Schmidt."""
    n = len(basis)
    if n == 0:
        return []
    d = len(basis[0])
    out: List[List[float]] = []
    for v in basis:
        w = v[:]
        for u in out:
            proj = sum(wi * ui for wi, ui in zip(w, u))
            w = [wi - proj * ui for wi, ui in zip(w, u)]
        norm = math.sqrt(sum(wi * wi for wi in w))
        if norm < 1.0e-14:
            continue
        out.append([wi / norm for wi in w])
    return out


def build_basis_from_library(
    material: str,
    T_grid_K: List[float],
    n_modes: int = 3,
) -> Tuple[List[List[float]], List[float]]:
    """Build an orthonormal basis for the cross-section library.

    We sample the cross-section surface at each temperature and energy
    group, then apply Gram-Schmidt to the resulting vectors.
    """
    lib = MultigroupCrossSection(material, T_grid_K)
    # each row is sigma_g(T) for a fixed T, over all g
    rows: List[List[float]] = []
    for T in T_grid_K:
        row = lib.spectrum(T)
        # normalise
        norm = math.sqrt(sum(x * x for x in row))
        if norm > pc.EPS_NUMERICAL:
            row = [x / norm for x in row]
        rows.append(row)
    # Gram-Schmidt
    basis = gram_schmidt(rows)
    # take first n_modes
    basis = basis[:n_modes]
    # pad with random-ish orthogonal vectors if needed
    while len(basis) < n_modes:
        # create a new vector orthogonal to all existing
        cand = [1.0 / math.sqrt(pc.N_GROUPS)] * pc.N_GROUPS
        for b in basis:
            proj = sum(ci * bi for ci, bi in zip(cand, b))
            cand = [ci - proj * bi for ci, bi in zip(cand, b)]
        norm = math.sqrt(sum(ci * ci for ci in cand))
        if norm < 1.0e-14:
            # fallback: unit vector along next axis
            cand = [0.0] * pc.N_GROUPS
            cand[len(basis) % pc.N_GROUPS] = 1.0
        else:
            cand = [ci / norm for ci in cand]
        basis.append(cand)
    # mean spectrum
    mean = [0.0] * pc.N_GROUPS
    for row in rows:
        for g in range(pc.N_GROUPS):
            mean[g] += row[g]
    if rows:
        mean = [m / len(rows) for m in mean]
    return basis, mean


# ---------------------------------------------------------------------------
# Encoder: sigma -> latent z
# ---------------------------------------------------------------------------
def encode(
    sigma: List[float],
    basis: List[List[float]],
    mean: List[float],
) -> Tuple[List[float], List[float]]:
    """Encode a cross-section vector sigma into latent (mu_z, log_var_z).

    The encoder is a linear projection:

        mu_z_k = <sigma - mean, basis_k>

    and the log-variance is set to a small constant (deterministic encoder).
    """
    d = len(basis)
    centred = [sigma[g] - mean[g] for g in range(len(sigma))]
    mu_z: List[float] = []
    for k in range(d):
        proj = sum(centred[g] * basis[k][g] for g in range(len(sigma)))
        mu_z.append(proj)
    log_var_z = [-5.0] * d    # small variance
    return mu_z, log_var_z


# ---------------------------------------------------------------------------
# Decoder: z -> sigma
# ---------------------------------------------------------------------------
def decode(
    z: List[float],
    basis: List[List[float]],
    mean: List[float],
) -> List[float]:
    """Decode a latent vector z back to a cross-section vector sigma.

    The decoder is the transpose of the encoder:

        sigma_reconstructed = mean + sum_k z_k basis_k
    """
    G = len(mean)
    recon = mean[:]
    for k, zk in enumerate(z):
        for g in range(G):
            recon[g] += zk * basis[k][g]
    # ensure positivity (cross sections must be >= 0)
    recon = [max(0.0, x) for x in recon]
    return recon


# ---------------------------------------------------------------------------
# VAE loss (reconstruction + KL divergence)
# ---------------------------------------------------------------------------
def vae_loss(
    sigma_orig: List[float],
    sigma_recon: List[float],
    mu_z: List[float],
    log_var_z: List[float],
) -> Dict[str, float]:
    """Compute the VAE loss = reconstruction + KL divergence.

    The reconstruction term is the MSE:

        L_recon = (1/G) sum_g (sigma_g - sigma_recon_g)^2

    The KL divergence from the prior N(0, I) is

        L_KL = -0.5 sum_k (1 + log_var_k - mu_k^2 - exp(log_var_k))
    """
    G = len(sigma_orig)
    mse = sum((sigma_orig[g] - sigma_recon[g]) ** 2 for g in range(G)) / max(G, 1)
    kl = -0.5 * sum(
        1.0 + lv - mu * mu - math.exp(lv)
        for mu, lv in zip(mu_z, log_var_z)
    )
    return {
        "reconstruction_mse": mse,
        "kl_divergence": kl,
        "total_loss": mse + kl,
    }


# ---------------------------------------------------------------------------
# Latent-space interpolation
# ---------------------------------------------------------------------------
def latent_interpolate(
    z1: List[float],
    z2: List[float],
    alpha: float,
) -> List[float]:
    """Linearly interpolate between two latent vectors.

    This allows smooth interpolation between two cross-section sets
    (e.g. at different temperatures) in the low-dimensional latent space.
    """
    if len(z1) != len(z2):
        raise ValueError("latent vectors must have same dimension")
    return [z1[k] + alpha * (z2[k] - z1[k]) for k in range(len(z1))]


# ---------------------------------------------------------------------------
# Full decomposition pipeline
# ---------------------------------------------------------------------------
class LatentDecomposer:
    """VAE-inspired cross-section library compressor.

    Parameters
    ----------
    material : str
        Material name (see MultigroupCrossSection).
    T_grid_K : list of float
        Temperature grid for building the basis.
    n_modes : int
        Number of latent dimensions.
    """

    def __init__(
        self,
        material: str = 'li2o',
        T_grid_K: List[float] = None,
        n_modes: int = 3,
    ) -> None:
        if T_grid_K is None:
            T_grid_K = [300.0, 600.0, 900.0, 1200.0, 1500.0]
        self.material = material
        self.T_grid = T_grid_K
        self.n_modes = n_modes
        self.basis, self.mean = build_basis_from_library(
            material, T_grid_K, n_modes
        )
        self.lib = MultigroupCrossSection(material, T_grid_K)

    def encode_temperature(self, T_K: float) -> Tuple[List[float], List[float]]:
        """Encode the cross-section at temperature T_K into latent space."""
        sigma = self.lib.spectrum(T_K)
        return encode(sigma, self.basis, self.mean)

    def decode_to_spectrum(self, z: List[float]) -> List[float]:
        """Decode a latent vector back to a cross-section spectrum."""
        return decode(z, self.basis, self.mean)

    def reconstruction_quality(self, T_K: float) -> Dict[str, float]:
        """Assess the reconstruction quality at temperature T_K."""
        sigma_orig = self.lib.spectrum(T_K)
        mu_z, log_var_z = self.encode_temperature(T_K)
        sigma_recon = self.decode_to_spectrum(mu_z)
        loss = vae_loss(sigma_orig, sigma_recon, mu_z, log_var_z)
        return loss

    def interpolate_temperatures(
        self, T1_K: float, T2_K: float, alpha: float
    ) -> List[float]:
        """Interpolate between two temperatures in latent space."""
        mu1, _ = self.encode_temperature(T1_K)
        mu2, _ = self.encode_temperature(T2_K)
        z_interp = latent_interpolate(mu1, mu2, alpha)
        return self.decode_to_spectrum(z_interp)
