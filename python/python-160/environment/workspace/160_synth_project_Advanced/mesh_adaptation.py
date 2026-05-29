"""
mesh_adaptation.py
==================
Non-uniform mesh adaptation for reactor simulation zones.

Incorporates algorithms from:
  - 245_cvt_1d_nonuniform (CVT with non-uniform density)

Scientific role:
  Generates adaptive computational meshes for the 1D reactor model.
  Mesh density is concentrated in regions with high gradients:
    - Combustion zone (high temperature gradient)
    - Reaction fronts (high species gradient)
  
  The mesh adaptation criterion is based on the equidistribution principle:
    ∫_{z_{i-1}}^{z_i} w(z) dz = constant
  where w(z) is a monitor function capturing solution variation.
  
  Monitor function options:
    w(z) = √(1 + α |dT/dz|²)   (temperature gradient)
    w(z) = √(1 + β |d²T/dz²|)  (curvature)
    w(z) = 1 + γ |ω_reaction|   (reaction rate)
"""

import math
import numpy as np
from cfd_solver import CVTMeshGenerator


class MeshAdapter:
    """
    Adaptive mesh generator for gasification reactor.
    """

    def __init__(self, z_min=0.0, z_max=2.5, n_nodes=50):
        self.z_min = float(z_min)
        self.z_max = float(z_max)
        self.n = int(n_nodes)

    def monitor_function(self, z, T_profile, alpha=1.0, beta=0.0):
        """
        Compute monitor function w(z) based on temperature gradient.
        w(z) = sqrt(1 + alpha * |dT/dz|^2 + beta * |d2T/dz2|)
        """
        z = np.asarray(z, dtype=float)
        T = np.asarray(T_profile, dtype=float)
        if len(T) != len(z):
            # Interpolate
            T = np.interp(z, np.linspace(self.z_min, self.z_max, len(T)), T)

        n = len(z)
        dTdz = np.zeros(n, dtype=float)
        for i in range(n):
            if i == 0:
                dz = z[1] - z[0]
                if abs(dz) > 1.0e-15:
                    dTdz[i] = (T[1] - T[0]) / dz
            elif i == n - 1:
                dz = z[n - 1] - z[n - 2]
                if abs(dz) > 1.0e-15:
                    dTdz[i] = (T[n - 1] - T[n - 2]) / dz
            else:
                dz_fwd = z[i + 1] - z[i]
                dz_bwd = z[i] - z[i - 1]
                if abs(dz_fwd) > 1.0e-15 and abs(dz_bwd) > 1.0e-15:
                    dTdz[i] = 0.5 * ((T[i + 1] - T[i]) / dz_fwd +
                                     (T[i] - T[i - 1]) / dz_bwd)

        d2Tdz2 = np.zeros(n, dtype=float)
        for i in range(1, n - 1):
            dz = z[i + 1] - z[i - 1]
            if abs(dz) > 1.0e-15:
                d2Tdz2[i] = (T[i + 1] - 2.0 * T[i] + T[i - 1]) / \
                            (0.25 * dz ** 2)

        w = np.sqrt(1.0 + alpha * dTdz ** 2 + beta * np.abs(d2Tdz2))
        return w

    def adapt_mesh(self, T_profile, method='cvt'):
        """
        Generate adapted mesh.
        """
        if method == 'cvt':
            return self._cvt_adapt(T_profile)
        elif method == 'equidistribute':
            return self._equidistribute(T_profile)
        else:
            return np.linspace(self.z_min, self.z_max, self.n)

    def _cvt_adapt(self, T_profile):
        """
        Use CVT with density proportional to monitor function.
        """
        # Sample monitor function on fine grid
        z_fine = np.linspace(self.z_min, self.z_max, 500)
        w = self.monitor_function(z_fine, T_profile)
        w = np.maximum(w, 1.0e-15)

        # Normalize to CDF
        W = np.cumsum(w)
        W = W / max(W[-1], 1.0e-15)

        # Use CVT generator with inverse CDF sampling
        cvt = CVTMeshGenerator(n_generators=self.n, density_func_id=0)
        generators = cvt.generate_mesh(z_min=self.z_min, z_max=self.z_max,
                                       n_samples=2000, n_steps=50)

        # Project generators to adapted positions using inverse CDF
        adapted = np.zeros(self.n, dtype=float)
        for i, g in enumerate(generators):
            # Find corresponding z where CDF matches generator position
            idx = np.searchsorted(W, (g - self.z_min) / (self.z_max - self.z_min))
            idx = min(idx, len(z_fine) - 1)
            adapted[i] = z_fine[idx]

        adapted = np.sort(adapted)
        adapted[0] = self.z_min
        adapted[-1] = self.z_max
        return adapted

    def _equidistribute(self, T_profile):
        """
        Direct equidistribution algorithm.
        """
        z_fine = np.linspace(self.z_min, self.z_max, 1000)
        w = self.monitor_function(z_fine, T_profile)
        w = np.maximum(w, 1.0e-15)

        # Compute cumulative monitor function
        W = np.zeros(len(z_fine), dtype=float)
        W[0] = 0.0
        for i in range(1, len(z_fine)):
            W[i] = W[i - 1] + w[i] * (z_fine[i] - z_fine[i - 1])

        W = W / max(W[-1], 1.0e-15)

        # Uniform target distribution
        target = np.linspace(0.0, 1.0, self.n)
        adapted = np.interp(target, W, z_fine)
        adapted[0] = self.z_min
        adapted[-1] = self.z_max
        return adapted

    def mesh_quality(self, z_nodes):
        """
        Compute mesh quality metrics.
        """
        z = np.asarray(z_nodes, dtype=float)
        dz = np.diff(z)
        metrics = {
            'min_spacing': float(np.min(dz)),
            'max_spacing': float(np.max(dz)),
            'mean_spacing': float(np.mean(dz)),
            'spacing_ratio': float(np.max(dz) / max(np.min(dz), 1.0e-15)),
            'uniformity': float(np.std(dz) / max(np.mean(dz), 1.0e-15))
        }
        return metrics
