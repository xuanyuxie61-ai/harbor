"""
Eddy Boundary Front Detection via Piecewise Polynomial Fitting
==============================================================
Derived from seed project 325_edge (piecewise test functions with
sharp transitions for edge detection).

Mesoscale eddies are delineated by sharp gradients in PV, vorticity,
and tracers. We detect eddy boundaries using a 2D generalization of
the edge-detection framework based on piecewise polynomial fitting.

For a scalar field φ(x,y), an edge (eddy front) is identified where
the directional derivative exceeds a threshold:

    |∇φ · n̂| > τ · max(|∇φ|)

where n̂ = ∇φ / |∇φ| is the normal direction and τ ∈ (0,1) is a
relative threshold.

The edge-strength metric uses a smoothed minmod limiter to avoid
spurious oscillations:

    S(x,y) = minmod( ∂φ/∂x⁺, ∂φ/∂x⁻, ∂φ/∂y⁺, ∂φ/∂y⁻ )

    minmod(a,b,c) = sgn(a) · min(|a|, |b|, |c|)
                    if sgn(a)=sgn(b)=sgn(c), else 0

Closed contours of S(x,y) = 0 enclosing high-vorticity regions
define individual eddy domains.
"""

import numpy as np

class EddyFrontDetector:
    """
    Detect eddy boundaries from vorticity or PV fields.
    """

    def __init__(self, threshold_ratio=0.3):
        self.threshold_ratio = threshold_ratio

    @staticmethod
    def minmod(a, b, c):
        """
        Minmod limiter for robust gradient estimation.
        """
        s = np.sign(a) + np.sign(b) + np.sign(c)
        # All same sign if |s| == 3
        same = (np.abs(s) >= 2.5).astype(float)
        mm = same * np.sign(a) * np.minimum(np.minimum(np.abs(a), np.abs(b)), np.abs(c))
        return mm

    def compute_gradient_magnitude(self, field, dx, dy):
        """
        Compute central-difference gradient magnitude with minmod limiter.
        """
        Ny, Nx = field.shape
        # Pad with Neumann boundary
        phi_padded = np.pad(field, ((1, 1), (1, 1)), mode='edge')

        dx_plus = (phi_padded[1:-1, 2:] - phi_padded[1:-1, 1:-1]) / dx
        dx_minus = (phi_padded[1:-1, 1:-1] - phi_padded[1:-1, :-2]) / dx
        dy_plus = (phi_padded[2:, 1:-1] - phi_padded[1:-1, 1:-1]) / dy
        dy_minus = (phi_padded[1:-1, 1:-1] - phi_padded[:-2, 1:-1]) / dy

        # Minmod-limited gradient components
        gx = self.minmod(dx_plus, dx_minus, 0.5 * (dx_plus + dx_minus))
        gy = self.minmod(dy_plus, dy_minus, 0.5 * (dy_plus + dy_minus))

        mag = np.sqrt(gx**2 + gy**2)
        return mag, gx, gy

    def detect_fronts(self, field, dx, dy):
        """
        Detect front pixels where gradient magnitude exceeds threshold.

        Returns
        -------
        front_mask : ndarray, bool
        strength : ndarray
            Normalized edge strength [0, 1].
        """
        mag, _, _ = self.compute_gradient_magnitude(field, dx, dy)
        max_mag = np.max(mag)
        if max_mag < 1e-14:
            return np.zeros_like(field, dtype=bool), np.zeros_like(field)
        strength = mag / max_mag
        front_mask = strength > self.threshold_ratio
        return front_mask, strength

    def segment_eddies(self, field, dx, dy, vorticity):
        """
        Segment individual eddies by finding connected regions of
        high vorticity bounded by fronts.

        Returns
        -------
        labels : ndarray, int
            Label map (0 = background).
        n_eddy : int
            Number of detected eddies.
        eddy_stats : list of dict
            Centroid, area, mean vorticity for each eddy.
        """
        front_mask, _ = self.detect_fronts(field, dx, dy)
        # Eddy interior: high |vorticity| and not on front
        vort_threshold = np.percentile(np.abs(vorticity), 75)
        interior = (np.abs(vorticity) > vort_threshold) & (~front_mask)

        # Simple connected-component labeling (4-connectivity)
        labels = np.zeros_like(interior, dtype=int)
        label_id = 0
        visited = np.zeros_like(interior, dtype=bool)
        Ny, Nx = interior.shape

        for j in range(Ny):
            for i in range(Nx):
                if interior[j, i] and not visited[j, i]:
                    label_id += 1
                    stack = [(i, j)]
                    visited[j, i] = True
                    labels[j, i] = label_id
                    while stack:
                        cx, cy = stack.pop()
                        for dx_n, dy_n in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nx, ny = cx + dx_n, cy + dy_n
                            if 0 <= nx < Nx and 0 <= ny < Ny:
                                if interior[ny, nx] and not visited[ny, nx]:
                                    visited[ny, nx] = True
                                    labels[ny, nx] = label_id
                                    stack.append((nx, ny))

        # Compute eddy statistics
        eddy_stats = []
        for lid in range(1, label_id + 1):
            mask = (labels == lid)
            if np.sum(mask) == 0:
                continue
            y_idx, x_idx = np.where(mask)
            centroid = (np.mean(x_idx) * dx, np.mean(y_idx) * dy)
            area = np.sum(mask) * dx * dy
            mean_vort = np.mean(vorticity[mask])
            eddy_stats.append({
                'id': lid,
                'centroid': centroid,
                'area': area,
                'mean_vorticity': mean_vort,
                'n_pixels': int(np.sum(mask))
            })

        return labels, label_id, eddy_stats


def shepp_logan_ocean_tracer(Nx, Ny, Lx, Ly):
    """
    Generate a synthetic ocean tracer field inspired by the Shepp-Logan
    phantom (from 325_edge), consisting of overlapping elliptical regions
    that mimic eddy-induced tracer anomalies.

    Ellipse formula:
        ((x−xc)cosθ + (y−yc)sinθ)²/a² + ((x−xc)sinθ − (y−yc)cosθ)²/b² ≤ 1
    """
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Define elliptical "eddies"
    ellipses = [
        {'xc': 0.35*Lx, 'yc': 0.5*Ly, 'a': 0.15*Lx, 'b': 0.10*Ly, 'theta': 0.3, 'amp': 1.0},
        {'xc': 0.65*Lx, 'yc': 0.45*Ly, 'a': 0.12*Lx, 'b': 0.08*Ly, 'theta': -0.5, 'amp': -0.8},
        {'xc': 0.5*Lx, 'yc': 0.7*Ly, 'a': 0.08*Lx, 'b': 0.12*Ly, 'theta': 1.0, 'amp': 0.6},
    ]

    tracer = np.zeros((Nx, Ny), dtype=np.float64)
    for ell in ellipses:
        cx = X - ell['xc']
        cy = Y - ell['yc']
        ct = np.cos(ell['theta'])
        st = np.sin(ell['theta'])
        val = ((cx * ct + cy * st)**2 / ell['a']**2 +
               (cx * st - cy * ct)**2 / ell['b']**2)
        tracer += ell['amp'] * np.exp(-val)

    return tracer
