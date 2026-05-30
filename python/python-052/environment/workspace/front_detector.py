
import numpy as np

class EddyFrontDetector:

    def __init__(self, threshold_ratio=0.3):
        self.threshold_ratio = threshold_ratio

    @staticmethod
    def minmod(a, b, c):
        s = np.sign(a) + np.sign(b) + np.sign(c)

        same = (np.abs(s) >= 2.5).astype(float)
        mm = same * np.sign(a) * np.minimum(np.minimum(np.abs(a), np.abs(b)), np.abs(c))
        return mm

    def compute_gradient_magnitude(self, field, dx, dy):
        Ny, Nx = field.shape

        phi_padded = np.pad(field, ((1, 1), (1, 1)), mode='edge')

        dx_plus = (phi_padded[1:-1, 2:] - phi_padded[1:-1, 1:-1]) / dx
        dx_minus = (phi_padded[1:-1, 1:-1] - phi_padded[1:-1, :-2]) / dx
        dy_plus = (phi_padded[2:, 1:-1] - phi_padded[1:-1, 1:-1]) / dy
        dy_minus = (phi_padded[1:-1, 1:-1] - phi_padded[:-2, 1:-1]) / dy


        gx = self.minmod(dx_plus, dx_minus, 0.5 * (dx_plus + dx_minus))
        gy = self.minmod(dy_plus, dy_minus, 0.5 * (dy_plus + dy_minus))

        mag = np.sqrt(gx**2 + gy**2)
        return mag, gx, gy

    def detect_fronts(self, field, dx, dy):
        mag, _, _ = self.compute_gradient_magnitude(field, dx, dy)
        max_mag = np.max(mag)
        if max_mag < 1e-14:
            return np.zeros_like(field, dtype=bool), np.zeros_like(field)
        strength = mag / max_mag
        front_mask = strength > self.threshold_ratio
        return front_mask, strength

    def segment_eddies(self, field, dx, dy, vorticity):
        front_mask, _ = self.detect_fronts(field, dx, dy)

        vort_threshold = np.percentile(np.abs(vorticity), 75)
        interior = (np.abs(vorticity) > vort_threshold) & (~front_mask)


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
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')


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
