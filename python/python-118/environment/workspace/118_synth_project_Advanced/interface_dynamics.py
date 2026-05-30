
import numpy as np
from utils_numeric import check_bounds, safe_sqrt


class InterfaceTracker:

    def __init__(self, q_threshold=0.40, delta_q=0.05, n_bins_xy=20):
        self.q_threshold = float(q_threshold)
        self.delta_q = float(delta_q)
        self.n_bins_xy = int(n_bins_xy)

    def compute_solid_weight(self, q_values):
        return 0.5 * (1.0 + np.tanh((q_values - self.q_threshold) / self.delta_q))

    def locate_interface(self, positions, q_values, box):
        n_atoms = positions.shape[0]
        weights = self.compute_solid_weight(q_values)

        x_edges = np.linspace(0, box[0], self.n_bins_xy + 1)
        y_edges = np.linspace(0, box[1], self.n_bins_xy + 1)
        z_interface = np.zeros((self.n_bins_xy, self.n_bins_xy), dtype=np.float64)
        count = np.zeros((self.n_bins_xy, self.n_bins_xy), dtype=np.float64)

        for i in range(n_atoms):
            x, y, z = positions[i]
            ix = int(x / box[0] * self.n_bins_xy)
            iy = int(y / box[1] * self.n_bins_xy)
            ix = min(ix, self.n_bins_xy - 1)
            iy = min(iy, self.n_bins_xy - 1)

            w = weights[i]
            z_interface[ix, iy] += z * (0.5 - w)
            count[ix, iy] += abs(0.5 - w)

        mask = count > 0
        z_interface[mask] /= count[mask]

        for ix in range(self.n_bins_xy):
            for iy in range(self.n_bins_xy):
                if count[ix, iy] == 0:

                    best_val = 0.0
                    best_dist = 1e10
                    for jx in range(self.n_bins_xy):
                        for jy in range(self.n_bins_xy):
                            if count[jx, jy] > 0:
                                d2 = (jx - ix) ** 2 + (jy - iy) ** 2
                                if d2 < best_dist:
                                    best_dist = d2
                                    best_val = z_interface[jx, jy]
                    z_interface[ix, iy] = best_val
        return z_interface, x_edges, y_edges

    def compute_roughness(self, z_interface):
        h = z_interface.flatten()
        h_mean = np.mean(h)
        W = safe_sqrt(np.mean((h - h_mean) ** 2))
        return W, h_mean

    def compute_capillary_waves_spectrum(self, z_interface, box_xy):
        nx, ny = z_interface.shape
        h_k = np.fft.fft2(z_interface - np.mean(z_interface))
        h_k2 = np.abs(h_k) ** 2 / (nx * ny)


        kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=box_xy[0] / nx)
        ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=box_xy[1] / ny)
        KX, KY = np.meshgrid(kx, ky, indexing='ij')
        k_mag = safe_sqrt(KX ** 2 + KY ** 2)

        k_bins = np.linspace(0, np.max(k_mag), nx // 2 + 1)
        spectrum = np.zeros(len(k_bins) - 1, dtype=np.float64)
        counts = np.zeros(len(k_bins) - 1, dtype=np.float64)

        for i in range(nx):
            for j in range(ny):
                kval = k_mag[i, j]
                idx = np.searchsorted(k_bins[1:], kval)
                if idx < len(spectrum):
                    spectrum[idx] += h_k2[i, j]
                    counts[idx] += 1.0

        mask = counts > 0
        spectrum[mask] /= counts[mask]
        k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
        return k_centers, spectrum

    def cluster_analysis(self, is_solid, positions, box, r_cut=3.5):
        n_atoms = positions.shape[0]
        visited = np.zeros(n_atoms, dtype=bool)
        clusters = []

        def dfs(start, phase_mask):
            stack = [start]
            cluster = []
            while stack:
                i = stack.pop()
                if visited[i]:
                    continue
                visited[i] = True
                cluster.append(i)
                for j in range(n_atoms):
                    if not phase_mask[j] or visited[j]:
                        continue
                    rij = positions[j] - positions[i]
                    rij -= box * np.round(rij / box)
                    if np.dot(rij, rij) < r_cut ** 2:
                        stack.append(j)
            return cluster


        solid_mask = is_solid
        for i in range(n_atoms):
            if solid_mask[i] and not visited[i]:
                cluster = dfs(i, solid_mask)
                clusters.append(cluster)

        cluster_sizes = [len(c) for c in clusters]
        if len(cluster_sizes) == 0:
            return 0, 0, []
        max_cluster_size = max(cluster_sizes)
        n_clusters = len(cluster_sizes)
        return max_cluster_size, n_clusters, cluster_sizes

    def compute_interface_velocity(self, z_interface_t1, z_interface_t2, dt):
        v_n = np.mean(z_interface_t2 - z_interface_t1) / dt
        return v_n


class CapillaryWaveTheory:

    def __init__(self, gamma=0.2, T=1200.0, a0=3.5):
        self.gamma = float(gamma)
        self.T = float(T)
        self.kb = 8.617333e-5
        self.a0 = float(a0)

    def predicted_correlation(self, r):
        r = np.maximum(r, self.a0)
        return self.kb * self.T / (np.pi * self.gamma) * np.log(r / self.a0)

    def fit_gamma_from_spectrum(self, k, spectrum):
        mask = (k > 0) & (spectrum > 0)
        if np.sum(mask) < 2:
            return self.gamma
        logk = np.log(k[mask])
        logS = np.log(spectrum[mask])

        A = np.vstack([logk, np.ones(len(logk))]).T
        slope, intercept = np.linalg.lstsq(A, logS, rcond=None)[0]


        fitted_gamma = self.kb * self.T / np.exp(intercept)
        return fitted_gamma
