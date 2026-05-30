
import numpy as np






_HILBERT_ROTATION_TABLE = np.array([


    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1],
], dtype=int)


def _hilbert_scalar_h_to_xyz(h, n_bits=8):
    x = y = z = 0
    state = 0
    for i in range(n_bits - 1, -1, -1):
        digit = (h >> (3 * i)) & 7
        if state == 0:
            if digit == 0:
                nx, ny, nz = 0, 0, 0; state = 1
            elif digit == 1:
                nx, ny, nz = 0, 0, 1; state = 0
            elif digit == 2:
                nx, ny, nz = 0, 1, 1; state = 0
            elif digit == 3:
                nx, ny, nz = 0, 1, 0; state = 1
            elif digit == 4:
                nx, ny, nz = 1, 1, 0; state = 0
            elif digit == 5:
                nx, ny, nz = 1, 1, 1; state = 1
            elif digit == 6:
                nx, ny, nz = 1, 0, 1; state = 1
            else:
                nx, ny, nz = 1, 0, 0; state = 0
        elif state == 1:
            if digit == 0:
                nx, ny, nz = 0, 0, 0; state = 0
            elif digit == 1:
                nx, ny, nz = 1, 0, 0; state = 1
            elif digit == 2:
                nx, ny, nz = 1, 0, 1; state = 0
            elif digit == 3:
                nx, ny, nz = 0, 0, 1; state = 1
            elif digit == 4:
                nx, ny, nz = 0, 1, 1; state = 1
            elif digit == 5:
                nx, ny, nz = 1, 1, 1; state = 0
            elif digit == 6:
                nx, ny, nz = 1, 1, 0; state = 0
            else:
                nx, ny, nz = 0, 1, 0; state = 1
        else:
            nx = ny = nz = 0
        x |= (nx << i)
        y |= (ny << i)
        z |= (nz << i)
    return x, y, z


def hilbert_h_to_xyz(h, n_bits=8):
    h = np.asarray(h, dtype=np.int64)
    scalar = (h.ndim == 0)
    h_flat = np.atleast_1d(h)
    out = np.zeros((h_flat.size, 3), dtype=np.int64)
    for idx, hi in enumerate(h_flat):
        out[idx, 0], out[idx, 1], out[idx, 2] = _hilbert_scalar_h_to_xyz(int(hi), n_bits)
    if scalar:
        return out[0]
    return out.reshape(h.shape + (3,))


def _hilbert_scalar_xyz_to_h(x, y, z, n_bits=8):
    h = 0
    state = 0
    for i in range(n_bits - 1, -1, -1):
        nx = (x >> i) & 1
        ny = (y >> i) & 1
        nz = (z >> i) & 1
        if state == 0:
            if nx == 0 and ny == 0 and nz == 0:
                digit = 0; state = 1
            elif nx == 0 and ny == 0 and nz == 1:
                digit = 1; state = 0
            elif nx == 0 and ny == 1 and nz == 1:
                digit = 2; state = 0
            elif nx == 0 and ny == 1 and nz == 0:
                digit = 3; state = 1
            elif nx == 1 and ny == 1 and nz == 0:
                digit = 4; state = 0
            elif nx == 1 and ny == 1 and nz == 1:
                digit = 5; state = 1
            elif nx == 1 and ny == 0 and nz == 1:
                digit = 6; state = 1
            else:
                digit = 7; state = 0
        elif state == 1:
            if nx == 0 and ny == 0 and nz == 0:
                digit = 0; state = 0
            elif nx == 1 and ny == 0 and nz == 0:
                digit = 1; state = 1
            elif nx == 1 and ny == 0 and nz == 1:
                digit = 2; state = 0
            elif nx == 0 and ny == 0 and nz == 1:
                digit = 3; state = 1
            elif nx == 0 and ny == 1 and nz == 1:
                digit = 4; state = 1
            elif nx == 1 and ny == 1 and nz == 1:
                digit = 5; state = 0
            elif nx == 1 and ny == 1 and nz == 0:
                digit = 6; state = 0
            else:
                digit = 7; state = 1
        else:
            digit = 0
        h |= (digit << (3 * i))
    return h


def hilbert_xyz_to_h(coords, n_bits=8):
    coords = np.asarray(coords, dtype=np.int64)
    scalar = (coords.ndim == 1 and coords.shape[0] == 3)
    c_flat = np.atleast_2d(coords)
    out = np.zeros(c_flat.shape[0], dtype=np.int64)
    for idx in range(c_flat.shape[0]):
        out[idx] = _hilbert_scalar_xyz_to_h(
            int(c_flat[idx, 0]), int(c_flat[idx, 1]), int(c_flat[idx, 2]), n_bits
        )
    if scalar:
        return out[0]
    if coords.ndim == 2:
        return out
    return out.reshape(coords.shape[:-1])


class HilbertSpatialIndex:
    
    def __init__(self, n_bits=8, bbox=None):
        self.n_bits = n_bits
        self.grid_size = 1 << n_bits
        if bbox is None:
            bbox = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]
        self.bbox = np.array(bbox, dtype=float)
        self.points = []
        self.indices = []
    
    def _world_to_grid(self, p):
        p = np.asarray(p, dtype=float)
        grid = np.zeros(3, dtype=np.int64)
        for dim in range(3):
            low, high = self.bbox[dim]

            t = (p[dim] - low) / (high - low)
            t = max(0.0, min(1.0, t))
            grid[dim] = int(t * (self.grid_size - 1))
        return grid
    
    def add_point(self, point, idx):
        self.points.append(np.asarray(point, dtype=float))
        self.indices.append(idx)
    
    def build_index(self):
        if len(self.points) == 0:
            self.ordered = np.array([], dtype=int)
            return
        
        pts = np.array(self.points)
        grid_coords = np.array([self._world_to_grid(p) for p in pts])
        h_vals = hilbert_xyz_to_h(grid_coords, self.n_bits)
        self.ordered = np.argsort(h_vals)
        self.sorted_h = np.sort(h_vals)
    
    def range_query(self, center, radius):
        if len(self.points) == 0:
            return []
        
        center = np.asarray(center, dtype=float)
        candidates = []
        

        gmin = self._world_to_grid(center - radius)
        gmax = self._world_to_grid(center + radius)
        

        h_min = hilbert_xyz_to_h(gmin.reshape(1, 3), self.n_bits)[0]
        h_max = hilbert_xyz_to_h(gmax.reshape(1, 3), self.n_bits)[0]
        
        if h_min > h_max:
            h_min, h_max = h_max, h_min
        

        left = np.searchsorted(self.sorted_h, h_min, side='left')
        right = np.searchsorted(self.sorted_h, h_max, side='right')
        
        for pos in range(left, right):
            candidates.append(self.indices[self.ordered[pos]])
        
        return candidates






def cvt_lloyd_2d(n_generators, density_func, n_samples=20000,
                 max_iter=50, tol=1e-5, seed=42, domain=None):
    rng = np.random.default_rng(seed)
    if domain is None:
        domain = [(0.0, 1.0), (0.0, 1.0)]
    

    gens = np.zeros((n_generators, 2), dtype=float)
    for d in range(2):
        gens[:, d] = rng.uniform(domain[d][0], domain[d][1], size=n_generators)
    
    lows = np.array([domain[d][0] for d in range(2)])
    highs = np.array([domain[d][1] for d in range(2)])
    
    for it in range(max_iter):

        samples = lows + rng.uniform(0.0, 1.0, size=(n_samples, 2)) * (highs - lows)
        weights = np.array([density_func(s[0], s[1]) for s in samples])
        weights = np.maximum(weights, 1e-15)
        


        diffs = samples[:, np.newaxis, :] - gens[np.newaxis, :, :]
        dists2 = np.sum(diffs**2, axis=2)
        nearest = np.argmin(dists2, axis=1)
        

        new_gens = np.zeros_like(gens)
        for j in range(n_generators):
            mask = (nearest == j)
            if np.any(mask):
                new_gens[j] = np.average(samples[mask], axis=0, weights=weights[mask])
            else:

                new_gens[j] = lows + rng.uniform(0.0, 1.0, size=2) * (highs - lows)
        

        displacement = np.max(np.linalg.norm(new_gens - gens, axis=1))
        gens = new_gens
        if displacement < tol:
            break
    
    info = {'iterations': it + 1, 'final_displacement': displacement}
    return gens, info


def test_adaptive_sampling():

    n_bits = 4
    h_vals = np.arange(0, 2**(3*n_bits), 7)
    coords = hilbert_h_to_xyz(h_vals, n_bits)
    h_back = hilbert_xyz_to_h(coords, n_bits)
    assert np.all(h_vals == h_back), "Hilbert round-trip failed"
    

    idx = HilbertSpatialIndex(n_bits=6, bbox=[(-100, 100), (-100, 100), (-100, 100)])
    rng = np.random.default_rng(123)
    pts = rng.uniform(-50, 50, size=(200, 3))
    for i, p in enumerate(pts):
        idx.add_point(p, i)
    idx.build_index()
    cands = idx.range_query(np.zeros(3), 20.0)
    assert len(cands) > 0, "Hilbert index query returned no candidates"
    

    def rho(x, y):
        return np.exp(-5.0 * ((x - 0.5)**2 + (y - 0.5)**2)) + 0.1
    
    gens, info = cvt_lloyd_2d(16, rho, n_samples=5000, max_iter=20, seed=42)
    assert gens.shape == (16, 2)
    assert info['iterations'] <= 20
    
    return True


if __name__ == "__main__":
    test_adaptive_sampling()
    print("Adaptive sampling tests passed.")
