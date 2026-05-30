
import math
import random
from typing import List, Tuple, Optional


class HistogramPDFSampler:

    def __init__(self, data: List[float], num_bins: int = 256):
        if not data:
            raise ValueError("Data must not be empty.")
        self.data = list(data)
        self.n = len(data)
        self.num_bins = max(num_bins, 4)
        self.x_min = min(data)
        self.x_max = max(data)
        if self.x_max <= self.x_min:
            self.x_max = self.x_min + 1.0
        self.bin_width = (self.x_max - self.x_min) / self.num_bins
        self._build_histogram()

    def _build_histogram(self):
        self.counts = [0] * self.num_bins
        for x in self.data:
            idx = int((x - self.x_min) / self.bin_width)
            idx = min(idx, self.num_bins - 1)
            self.counts[idx] += 1
        total = sum(self.counts)
        if total == 0:
            total = 1

        self.pdf = [c / (total * self.bin_width) for c in self.counts]

        self.cdf = [0.0] * (self.num_bins + 1)
        for i in range(self.num_bins):
            self.cdf[i + 1] = self.cdf[i] + self.counts[i] / total

    def quantile(self, q: float) -> float:
        if q <= 0.0:
            return self.x_min
        if q >= 1.0:
            return self.x_max

        for i in range(self.num_bins):
            if self.cdf[i] <= q <= self.cdf[i + 1]:

                if self.cdf[i + 1] > self.cdf[i]:
                    frac = (q - self.cdf[i]) / (self.cdf[i + 1] - self.cdf[i])
                else:
                    frac = 0.5
                return self.x_min + (i + frac) * self.bin_width
        return self.x_max

    def partition_boundaries(self, num_partitions: int) -> List[float]:
        bounds = []
        for i in range(num_partitions + 1):
            q = i / num_partitions
            bounds.append(self.quantile(q))

        for i in range(1, len(bounds)):
            if bounds[i] < bounds[i - 1]:
                bounds[i] = bounds[i - 1]
        return bounds

    def sample(self, n_samples: int, seed: int = 0) -> List[float]:
        random.seed(seed)
        samples = []
        for _ in range(n_samples):
            u = random.random()
            samples.append(self.quantile(u))
        return samples


class RBFInterpolator:

    def __init__(self, x_data: List[float], f_data: List[float],
                 r0: float = 1.0, kernel_type: int = 1):
        if len(x_data) != len(f_data):
            raise ValueError("x_data and f_data must have the same length.")
        if len(x_data) < 2:
            raise ValueError("Need at least 2 data points.")
        self.xd = list(x_data)
        self.fd = list(f_data)
        self.nd = len(x_data)
        self.r0 = max(r0, 1e-10)
        self.kernel_type = kernel_type
        self._solve_weights()

    def _phi(self, r: float) -> float:
        if self.kernel_type == 1:

            return math.exp(-0.5 * r * r)
        elif self.kernel_type == 2:

            return 1.0 / math.sqrt(1.0 + r * r)
        elif self.kernel_type == 3:

            if r < 1e-12:
                return 0.0
            return r * r * math.log(r)
        else:

            return math.sqrt(1.0 + r * r)

    def _solve_weights(self):
        n = self.nd

        A = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                r = abs(self.xd[i] - self.xd[j]) / self.r0
                A[i][j] = self._phi(r)


        w = list(self.fd)
        pivot = list(range(n))
        for col in range(n):

            max_row = col
            max_val = abs(A[pivot[col]][col])
            for row in range(col + 1, n):
                if abs(A[pivot[row]][col]) > max_val:
                    max_val = abs(A[pivot[row]][col])
                    max_row = row
            pivot[col], pivot[max_row] = pivot[max_row], pivot[col]

            piv = pivot[col]
            if abs(A[piv][col]) < 1e-15:
                raise ValueError("RBF interpolation matrix is singular or ill-conditioned.")


            for row in range(col + 1, n):
                pr = pivot[row]
                factor = A[pr][col] / A[piv][col]
                for c in range(col, n):
                    A[pr][c] -= factor * A[piv][c]
                w[pr] -= factor * w[piv]


        self.weights = [0.0] * n
        for i in range(n - 1, -1, -1):
            piv = pivot[i]
            s = w[piv]
            for j in range(i + 1, n):
                s -= A[piv][j] * self.weights[j]
            self.weights[i] = s / A[piv][i]

    def evaluate(self, x: float) -> float:
        result = 0.0
        for j in range(self.nd):
            r = abs(x - self.xd[j]) / self.r0
            result += self.weights[j] * self._phi(r)
        return result


class FEM1DProjector:

    def __init__(self, nodes: List[float]):
        if len(nodes) < 2:
            raise ValueError("Need at least 2 nodes.")
        self.nodes = sorted(nodes)
        self.n_nodes = len(nodes)

    def _basis(self, x: float, xl: float, xr: float) -> Tuple[float, float]:
        h = xr - xl
        if h < 1e-15:
            return 1.0, 0.0
        phil = (xr - x) / h
        phir = (x - xl) / h
        return phil, phir

    def project(self, f_func) -> List[float]:
        n = self.n_nodes

        M = [[0.0] * n for _ in range(n)]
        b = [0.0] * n


        xi = [-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0)]
        wg = [1.0, 1.0]

        for e in range(n - 1):
            xl, xr = self.nodes[e], self.nodes[e + 1]
            h = xr - xl
            mid = 0.5 * (xl + xr)

            for q in range(2):
                xq = mid + 0.5 * h * xi[q]
                wq = 0.5 * h * wg[q]
                phil, phir = self._basis(xq, xl, xr)
                fv = f_func(xq)


                M[e][e]     += wq * phil * phil
                M[e][e+1]   += wq * phil * phir
                M[e+1][e]   += wq * phir * phil
                M[e+1][e+1] += wq * phir * phir
                b[e]        += wq * fv * phil
                b[e+1]      += wq * fv * phir


        return self._thomas_solve(M, b)

    def _thomas_solve(self, M: List[List[float]], b: List[float]) -> List[float]:
        n = len(b)

        lower = [0.0] * n
        diag  = [0.0] * n
        upper = [0.0] * n
        for i in range(n):
            diag[i] = M[i][i]
            if i > 0:
                lower[i] = M[i][i-1]
            if i < n - 1:
                upper[i] = M[i][i+1]


        cp = [0.0] * n
        dp = [0.0] * n
        cp[0] = upper[0] / diag[0] if abs(diag[0]) > 1e-15 else 0.0
        dp[0] = b[0] / diag[0] if abs(diag[0]) > 1e-15 else 0.0
        for i in range(1, n):
            denom = diag[i] - lower[i] * cp[i-1]
            if abs(denom) < 1e-15:
                denom = 1e-15
            cp[i] = upper[i] / denom if i < n - 1 else 0.0
            dp[i] = (b[i] - lower[i] * dp[i-1]) / denom


        x = [0.0] * n
        x[-1] = dp[-1]
        for i in range(n - 2, -1, -1):
            x[i] = dp[i] - cp[i] * x[i+1]
        return x


class PolygonPartitionSampler:

    def __init__(self, vertices: List[Tuple[float, float]]):
        if len(vertices) < 3:
            raise ValueError("Polygon must have at least 3 vertices.")
        self.vertices = vertices
        self.triangles = self._ear_clipping_triangulation()

    def _triangle_area(self, a: Tuple[float, float], b: Tuple[float, float],
                       c: Tuple[float, float]) -> float:
        return 0.5 * abs(
            a[0]*(b[1]-c[1]) + b[0]*(c[1]-a[1]) + c[0]*(a[1]-b[1])
        )

    def _is_convex(self, prev: int, curr: int, next_idx: int) -> bool:
        n = len(self.vertices)
        a = self.vertices[prev % n]
        b = self.vertices[curr % n]
        c = self.vertices[next_idx % n]
        cross = (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
        return cross > 0

    def _ear_clipping_triangulation(self) -> List[Tuple[int, int, int]]:
        n = len(self.vertices)
        if n == 3:
            return [(0, 1, 2)]

        indices = list(range(n))
        triangles = []
        while len(indices) > 3:
            m = len(indices)
            found = False
            for i in range(m):
                prev = indices[(i - 1) % m]
                curr = indices[i]
                next_idx = indices[(i + 1) % m]
                if self._is_convex(prev, curr, next_idx):
                    triangles.append((prev, curr, next_idx))
                    indices.pop(i)
                    found = True
                    break
            if not found:

                triangles.append((indices[0], indices[1], indices[2]))
                indices.pop(1)
        triangles.append((indices[0], indices[1], indices[2]))
        return triangles

    def sample(self, n_samples: int, seed: int = 0) -> List[Tuple[float, float]]:
        random.seed(seed)
        areas = []
        for t in self.triangles:
            a = self.vertices[t[0]]
            b = self.vertices[t[1]]
            c = self.vertices[t[2]]
            areas.append(self._triangle_area(a, b, c))
        total_area = sum(areas)
        if total_area < 1e-15:
            return [self.vertices[0]] * n_samples


        cum = [0.0]
        for ar in areas:
            cum.append(cum[-1] + ar / total_area)

        samples = []
        for _ in range(n_samples):
            u = random.random()
            idx = 0
            for i in range(len(cum) - 1):
                if cum[i] <= u <= cum[i + 1]:
                    idx = i
                    break
            t = self.triangles[idx]
            a = self.vertices[t[0]]
            b = self.vertices[t[1]]
            c = self.vertices[t[2]]
            r1 = random.random()
            r2 = random.random()
            if r1 + r2 > 1.0:
                r1 = 1.0 - r1
                r2 = 1.0 - r2
            px = a[0] + r1 * (b[0] - a[0]) + r2 * (c[0] - a[0])
            py = a[1] + r1 * (b[1] - a[1]) + r2 * (c[1] - a[1])
            samples.append((px, py))
        return samples


def adaptive_partition_estimation(
    sample_data: List[float],
    num_partitions: int,
    use_rbf_refinement: bool = True
) -> List[float]:

    hist_sampler = HistogramPDFSampler(sample_data, num_bins=max(num_partitions * 4, 64))
    coarse_bounds = hist_sampler.partition_boundaries(num_partitions)

    if not use_rbf_refinement or len(sample_data) < 20:
        return coarse_bounds



    x_rbf = coarse_bounds
    y_rbf = [hist_sampler.quantile(q) for q in [i / len(coarse_bounds) for i in range(len(coarse_bounds))]]

    x_rbf = [hist_sampler.x_min + i * (hist_sampler.x_max - hist_sampler.x_min) / 50.0 for i in range(51)]
    y_rbf = [hist_sampler.quantile(i / 50.0) for i in range(51)]

    try:
        rbf = RBFInterpolator(x_rbf, y_rbf, r0=(hist_sampler.x_max - hist_sampler.x_min) / 10.0, kernel_type=1)
        refined = [rbf.evaluate(hist_sampler.x_min + q * (hist_sampler.x_max - hist_sampler.x_min))
                   for q in [i / num_partitions for i in range(num_partitions + 1)]]

        for i in range(1, len(refined)):
            if refined[i] < refined[i - 1]:
                refined[i] = refined[i - 1]
        return refined
    except Exception:
        return coarse_bounds
