
import numpy as np
from typing import List, Tuple, Callable


class SparseGridQuadrature:

    def __init__(self, dim_num: int, level_max: int):
        self.dim_num = dim_num
        self.level_max = level_max

    def clenshaw_curtis_nodes_weights(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        if n == 1:
            return np.array([0.0]), np.array([2.0])


        j = np.arange(n)
        x = np.cos(np.pi * j / (n - 1))



        w = np.ones(n)
        w[0] = 1.0 / ((n - 1) ** 2)
        w[-1] = 1.0 / ((n - 1) ** 2)
        for k in range(1, n - 1):
            theta = np.pi * k / (n - 1)
            w[k] = 2.0 / (n - 1) * (1.0 - np.sum([
                np.cos(2.0 * j * theta) / (4.0 * j * j - 1.0)
                for j in range(1, (n - 1) // 2 + 1)
            ]))


        w[0] = 1.0 / (n - 1)
        w[-1] = 1.0 / (n - 1)
        for k in range(1, n - 1):
            theta = np.pi * k / (n - 1)
            s = 0.0
            for j in range(1, (n - 1) // 2):
                s += np.cos(2.0 * j * theta) / (4.0 * j * j - 1.0)
            if (n - 1) % 2 == 0:
                s += np.cos((n - 1) * theta) / (2.0 * ((n - 1) ** 2 - 1.0))
            w[k] = 2.0 / (n - 1) * (1.0 - 2.0 * s - np.cos((n - 1) * theta) / ((n - 1) ** 2 - 1.0))




        w = np.ones(n)
        w[0] = 0.5
        w[-1] = 0.5
        w *= 2.0 / (n - 1)

        return x, w

    def level_to_order(self, level: int) -> int:
        if level == 0:
            return 1
        return 2 ** level + 1

    def comp_next(self, n: int, k: int, a: np.ndarray, more: bool, h: int, t: int) -> Tuple[np.ndarray, bool, int, int]:
        if not more:
            a[:] = 0
            a[0] = n
            h = 0
            t = n
            more = True if k > 1 else False
            return a, more, h, t

        if 1 < t:
            h = 0
        h += 1
        t = a[h - 1]
        a[h - 1] = 0
        a[0] = t - 1
        a[h] += 1
        more = True if a[k - 1] != n else False
        return a, more, h, t

    def build_sparse_grid(self) -> Tuple[np.ndarray, np.ndarray, int]:

        max_order = self.level_to_order(self.level_max)
        point_num_est = max_order ** self.dim_num


        grids = []
        weights = []



        for level_sum in range(self.level_max, self.level_max + self.dim_num + 1):
            a = np.zeros(self.dim_num, dtype=int)
            more = False
            h = 0
            t = 0
            while True:
                a, more, h, t = self.comp_next(level_sum, self.dim_num, a, more, h, t)

                if np.all(a >= 1) and np.sum(a) == level_sum:

                    sub_grid, sub_weight = self._tensor_product_for_levels(a)
                    grids.append(sub_grid)
                    weights.append(sub_weight)
                if not more:
                    break

        if not grids:

            return np.zeros((self.dim_num, 1)), np.array([1.0]), 1


        all_pts = np.hstack(grids)
        all_wts = np.hstack(weights)


        unique_pts = []
        unique_wts = []
        tol = 1e-10
        for i in range(all_pts.shape[1]):
            pt = all_pts[:, i]
            found = False
            for j, upt in enumerate(unique_pts):
                if np.linalg.norm(pt - upt) < tol:
                    unique_wts[j] += all_wts[i]
                    found = True
                    break
            if not found:
                unique_pts.append(pt)
                unique_wts.append(all_wts[i])

        grid_points = np.array(unique_pts).T
        grid_weights = np.array(unique_wts)


        if np.sum(grid_weights) > 0:
            grid_weights /= np.sum(grid_weights)
            grid_weights *= 2.0 ** self.dim_num

        return grid_points, grid_weights, grid_points.shape[1]

    def _tensor_product_for_levels(self, levels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        nodes_list = []
        weights_list = []
        for d in range(self.dim_num):
            n = self.level_to_order(int(levels[d]))
            x, w = self.clenshaw_curtis_nodes_weights(n)
            nodes_list.append(x)
            weights_list.append(w)


        mesh = np.meshgrid(*nodes_list, indexing="ij")
        grid = np.vstack([m.reshape(-1) for m in mesh])

        wt_mesh = np.meshgrid(*weights_list, indexing="ij")
        weights = np.prod(np.vstack([w.reshape(-1) for w in wt_mesh]), axis=0)

        return grid, weights

    def integrate(
        self,
        func: Callable[[np.ndarray], np.ndarray],
        param_mins: np.ndarray,
        param_maxs: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        grid_points, grid_weights, _ = self.build_sparse_grid()


        scale = (param_maxs - param_mins) / 2.0
        shift = (param_maxs + param_mins) / 2.0
        physical_pts = scale[:, None] * grid_points + shift[:, None]

        fvals = func(physical_pts)


        integral = float(np.dot(grid_weights, fvals))

        jacobian = np.prod(scale) * (2.0 ** self.dim_num)
        integral *= jacobian / (2.0 ** self.dim_num)


        variance = np.zeros(self.dim_num)
        fmean = np.average(fvals, weights=np.maximum(grid_weights, 0))
        for d in range(self.dim_num):

            unique_coords = np.unique(np.round(grid_points[d, :], 8))
            var_d = 0.0
            for uc in unique_coords:
                mask = np.abs(grid_points[d, :] - uc) < 1e-8
                if np.sum(mask) > 0:
                    local_mean = np.average(fvals[mask], weights=np.maximum(grid_weights[mask], 0))
                    var_d += (local_mean - fmean) ** 2
            variance[d] = var_d

        return integral, variance
