
import numpy as np
from scipy.linalg import qr


class TopologyOptimizer:

    def __init__(self, n_levels=16):
        self.n_levels = n_levels
        self.discrete_phases = np.linspace(0, 2 * np.pi, n_levels, endpoint=False)




    def quantize_phase_dp(self, target_phases, weights=None):
        N = len(target_phases)
        L = self.n_levels
        if weights is None:
            weights = np.ones(N)


        cost_local = np.zeros((N, L), dtype=np.float64)
        for i in range(N):
            for j in range(L):
                dp = self.discrete_phases[j]

                cost_local[i, j] = weights[i] * abs(
                    np.exp(1.0j * dp) - np.exp(1.0j * target_phases[i])
                ) ** 2



        dp_table = np.full((N, L), np.inf)
        choice = np.zeros((N, L), dtype=np.int32)


        dp_table[0, :] = cost_local[0, :]


        for i in range(1, N):
            for j in range(L):

                best_cost = np.inf
                best_k = 0
                for k in range(L):

                    continuity_penalty = 0.1 * abs(
                        np.exp(1.0j * self.discrete_phases[j]) -
                        np.exp(1.0j * self.discrete_phases[k])
                    ) ** 2
                    c = dp_table[i - 1, k] + cost_local[i, j] + continuity_penalty
                    if c < best_cost:
                        best_cost = c
                        best_k = k
                dp_table[i, j] = best_cost
                choice[i, j] = best_k


        quantized = np.zeros(N, dtype=np.float64)
        min_final = np.inf
        best_j = 0
        for j in range(L):
            if dp_table[N - 1, j] < min_final:
                min_final = dp_table[N - 1, j]
                best_j = j
        quantized[N - 1] = self.discrete_phases[best_j]
        for i in range(N - 1, 0, -1):
            best_j = choice[i, best_j]
            quantized[i - 1] = self.discrete_phases[best_j]

        return quantized, min_final

    def optimize_pillar_geometry_dp(self, target_phases, param_table,
                                     weights=None):
        N = len(target_phases)
        L = len(param_table)
        if weights is None:
            weights = np.ones(N)

        cost_local = np.zeros((N, L), dtype=np.float64)
        for i in range(N):
            for j in range(L):
                dp = param_table[j]['phase']
                cost_local[i, j] = weights[i] * abs(
                    np.exp(1.0j * dp) - np.exp(1.0j * target_phases[i])
                ) ** 2

        dp_table = np.full((N, L), np.inf)
        choice = np.zeros((N, L), dtype=np.int32)
        dp_table[0, :] = cost_local[0, :]

        for i in range(1, N):
            for j in range(L):
                best_cost = np.inf
                best_k = 0
                for k in range(L):

                    h_jump = abs(param_table[j]['height'] - param_table[k]['height'])
                    w_jump = abs(param_table[j]['width'] - param_table[k]['width'])
                    geo_penalty = 1e12 * (h_jump + w_jump)
                    c = dp_table[i - 1, k] + cost_local[i, j] + geo_penalty
                    if c < best_cost:
                        best_cost = c
                        best_k = k
                dp_table[i, j] = best_cost
                choice[i, j] = best_k


        params_opt = []
        min_final = np.inf
        best_j = 0
        for j in range(L):
            if dp_table[N - 1, j] < min_final:
                min_final = dp_table[N - 1, j]
                best_j = j
        params_opt.insert(0, param_table[best_j])
        for i in range(N - 1, 0, -1):
            best_j = choice[i, best_j]
            params_opt.insert(0, param_table[best_j])

        return params_opt, min_final




    def compressed_inverse_design(self, A, b_target, sparsity_factor=0.3):
        M, N = A.shape

        Q, R, p = qr(A, pivoting=True, mode='economic')

        tol = max(M, N) * np.finfo(float).eps * np.abs(R[0, 0])
        r = np.sum(np.abs(np.diag(R)) > tol)
        r = min(r, int(N * sparsity_factor))
        r = max(r, 1)

        y = Q[:, :r].T.conj() @ b_target
        x_reduced = np.linalg.solve(R[:r, :r], y)
        x = np.zeros(N, dtype=np.complex128)
        x[p[:r]] = x_reduced

        residual = np.linalg.norm(A @ x - b_target)
        return x, residual

    def greedy_pillar_selection(self, A, b_target, max_pillars=50):
        M, N = A.shape
        residual = b_target.copy()
        selected = []
        x = np.zeros(N, dtype=np.complex128)

        for _ in range(max_pillars):

            correlations = np.abs(A.T.conj() @ residual)
            if np.max(correlations) < 1e-15:
                break
            idx = np.argmax(correlations)
            selected.append(idx)


            A_sel = A[:, selected]
            x_sel, _, _, _ = np.linalg.lstsq(A_sel, b_target, rcond=None)
            x[selected] = x_sel
            residual = b_target - A_sel @ x_sel

        return x, np.linalg.norm(residual), selected

    def optimize_phase_gradient(self, target_field, x_coords, y_coords,
                                 k0, n_levels=8):

        dx = np.gradient(x_coords)
        dy = np.gradient(y_coords)

        r = np.sqrt(x_coords ** 2 + y_coords ** 2)
        dr = np.gradient(r)

        target_phases = np.angle(target_field)

        quantized, error = self.quantize_phase_dp(target_phases)
        return quantized, error


def demo():
    opt = TopologyOptimizer(n_levels=8)


    N = 100
    target = np.linspace(0, 4 * np.pi, N)
    weights = np.ones(N)
    quantized, err = opt.quantize_phase_dp(target, weights)
    print(f"[topology_optimize] DP 量化误差: {err:.4f}")
    print(f"[topology_optimize] 量化前 5 个相位: " +
          ", ".join(f"{q:.3f}" for q in quantized[:5]))


    M = 50
    N_lib = 200
    np.random.seed(0)
    A = np.random.randn(M, N_lib) + 1.0j * np.random.randn(M, N_lib)

    x_true = np.zeros(N_lib, dtype=np.complex128)
    x_true[np.random.choice(N_lib, 10, replace=False)] = np.random.randn(10) + 1.0j * np.random.randn(10)
    b_target = A @ x_true

    x_est, res = opt.compressed_inverse_design(A, b_target, sparsity_factor=0.3)
    nnz = np.sum(np.abs(x_est) > 1e-10)
    print(f"[topology_optimize] 压缩感知解: 非零元素={nnz}, 残差={res:.4e}")

    x_greedy, res_greedy, sel = opt.greedy_pillar_selection(A, b_target, max_pillars=15)
    print(f"[topology_optimize] 贪心选择: 选中 {len(sel)} 个, 残差={res_greedy:.4e}")
    return quantized, x_est


if __name__ == "__main__":
    demo()
