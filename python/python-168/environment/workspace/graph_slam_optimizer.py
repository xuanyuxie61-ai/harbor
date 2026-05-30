
import numpy as np
from sparse_matrix_ops import SORSolver


class PoseGraph:

    def __init__(self):
        self.poses = []
        self.edges = []

    def add_vertex(self, pose):
        pose = np.asarray(pose, dtype=np.float64)
        if pose.shape != (3,):
            raise ValueError("pose must be 3-vector")
        self.poses.append(pose)
        return len(self.poses) - 1

    def add_edge(self, i, j, measurement, information):
        if i < 0 or i >= len(self.poses) or j < 0 or j >= len(self.poses):
            raise ValueError("vertex index out of range")
        measurement = np.asarray(measurement, dtype=np.float64)
        information = np.asarray(information, dtype=np.float64)
        if measurement.shape != (3,) or information.shape != (3, 3):
            raise ValueError("measurement/information shape mismatch")
        self.edges.append({
            'i': int(i), 'j': int(j),
            'measurement': measurement,
            'information': information
        })

    def get_state_vector(self):
        return np.hstack(self.poses)

    def set_state_vector(self, vec):
        n = len(self.poses)
        if vec.shape[0] != 3 * n:
            raise ValueError("state vector dimension mismatch")
        self.poses = [vec[3*i:3*i+3].copy() for i in range(n)]


class GraphSLAMOptimizer:

    def __init__(self, max_iterations=50, linear_solver='sor', tol=1e-6):
        self.max_iterations = max(int(max_iterations), 1)
        self.tol = max(float(tol), 1e-15)
        self.linear_solver = linear_solver
        self.sor_solver = SORSolver(omega=1.5, max_iter=2000, tol=1e-10)

    def optimize(self, graph):
        n = len(graph.poses)
        if n < 2:
            return graph, 0.0, 0

        lambda_lm = 1e-4
        prev_cost = self._compute_cost(graph)

        for iteration in range(self.max_iterations):
            H, b = self._build_linear_system(graph)


            H[0:3, :] = 0.0
            H[:, 0:3] = 0.0
            H[0:3, 0:3] = np.eye(3)
            b[0:3] = 0.0


            H_lm = H + lambda_lm * np.eye(H.shape[0])


            try:
                delta = np.linalg.solve(H_lm, -b)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(H_lm, -b, rcond=None)[0]


            test_graph = PoseGraph()
            test_graph.poses = [p.copy() for p in graph.poses]
            test_graph.edges = list(graph.edges)
            self._apply_delta(test_graph, delta)
            test_cost = self._compute_cost(test_graph)

            if test_cost < prev_cost and np.isfinite(test_cost):
                graph = test_graph
                prev_cost = test_cost
                lambda_lm = max(lambda_lm * 0.1, 1e-12)
            else:
                lambda_lm = min(lambda_lm * 10.0, 1e8)
                if lambda_lm > 1e6:
                    break

            if abs(prev_cost - test_cost) < self.tol:
                break

        return graph, prev_cost, iteration + 1

    def _build_linear_system(self, graph):
        n = len(graph.poses)
        dim = 3 * n
        H = np.zeros((dim, dim), dtype=np.float64)
        b = np.zeros(dim, dtype=np.float64)

        for edge in graph.edges:
            i = edge['i']
            j = edge['j']
            z = edge['measurement']
            omega = edge['information']

            e, Ji, Jj = self._compute_error_and_jacobians(graph.poses[i], graph.poses[j], z)

            ii, jj = 3*i, 3*j
            H[ii:ii+3, ii:ii+3] += Ji.T @ omega @ Ji
            H[ii:ii+3, jj:jj+3] += Ji.T @ omega @ Jj
            H[jj:jj+3, ii:ii+3] += Jj.T @ omega @ Ji
            H[jj:jj+3, jj:jj+3] += Jj.T @ omega @ Jj
            b[ii:ii+3] += Ji.T @ omega @ e
            b[jj:jj+3] += Jj.T @ omega @ e

        return H, b

    @staticmethod
    def _compute_error_and_jacobians(xi, xj, z):







        raise NotImplementedError("Hole 1: SE(2) error and Jacobians not implemented")

    @staticmethod
    def _apply_delta(graph, delta):
        for i in range(len(graph.poses)):
            d = delta[3*i:3*i+3]
            x, y, th = graph.poses[i]
            c, s = np.cos(th), np.sin(th)
            graph.poses[i] = np.array([
                x + c*d[0] - s*d[1],
                y + s*d[0] + c*d[1],
                th + d[2]
            ], dtype=np.float64)

            while graph.poses[i][2] > np.pi:
                graph.poses[i][2] -= 2.0 * np.pi
            while graph.poses[i][2] < -np.pi:
                graph.poses[i][2] += 2.0 * np.pi

    @staticmethod
    def _compute_cost(graph):
        cost = 0.0
        for edge in graph.edges:
            e, _, _ = GraphSLAMOptimizer._compute_error_and_jacobians(
                graph.poses[edge['i']], graph.poses[edge['j']], edge['measurement']
            )
            cost += e.T @ edge['information'] @ e
        return cost


class ObservabilityAnalyzer:

    @staticmethod
    def analyze_hessian(H, graph_n_poses):
        H_sym = 0.5 * (H + H.T)
        try:
            eigenvalues = np.linalg.eigvalsh(H_sym)
        except np.linalg.LinAlgError:
            eigenvalues = np.linalg.eigvalsh(H_sym + 1e-10 * np.eye(H.shape[0]))

        eigenvalues = np.sort(eigenvalues)
        abs_eig = np.abs(eigenvalues)
        nonzero = abs_eig[abs_eig > 1e-8]
        condition_number = np.max(nonzero) / np.min(nonzero) if len(nonzero) > 0 else np.inf
        nullity = np.sum(abs_eig < 1e-8)
        observability_index = np.exp(np.mean(np.log(nonzero + 1e-15))) if len(nonzero) > 0 else 0.0

        return {
            'eigenvalues': eigenvalues,
            'condition_number': condition_number,
            'nullity': nullity,
            'observability_index': observability_index,
            'gauge_dofs': 3
        }

    @staticmethod
    def generate_random_schur_matrix(n, lambda_mean=1.0, lambda_dev=0.5):
        T = np.triu(np.random.normal(0, 1, (n, n)))
        diag_vals = np.random.normal(lambda_mean, lambda_dev, n)
        np.fill_diagonal(T, diag_vals)
        M = np.random.normal(0, 1, (n, n))
        Q, _ = np.linalg.qr(M)
        A = Q.T @ T @ Q
        return A, Q, T
