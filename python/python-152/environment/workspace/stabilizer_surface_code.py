import numpy as np
from scipy.sparse import csr_matrix
from utils import binary_gaussian_elimination, symplectic_inner_product, hamming_weight


class SurfaceCode:

    def __init__(self, L: int, boundary: str = "toric"):
        if L < 2:
            raise ValueError("Lattice size L must be >= 2.")
        self.L = L
        self.boundary = boundary
        self.n_qubits = None
        self.n_stabilizers = None
        self.n_logical = None
        self.distance = None
        self.Hx = None
        self.Hz = None
        self._build_code()

    def _build_code(self):
        L = self.L
        if self.boundary == "toric":

            n_vertices = L * L
            n_faces = L * L
            self.n_qubits = 2 * L * L


            self.Hx = np.zeros((n_vertices, self.n_qubits), dtype=int)

            self.Hz = np.zeros((n_faces, self.n_qubits), dtype=int)

            def h_edge(r, c):
                return r * L + c

            def v_edge(r, c):
                return L * L + r * L + c

            for r in range(L):
                for c in range(L):
                    v = r * L + c


                    self.Hx[v, h_edge(r, c)] = 1

                    self.Hx[v, h_edge(r, (c - 1) % L)] = 1

                    self.Hx[v, v_edge(r, c)] = 1

                    self.Hx[v, v_edge((r - 1) % L, c)] = 1


                    f = r * L + c
                    self.Hz[f, h_edge(r, c)] = 1
                    self.Hz[f, h_edge((r + 1) % L, c)] = 1
                    self.Hz[f, v_edge(r, c)] = 1
                    self.Hz[f, v_edge(r, (c + 1) % L)] = 1

            self.n_stabilizers = n_vertices + n_faces - 2
            self.n_logical = 2
            self.distance = L
        else:



            n_h = L * (L + 1)
            n_v = (L + 1) * L
            self.n_qubits = n_h + n_v

            self.Hx = np.zeros((L * L, self.n_qubits), dtype=int)

            self.Hz = np.zeros((L * L, self.n_qubits), dtype=int)

            def h_edge_p(r, c):
                return r * (L + 1) + c

            def v_edge_p(r, c):
                return n_h + r * L + c

            for r in range(L):
                for c in range(L):
                    v = r * L + c

                    self.Hx[v, h_edge_p(r, c + 1)] = 1
                    self.Hx[v, h_edge_p(r + 1, c + 1)] = 1
                    self.Hx[v, v_edge_p(r + 1, c)] = 1
                    self.Hx[v, v_edge_p(r + 1, c + 1)] = 1

                    f = r * L + c
                    self.Hz[f, h_edge_p(r, c + 1)] = 1
                    self.Hz[f, h_edge_p(r + 1, c + 1)] = 1
                    self.Hz[f, v_edge_p(r + 1, c)] = 1
                    self.Hz[f, v_edge_p(r + 1, c + 1)] = 1

            self.n_stabilizers = 2 * L * L
            self.n_logical = 1
            self.distance = L

    def get_parity_check_matrix(self) -> np.ndarray:
        n = self.n_qubits
        m = self.Hx.shape[0] + self.Hz.shape[0]
        S = np.zeros((m, 2 * n), dtype=int)
        S[:self.Hx.shape[0], :n] = self.Hx
        S[self.Hx.shape[0]:, n:] = self.Hz
        return S

    def convert_to_crs(self, H: np.ndarray) -> tuple:
        m, n = H.shape
        val = []
        col_ind = []
        row_ptr = [0]
        for i in range(m):
            row_nnz = 0
            for j in range(n):
                if H[i, j] != 0:
                    val.append(float(H[i, j]))
                    col_ind.append(j)
                    row_nnz += 1
            row_ptr.append(row_ptr[-1] + row_nnz)
        return np.array(val), np.array(col_ind, dtype=int), np.array(row_ptr, dtype=int)

    def sparse_parity_check(self, which: str = "x") -> csr_matrix:
        if which == "x":
            return csr_matrix(self.Hx)
        elif which == "z":
            return csr_matrix(self.Hz)
        else:
            raise ValueError("which must be 'x' or 'z'.")

    def compute_code_distance_brute_force(self) -> int:
        if self.L > 4:

            return self.distance
        S = self.get_parity_check_matrix()
        n = self.n_qubits

        from utils import stabilizer_centralizer
        C = stabilizer_centralizer(S)
        if C.shape[0] == 0:
            return 0
        d_min = n + 1

        nc = C.shape[0]
        for mask in range(1, 1 << nc):
            vec = np.zeros(2 * n, dtype=int)
            for i in range(nc):
                if (mask >> i) & 1:
                    vec = (vec + C[i]) % 2

            in_S = False

            Sext = np.vstack([S, vec])
            _, rank1, _ = binary_gaussian_elimination(S)
            _, rank2, _ = binary_gaussian_elimination(Sext)
            in_S = (rank1 == rank2)
            if not in_S:
                w = hamming_weight(vec[:n]) + hamming_weight(vec[n:])


                x_part = vec[:n]
                z_part = vec[n:]
                w_pauli = int(np.sum((x_part + z_part) > 0))
                d_min = min(d_min, w_pauli)
        return d_min

    def boundary_word_topology(self) -> str:
        L = self.L
        if self.boundary == "toric":


            word = "r" * L + "u" * L + "l" * L + "d" * L
            return word
        else:

            word = "r" * L + "u" * L + "l" * L + "d" * L
            return word

    def box_distance_logical_operators(self) -> dict:
        if self.boundary != "toric":
            return {"dx": float(self.L), "dz": float(self.L)}
        L = self.L


        dx = float(L)
        dz = float(L)



        return {"dx": dx, "dz": dz, "mean_box": L / np.sqrt(np.pi)}

    def syndrome_of_error(self, error_vec: np.ndarray) -> np.ndarray:



        raise NotImplementedError("Hole 1: syndrome_of_error to be implemented.")

    def logical_error_indicator(self, recovery: np.ndarray, error: np.ndarray) -> np.ndarray:
        combined = (recovery + error) % 2


        n = self.n_qubits

        xl1 = np.zeros(2 * n, dtype=int)
        if self.boundary == "toric":
            for c in range(self.L):
                xl1[c] = 1

        zl1 = np.zeros(2 * n, dtype=int)
        if self.boundary == "toric":
            for r in range(self.L):
                zl1[self.L * self.L + r * self.L] = 1
        ind_x = symplectic_inner_product(combined, zl1) % 2
        ind_z = symplectic_inner_product(combined, xl1) % 2
        return np.array([ind_x, ind_z])
