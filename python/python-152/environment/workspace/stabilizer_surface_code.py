"""
Stabilizer surface code construction, boundary analysis, and sparse matrix handling.

Incorporates:
- 110_boundary_word_square: boundary word / cyclic boundary topology
- 113_box_distance: distance metrics for code distance analysis
- 1155_st_to_crs: sparse matrix CRS conversion for parity check matrices
"""
import numpy as np
from scipy.sparse import csr_matrix
from utils import binary_gaussian_elimination, symplectic_inner_product, hamming_weight


class SurfaceCode:
    """
    Kitaev's surface code on an L × L square lattice with periodic or open boundary.

    Stabilizers:
        A_v = ∏_{e∈δ(v)} X_e   (star / vertex operators)
        B_p = ∏_{e∈∂p} Z_e     (plaquette operators)

    For a periodic (toric) code on L × L vertices:
        n = 2 L² qubits (edges)
        k = 2 logical qubits
        d = L

    For a planar code with smooth/rough boundaries:
        n ≈ 2 L²
        k = 1
        d = L
    """

    def __init__(self, L: int, boundary: str = "toric"):
        if L < 2:
            raise ValueError("Lattice size L must be >= 2.")
        self.L = L
        self.boundary = boundary
        self.n_qubits = None
        self.n_stabilizers = None
        self.n_logical = None
        self.distance = None
        self.Hx = None  # X-stabilizer parity check (Z-type syndrome)
        self.Hz = None  # Z-stabilizer parity check (X-type syndrome)
        self._build_code()

    def _build_code(self):
        L = self.L
        if self.boundary == "toric":
            # Periodic boundary: vertices on L x L grid
            n_vertices = L * L
            n_faces = L * L
            self.n_qubits = 2 * L * L  # horizontal + vertical edges
            # Index edges: horizontal e_h(v) for each vertex v, vertical e_v(v) for each vertex v
            # Hx (star): one row per vertex, cols are horizontal edges around it
            self.Hx = np.zeros((n_vertices, self.n_qubits), dtype=int)
            # Hz (plaquette): one row per face, cols are vertical edges around it
            self.Hz = np.zeros((n_faces, self.n_qubits), dtype=int)

            def h_edge(r, c):
                return r * L + c

            def v_edge(r, c):
                return L * L + r * L + c

            for r in range(L):
                for c in range(L):
                    v = r * L + c
                    # Star operator X on 4 edges: up, down, left, right
                    # Left horizontal edge at (r, c)
                    self.Hx[v, h_edge(r, c)] = 1
                    # Right horizontal edge at (r, (c-1)%L)
                    self.Hx[v, h_edge(r, (c - 1) % L)] = 1
                    # Up vertical edge at (r, c)
                    self.Hx[v, v_edge(r, c)] = 1
                    # Down vertical edge at ((r-1)%L, c)
                    self.Hx[v, v_edge((r - 1) % L, c)] = 1

                    # Plaquette operator Z on 4 edges around face (r,c)
                    f = r * L + c
                    self.Hz[f, h_edge(r, c)] = 1
                    self.Hz[f, h_edge((r + 1) % L, c)] = 1
                    self.Hz[f, v_edge(r, c)] = 1
                    self.Hz[f, v_edge(r, (c + 1) % L)] = 1

            self.n_stabilizers = n_vertices + n_faces - 2  # 2 redundant
            self.n_logical = 2
            self.distance = L
        else:
            # Planar code with open boundaries (smooth on left/right, rough on top/bottom)
            # Vertices: (L+1) x (L+1), but boundary vertices have partial stars
            # Edges: horizontal (L x (L+1)) + vertical ((L+1) x L) = 2L(L+1)
            n_h = L * (L + 1)
            n_v = (L + 1) * L
            self.n_qubits = n_h + n_v
            # Stars for interior vertices: L x L
            self.Hx = np.zeros((L * L, self.n_qubits), dtype=int)
            # Plaquettes for interior faces: L x L
            self.Hz = np.zeros((L * L, self.n_qubits), dtype=int)

            def h_edge_p(r, c):
                return r * (L + 1) + c

            def v_edge_p(r, c):
                return n_h + r * L + c

            for r in range(L):
                for c in range(L):
                    v = r * L + c
                    # Star at vertex (r+1, c+1) in 1-indexed interior
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
        """
        Full symplectic stabilizer matrix S of shape (m, 2n).
        Row i = (x_part | z_part).
        """
        n = self.n_qubits
        m = self.Hx.shape[0] + self.Hz.shape[0]
        S = np.zeros((m, 2 * n), dtype=int)
        S[:self.Hx.shape[0], :n] = self.Hx
        S[self.Hx.shape[0]:, n:] = self.Hz
        return S

    def convert_to_crs(self, H: np.ndarray) -> tuple:
        """
        Convert dense parity check matrix H to Compressed Row Storage (CRS).
        Returns (val, col_ind, row_ptr).
        Adapted from 1155_st_to_crs (ST/GE to CRS).
        """
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
        """Return scipy CSR sparse parity check matrix."""
        if which == "x":
            return csr_matrix(self.Hx)
        elif which == "z":
            return csr_matrix(self.Hz)
        else:
            raise ValueError("which must be 'x' or 'z'.")

    def compute_code_distance_brute_force(self) -> int:
        """
        Compute code distance d = min_{E ∈ N(S)\S} |E|.
        Brute-force over logical operator candidates using centralizer.
        For small L only (L ≤ 4).
        """
        if self.L > 4:
            # Use analytical value for toric/planar code
            return self.distance
        S = self.get_parity_check_matrix()
        n = self.n_qubits
        # Compute centralizer basis (nullspace of symplectic form)
        from utils import stabilizer_centralizer
        C = stabilizer_centralizer(S)
        if C.shape[0] == 0:
            return 0
        d_min = n + 1
        # Enumerate all combinations of centralizer basis vectors
        nc = C.shape[0]
        for mask in range(1, 1 << nc):
            vec = np.zeros(2 * n, dtype=int)
            for i in range(nc):
                if (mask >> i) & 1:
                    vec = (vec + C[i]) % 2
            # Check if in S
            in_S = False
            # S may be redundant; check if vec is linear combination of S rows
            Sext = np.vstack([S, vec])
            _, rank1, _ = binary_gaussian_elimination(S)
            _, rank2, _ = binary_gaussian_elimination(Sext)
            in_S = (rank1 == rank2)
            if not in_S:
                w = hamming_weight(vec[:n]) + hamming_weight(vec[n:])
                # Pauli weight of operator represented by vec is number of qubits
                # where X or Z (or both Y) act nontrivially
                x_part = vec[:n]
                z_part = vec[n:]
                w_pauli = int(np.sum((x_part + z_part) > 0))
                d_min = min(d_min, w_pauli)
        return d_min

    def boundary_word_topology(self) -> str:
        """
        Encode the cyclic boundary structure as a boundary word.
        For toric code, boundary word is cyclic: r^L u^L l^L d^L repeated.
        Adapted from 110_boundary_word_square.
        """
        L = self.L
        if self.boundary == "toric":
            # Torus has no boundary word in the polyomino sense,
            # but we encode the homology cycles.
            word = "r" * L + "u" * L + "l" * L + "d" * L
            return word
        else:
            # Planar code: smooth boundaries left/right, rough top/bottom
            word = "r" * L + "u" * L + "l" * L + "d" * L
            return word

    def box_distance_logical_operators(self) -> dict:
        """
        Compute minimum box distance between equivalent logical operators.
        Uses metric from 113_box_distance: distance in the embedding space.
        For toric code, logical operators wrap around the torus.
        Returns distances for X_L and Z_L logical operators.
        """
        if self.boundary != "toric":
            return {"dx": float(self.L), "dz": float(self.L)}
        L = self.L
        # X logical runs horizontally: distance = L (wrap around)
        # Z logical runs vertically: distance = L
        dx = float(L)
        dz = float(L)
        # For a more refined analysis using Euclidean distance on embedded torus:
        # Average distance between random points on L x L torus = (L/√π) for Gaussian measure
        # We keep the analytical code distance
        return {"dx": dx, "dz": dz, "mean_box": L / np.sqrt(np.pi)}

    def syndrome_of_error(self, error_vec: np.ndarray) -> np.ndarray:
        """
        Compute syndrome s = H · e (mod 2).
        error_vec is binary vector of length 2n (x|z).
        Returns syndrome as binary vector of length m.
        """
        # TODO: Implement syndrome computation for parity check matrix.
        # The stabilizer matrix S has shape (m, 2n) with S = [Hx | 0; 0 | Hz].
        # Syndrome is computed as s = S @ error_vec (mod 2).
        raise NotImplementedError("Hole 1: syndrome_of_error to be implemented.")

    def logical_error_indicator(self, recovery: np.ndarray, error: np.ndarray) -> np.ndarray:
        """
        Determine if recovery · error forms a nontrivial logical operator.
        Returns binary vector indicating which logical qubits are flipped.
        For toric code with 2 logical qubits.
        """
        combined = (recovery + error) % 2
        # Simplified: check homology class of combined error
        # In practice, this would involve checking commutation with logical operators
        n = self.n_qubits
        # Logical X operator on first logical qubit: horizontal loop on top row
        xl1 = np.zeros(2 * n, dtype=int)
        if self.boundary == "toric":
            for c in range(self.L):
                xl1[c] = 1  # horizontal edges on row 0
        # Logical Z operator on first logical qubit: vertical loop on left column
        zl1 = np.zeros(2 * n, dtype=int)
        if self.boundary == "toric":
            for r in range(self.L):
                zl1[self.L * self.L + r * self.L] = 1  # vertical edges on col 0
        ind_x = symplectic_inner_product(combined, zl1) % 2
        ind_z = symplectic_inner_product(combined, xl1) % 2
        return np.array([ind_x, ind_z])
