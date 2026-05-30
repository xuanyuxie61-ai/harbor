
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


class FEM2DSemanticProjection:

    def __init__(self, xl: float = 0.0, xr: float = 1.0,
                 yb: float = 0.0, yt: float = 1.0,
                 nx: int = 17, ny: int = 17):
        if nx < 2 or ny < 2:
            raise ValueError(f"nx and ny must be at least 2, got nx={nx}, ny={ny}")

        self.xl = float(xl)
        self.xr = float(xr)
        self.yb = float(yb)
        self.yt = float(yt)
        self.nx = int(nx)
        self.ny = int(ny)

        self._build_mesh()
        self._build_elements()

    def _build_mesh(self):
        self.node_num = self.nx * self.ny
        self.x = np.zeros(self.node_num)
        self.y = np.zeros(self.node_num)

        k = 0
        for j in range(self.ny):
            for i in range(self.nx):
                self.x[k] = ((self.nx - 1 - i) * self.xl + i * self.xr) / (self.nx - 1)
                self.y[k] = ((self.ny - 1 - j) * self.yb + j * self.yt) / (self.ny - 1)
                k += 1

    def _build_elements(self):
        self.element_num = 2 * (self.nx - 1) * (self.ny - 1)
        self.element_node = np.zeros((3, self.element_num), dtype=int)

        k = 0
        for j in range(self.ny - 1):
            for i in range(self.nx - 1):

                self.element_node[0, k] = i + j * self.nx
                self.element_node[1, k] = i + 1 + j * self.nx
                self.element_node[2, k] = i + (j + 1) * self.nx
                k += 1

                self.element_node[0, k] = i + 1 + (j + 1) * self.nx
                self.element_node[1, k] = i + (j + 1) * self.nx
                self.element_node[2, k] = i + 1 + j * self.nx
                k += 1

    def _triangle_area(self, i1: int, i2: int, i3: int) -> float:
        area = 0.5 * abs(
            self.x[i1] * (self.y[i2] - self.y[i3])
            + self.x[i2] * (self.y[i3] - self.y[i1])
            + self.x[i3] * (self.y[i1] - self.y[i2])
        )
        return area

    def _is_boundary_node(self, k: int) -> bool:
        i = k % self.nx
        j = k // self.nx
        return (i == 0 or i == self.nx - 1 or j == 0 or j == self.ny - 1)

    def project(self, semantic_func) -> np.ndarray:
        b = np.zeros(self.node_num)
        A_data = []
        A_row = []
        A_col = []

        for e in range(self.element_num):
            i1 = self.element_node[0, e]
            i2 = self.element_node[1, e]
            i3 = self.element_node[2, e]

            area = self._triangle_area(i1, i2, i3)
            if area < 1e-15:
                continue



            quad_points = [
                (0.5, 0.5, 0.0),
                (0.0, 0.5, 0.5),
                (0.5, 0.0, 0.5),
            ]
            wq = 1.0 / 3.0

            for q_coords in quad_points:
                q1, q2, q3 = q_coords
                xq = q1 * self.x[i1] + q2 * self.x[i2] + q3 * self.x[i3]
                yq = q1 * self.y[i1] + q2 * self.y[i2] + q3 * self.y[i3]

                w = semantic_func(xq, yq)


                for ti1 in range(3):
                    nti1 = self.element_node[ti1, e]



                    phi_val = q_coords[ti1]

                    b[nti1] += area * wq * w * phi_val

                    for tj1 in range(3):
                        ntj1 = self.element_node[tj1, e]
                        phi_j_val = q_coords[tj1]
                        A_data.append(area * wq * phi_val * phi_j_val)
                        A_row.append(nti1)
                        A_col.append(ntj1)


        A_mat = csr_matrix((A_data, (A_row, A_col)), shape=(self.node_num, self.node_num))


        for k in range(self.node_num):
            if self._is_boundary_node(k):
                A_mat[k, :] = 0.0
                A_mat[k, k] = 1.0
                b[k] = semantic_func(self.x[k], self.y[k])


        U = spsolve(A_mat, b)
        return U

    def compute_l2_error(self, U: np.ndarray, semantic_func) -> dict:
        q1 = np.array([0.0, 0.5, 0.5, 4.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
        q2 = np.array([0.5, 0.0, 0.5, 1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0])
        q3 = np.array([0.5, 0.5, 0.0, 1.0 / 6.0, 1.0 / 6.0, 4.0 / 6.0])
        wq = np.array([1.0 / 30.0, 1.0 / 30.0, 1.0 / 30.0,
                       9.0 / 30.0, 9.0 / 30.0, 9.0 / 30.0])

        u_norm = 0.0
        w_norm = 0.0
        uw_norm = 0.0

        for e in range(self.element_num):
            i1 = self.element_node[0, e]
            i2 = self.element_node[1, e]
            i3 = self.element_node[2, e]

            area = self._triangle_area(i1, i2, i3)
            if area < 1e-15:
                continue

            for q in range(6):
                xq = q1[q] * self.x[i1] + q2[q] * self.x[i2] + q3[q] * self.x[i3]
                yq = q1[q] * self.y[i1] + q2[q] * self.y[i2] + q3[q] * self.y[i3]


                u = q1[q] * U[i1] + q2[q] * U[i2] + q3[q] * U[i3]
                w = semantic_func(xq, yq)

                u_norm += area * wq[q] * u ** 2
                w_norm += area * wq[q] * w ** 2
                uw_norm += area * wq[q] * (u - w) ** 2

        return {
            'u_norm': np.sqrt(u_norm),
            'w_norm': np.sqrt(w_norm),
            'uw_norm': np.sqrt(uw_norm),
            'relative_error': np.sqrt(uw_norm) / np.sqrt(w_norm) if w_norm > 1e-15 else float('inf')
        }


def demo():
    print("=" * 60)
    print("2D有限元语义密度投影演示")
    print("=" * 60)

    fem = FEM2DSemanticProjection(xl=0.0, xr=1.0, yb=0.0, yt=1.0, nx=9, ny=9)
    print(f"\n网格节点数: {fem.node_num}")
    print(f"三角形单元数: {fem.element_num}")


    def semantic_density(x, y):
        return np.sin(np.pi * x) * np.sin(np.pi * y) + x

    U = fem.project(semantic_density)
    print(f"\n投影解范围: [{U.min():.6f}, {U.max():.6f}]")

    errors = fem.compute_l2_error(U, semantic_density)
    print(f"\nL2 误差分析:")
    print(f"  |U|   = {errors['u_norm']:.6e}")
    print(f"  |W|   = {errors['w_norm']:.6e}")
    print(f"  |U-W| = {errors['uw_norm']:.6e}")
    print(f"  相对误差 = {errors['relative_error']:.6e}")

    print("\n模块运行完成")
    return fem, U, errors


if __name__ == "__main__":
    demo()
