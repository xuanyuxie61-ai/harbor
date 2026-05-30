
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


class MaxwellFEM2D:

    def __init__(self, wavelength=1.55e-6, n_si=3.48, n_air=1.0,
                 pml_width=0.5e-6, pml_sigma_max=1.0e7):
        self.wavelength = wavelength
        self.k0 = 2.0 * np.pi / wavelength
        self.n_si = n_si
        self.n_air = n_air
        self.eps_si = n_si ** 2
        self.eps_air = n_air ** 2
        self.pml_width = pml_width
        self.pml_sigma_max = pml_sigma_max
        self.c = 2.99792458e8
        self.omega = 2.0 * np.pi * self.c / wavelength




    def build_rectangular_mesh(self, nx, ny, xlim, ylim):
        xl, xr = xlim
        yb, yt = ylim
        dx = (xr - xl) / (nx - 1)
        dy = (yt - yb) / (ny - 1)


        node_num = (2 * nx - 1) * (2 * ny - 1)
        nodes = np.zeros((node_num, 2), dtype=np.float64)

        idx = 0
        for j in range(2 * ny - 1):
            for i in range(2 * nx - 1):
                if j % 2 == 0:
                    y = yb + (j // 2) * dy
                else:
                    y = yb + (j // 2) * dy + dy / 2.0
                if i % 2 == 0:
                    x = xl + (i // 2) * dx
                else:
                    x = xl + (i // 2) * dx + dx / 2.0
                nodes[idx] = [x, y]
                idx += 1

        element_num = 2 * (nx - 1) * (ny - 1)
        elements = np.zeros((element_num, 6), dtype=np.int32)
        e = 0
        for j in range(ny - 1):
            for i in range(nx - 1):
                sw = j * 2 * (2 * nx - 1) + 2 * i
                w = sw + 1
                nw = sw + 2
                s = sw + (2 * nx - 1)
                c = s + 1
                n = s + 2
                se = s + (2 * nx - 1)
                ee = se + 1
                ne = se + 2

                elements[e] = [sw, se, nw, s, c, w]
                e += 1
                elements[e] = [ne, nw, se, n, c, ee]
                e += 1

        return nodes, elements




    def epsilon_profile(self, x, y, pillar_center, pillar_size):


        cx, cy = pillar_center
        w, h = pillar_size
        ...
        eps = ...
        return eps.astype(np.complex128)

    def pml_stretch(self, x, y, xlim, ylim):
        xl, xr = xlim
        yb, yt = ylim
        m_order = 2
        sx = 1.0 + 0.0j
        sy = 1.0 + 0.0j

        if x < xl + self.pml_width:
            d = xl + self.pml_width - x
            sigma = self.pml_sigma_max * (d / self.pml_width) ** m_order
            sx = 1.0 - 1.0j * sigma / self.omega
        elif x > xr - self.pml_width:
            d = x - (xr - self.pml_width)
            sigma = self.pml_sigma_max * (d / self.pml_width) ** m_order
            sx = 1.0 - 1.0j * sigma / self.omega

        if y < yb + self.pml_width:
            d = yb + self.pml_width - y
            sigma = self.pml_sigma_max * (d / self.pml_width) ** m_order
            sy = 1.0 - 1.0j * sigma / self.omega
        elif y > yt - self.pml_width:
            d = y - (yt - self.pml_width)
            sigma = self.pml_sigma_max * (d / self.pml_width) ** m_order
            sy = 1.0 - 1.0j * sigma / self.omega

        return sx, sy




    @staticmethod
    def quad_points_t6():

        qp = np.array([
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            [0.059715871789770, 0.470142064105115, 0.470142064105115],
            [0.470142064105115, 0.059715871789770, 0.470142064105115],
            [0.470142064105115, 0.470142064105115, 0.059715871789770],
            [0.797426985353087, 0.101286507323456, 0.101286507323456],
            [0.101286507323456, 0.797426985353087, 0.101286507323456],
            [0.101286507323456, 0.101286507323456, 0.797426985353087],
        ], dtype=np.float64)
        wq = np.array([
            0.225000000000000,
            0.132394152788506,
            0.132394152788506,
            0.132394152788506,
            0.125939180544827,
            0.125939180544827,
            0.125939180544827,
        ], dtype=np.float64)
        return qp, wq

    def shape_t6(self, r, s):
        t = 1.0 - r - s
        N = np.array([
            2.0 * t * (t - 0.5),
            2.0 * r * (r - 0.5),
            2.0 * s * (s - 0.5),
            4.0 * r * t,
            4.0 * r * s,
            4.0 * s * t,
        ], dtype=np.float64)
        dNdr = np.array([
            -3.0 + 4.0 * r + 4.0 * s,
            4.0 * r - 1.0,
            0.0,
            4.0 * (1.0 - 2.0 * r - s),
            4.0 * s,
            -4.0 * s,
        ], dtype=np.float64)
        dNds = np.array([
            -3.0 + 4.0 * r + 4.0 * s,
            0.0,
            4.0 * s - 1.0,
            -4.0 * r,
            4.0 * r,
            4.0 * (1.0 - r - 2.0 * s),
        ], dtype=np.float64)
        return N, dNdr, dNds




    def assemble_system(self, nodes, elements, pillar_center, pillar_size,
                        xlim, ylim, source_type='plane_wave'):
        node_num = nodes.shape[0]
        element_num = elements.shape[0]
        qp, wq = self.quad_points_t6()

        row_idx = []
        col_idx = []
        data = []
        b = np.zeros(node_num, dtype=np.complex128)

        for e in range(element_num):
            en = elements[e]
            xn = nodes[en, 0]
            yn = nodes[en, 1]





            x1, x2, x3 = xn[0], xn[1], xn[2]
            y1, y2, y3 = yn[0], yn[1], yn[2]
            detJ = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
            if abs(detJ) < 1e-18:
                continue
            area = 0.5 * abs(detJ)



            for iq in range(len(wq)):
                L1, L2, L3 = qp[iq]
                r = L2
                s = L3
                w = area * wq[iq]

                N, dNdr, dNds = self.shape_t6(r, s)


                x = np.dot(N, xn)
                y = np.dot(N, yn)


                dxdr = np.dot(dNdr, xn)
                dxds = np.dot(dNds, xn)
                dydr = np.dot(dNdr, yn)
                dyds = np.dot(dNds, yn)
                detJ_q = dxdr * dyds - dxds * dydr
                if abs(detJ_q) < 1e-18:
                    continue


                inv_det = 1.0 / detJ_q
                dNdx = (dNdr * dyds - dNds * dydr) * inv_det
                dNdy = (dNds * dxdr - dNdr * dxds) * inv_det


                eps = self.epsilon_profile(x, y, pillar_center, pillar_size)
                sx, sy = self.pml_stretch(x, y, xlim, ylim)


                for i in range(6):
                    for j in range(6):
                        aij = (sy / sx) * dNdx[i] * dNdx[j] + (sx / sy) * dNdy[i] * dNdy[j]
                        aij -= self.k0 ** 2 * sx * sy * eps * N[i] * N[j]
                        aij *= w * abs(detJ_q)
                        row_idx.append(en[i])
                        col_idx.append(en[j])
                        data.append(aij)


                if source_type == 'plane_wave':

                    f_val = -self.k0 ** 2 * (eps - self.eps_air) * np.exp(1.0j * self.k0 * x)
                else:
                    f_val = 0.0
                for i in range(6):
                    b[en[i]] += w * abs(detJ_q) * f_val * N[i]

        A = sparse.coo_matrix((data, (row_idx, col_idx)),
                               shape=(node_num, node_num), dtype=np.complex128).tocsc()
        return A, b

    def apply_dirichlet_boundary(self, A, b, nodes, xlim, ylim):
        xl, xr = xlim
        yb, yt = ylim
        tol = 1e-12
        boundary_nodes = []
        for i, (x, y) in enumerate(nodes):
            if (abs(x - xl) < tol or abs(x - xr) < tol or
                    abs(y - yb) < tol or abs(y - yt) < tol):
                boundary_nodes.append(i)

        boundary_nodes = np.array(boundary_nodes, dtype=np.int32)
        if len(boundary_nodes) == 0:
            return A, b


        for idx in boundary_nodes:


            pass

        A_lil = A.tolil()
        for idx in boundary_nodes:
            A_lil.rows[idx] = [idx]
            A_lil.data[idx] = [1.0 + 0.0j]
            b[idx] = 0.0 + 0.0j


        for i in range(A_lil.shape[0]):
            if i in boundary_nodes:
                continue
            new_rows = []
            new_data = []
            for j, val in zip(A_lil.rows[i], A_lil.data[i]):
                if j not in boundary_nodes:
                    new_rows.append(j)
                    new_data.append(val)
            A_lil.rows[i] = new_rows
            A_lil.data[i] = new_data

        return A_lil.tocsc(), b

    def solve_scattering(self, nx=25, ny=25,
                         domain=(-2.0e-6, 2.0e-6, -2.0e-6, 2.0e-6),
                         pillar_center=(0.0, 0.0),
                         pillar_size=(0.5e-6, 1.0e-6)):
        xlim = (domain[0], domain[1])
        ylim = (domain[2], domain[3])
        nodes, elements = self.build_rectangular_mesh(nx, ny, xlim, ylim)
        A, b = self.assemble_system(nodes, elements, pillar_center, pillar_size,
                                     xlim, ylim)
        A, b = self.apply_dirichlet_boundary(A, b, nodes, xlim, ylim)
        E_z = spsolve(A, b)
        return E_z, nodes, elements

    def compute_transmission_phase(self, E_z, nodes, y_eval):

        tol = 1e-12
        line_nodes = np.where(np.abs(nodes[:, 1] - y_eval) < tol)[0]
        if len(line_nodes) == 0:

            line_nodes = np.argsort(np.abs(nodes[:, 1] - y_eval))[:nodes.shape[0] // nx]
        x_vals = nodes[line_nodes, 0]
        E_vals = E_z[line_nodes]

        order = np.argsort(x_vals)
        return x_vals[order], E_vals[order]


def demo():
    fem = MaxwellFEM2D(wavelength=1.55e-6)
    E_z, nodes, elements = fem.solve_scattering(
        nx=17, ny=17,
        domain=(-1.5e-6, 1.5e-6, -1.5e-6, 1.5e-6),
        pillar_center=(0.0, 0.0),
        pillar_size=(0.3e-6, 0.6e-6)
    )
    phase = np.angle(E_z)
    print(f"[maxwell_fem] 节点数: {len(E_z)}, 相位范围: [{phase.min():.4f}, {phase.max():.4f}] rad")

    cx, cy = 0.0, 0.0
    dist = np.sqrt((nodes[:, 0] - cx) ** 2 + (nodes[:, 1] - cy) ** 2)
    mask = dist < 0.2e-6
    if mask.any():
        avg_phase = np.angle(np.mean(E_z[mask]))
        print(f"[maxwell_fem] 中心区域平均透射相位: {avg_phase:.4f} rad = {np.degrees(avg_phase):.2f}°")
    return E_z, nodes, elements


if __name__ == "__main__":
    demo()
