
import numpy as np


class FEM1DRadial:

    def __init__(self, r_min: float, r_max: float, n_elements: int):
        if r_min <= 0.0:
            raise ValueError("径向内半径必须为正")
        if r_max <= r_min:
            raise ValueError("径向外半径必须大于内半径")
        if n_elements < 2:
            raise ValueError("单元数至少为2")
        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.n_elements = int(n_elements)
        self.n_nodes = n_elements + 1


        self.nodes = np.linspace(r_min, r_max, self.n_nodes)
        self.h = np.diff(self.nodes)



        self.ibc = 1
        self.ul = 0.0
        self.ur = 0.0


        self.indx = np.zeros(self.n_nodes, dtype=int)
        self._setup_indices()

    def _setup_indices(self):
        nu = 0

        self.indx[0] = nu
        nu += 1

        for i in range(1, self.n_elements):
            self.indx[i] = nu
            nu += 1

        self.indx[self.n_elements] = -1
        self.nu = nu

    def _basis(self, ie: int, il: int, r: float) -> tuple:
        r_left = self.nodes[ie]
        r_right = self.nodes[ie + 1]
        h_e = self.h[ie]
        if r < r_left or r > r_right:
            return 0.0, 0.0
        if il == 0:
            phi = (r_right - r) / h_e
            dphi = -1.0 / h_e
        else:
            phi = (r - r_left) / h_e
            dphi = 1.0 / h_e
        return phi, dphi

    def assemble(
        self, nu_func, source_func, nquad: int = 3
    ) -> tuple:

        f = np.zeros(self.nu)
        adiag = np.zeros(self.nu)
        aleft = np.zeros(self.nu)
        arite = np.zeros(self.nu)


        from quadrature_engine import GaussLegendreQuadrature

        glq = GaussLegendreQuadrature(nquad)

        for ie in range(self.n_elements):
            r_left = self.nodes[ie]
            r_right = self.nodes[ie + 1]
            h_e = self.h[ie]
            scale = 0.5 * h_e
            shift = 0.5 * (r_left + r_right)

            quad_pts = shift + scale * glq.nodes
            quad_wts = scale * glq.weights

            for iq in range(nquad):
                rq = quad_pts[iq]
                wq = quad_wts[iq]
                nu_r = nu_func(rq)
                js_r = source_func(rq)

                for il in range(2):
                    ig = ie + il
                    iu = self.indx[ig]
                    if iu < 0:
                        continue

                    phi_i, dphi_i = self._basis(ie, il, rq)
                    f[iu] += wq * js_r * phi_i



                    for jl in range(2):
                        jg = ie + jl
                        ju = self.indx[jg]
                        phi_j, dphi_j = self._basis(ie, jl, rq)



                        aij = wq * (
                            nu_r * rq * dphi_i * dphi_j
                            + (nu_r / rq) * phi_i * phi_j
                        )

                        if ju < 0:

                            if jg == self.n_elements:
                                f[iu] -= aij * self.ul
                        elif iu == ju:
                            adiag[iu] += aij
                        elif ju < iu:
                            aleft[iu] += aij
                        else:
                            arite[iu] += aij

        return adiag, aleft, arite, f

    @staticmethod
    def solve_tridiagonal(adiag: np.ndarray, aleft: np.ndarray, arite: np.ndarray, f: np.ndarray) -> np.ndarray:
        n = len(f)
        adiag = adiag.copy()
        f = f.copy()


        if abs(adiag[0]) < 1.0e-30:
            adiag[0] = 1.0e-30 * np.sign(adiag[0] + 1.0e-30)


        for i in range(1, n):
            m = aleft[i] / adiag[i - 1]
            adiag[i] -= m * arite[i - 1]
            f[i] -= m * f[i - 1]
            if abs(adiag[i]) < 1.0e-30:
                adiag[i] = 1.0e-30 * np.sign(adiag[i] + 1.0e-30)


        u = np.zeros(n)
        u[-1] = f[-1] / adiag[-1]
        for i in range(n - 2, -1, -1):
            u[i] = (f[i] - arite[i] * u[i + 1]) / adiag[i]
        return u

    def solve(self, nu_func, source_func, nquad: int = 3) -> np.ndarray:
        adiag, aleft, arite, f = self.assemble(nu_func, source_func, nquad)
        u_sol = self.solve_tridiagonal(adiag, aleft, arite, f)


        A_full = np.zeros(self.n_nodes)
        for i in range(self.n_nodes):
            idx = self.indx[i]
            if idx >= 0:
                A_full[i] = u_sol[idx]
            else:
                A_full[i] = self.ul
        return A_full

    def compute_radial_b_field(self, A: np.ndarray) -> np.ndarray:

        raise NotImplementedError("Hole_3: 需实现由 A_θ 计算 B_r 的公式")

    def compute_energy(self, A: np.ndarray, nu_func, nquad: int = 3) -> float:
        from quadrature_engine import GaussLegendreQuadrature

        glq = GaussLegendreQuadrature(nquad)
        W = 0.0
        L = 1.0

        for ie in range(self.n_elements):
            r_left = self.nodes[ie]
            r_right = self.nodes[ie + 1]
            h_e = self.h[ie]
            scale = 0.5 * h_e
            shift = 0.5 * (r_left + r_right)
            quad_pts = shift + scale * glq.nodes
            quad_wts = scale * glq.weights


            A_i = A[ie]
            A_ip1 = A[ie + 1]
            dA = (A_ip1 - A_i) / h_e

            for iq in range(nquad):
                rq = quad_pts[iq]
                wq = quad_wts[iq]
                Aq = A_i + dA * (rq - r_left)

                Bq = Aq / rq + dA
                nu_r = nu_func(rq)
                W += 0.5 * nu_r * Bq * Bq * 2.0 * np.pi * rq * L * wq

        return W
