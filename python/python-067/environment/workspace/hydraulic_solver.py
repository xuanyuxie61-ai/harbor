# -*- coding: utf-8 -*-

import numpy as np
from typing import Tuple, Optional, Callable


class HydraulicSolver:

    def __init__(self, nx: int, ny: int, dx: float, dy: float):
        if nx <= 0 or ny <= 0:
            raise ValueError("nx 和 ny 必须为正")
        if dx <= 0 or dy <= 0:
            raise ValueError("dx 和 dy 必须为正")
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.head = np.zeros((ny, nx))
        self.source = np.zeros((ny, nx))

    def _build_tridiagonal_system(self, T: np.ndarray,
                                  axis: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if axis == 0:
            n = self.nx
            d = self.dx
            m = self.ny
        else:
            n = self.ny
            d = self.dy
            m = self.nx

        a_sub = np.zeros(n)
        a_diag = np.zeros(n)
        a_sup = np.zeros(n)
        rhs = np.zeros(n)



        for i in range(1, n - 1):
            a_sub[i] = -1.0 / (d ** 2)
            a_diag[i] = 2.0 / (d ** 2)
            a_sup[i] = -1.0 / (d ** 2)


        a_diag[0] = 1.0
        a_sup[0] = 0.0
        a_sub[-1] = 0.0
        a_diag[-1] = 1.0

        return a_sub, a_diag, a_sup, rhs

    def solve_gauss_seidel(self, T: np.ndarray,
                           h_boundary: dict = None,
                           max_iter: int = 10000,
                           tol: float = 1.0e-8,
                           omega: float = 1.0) -> np.ndarray:
        if omega <= 0 or omega >= 2:
            raise ValueError("omega 必须在 (0, 2) 区间内")
        if T.shape != (self.ny, self.nx):
            raise ValueError("T 的形状必须与网格匹配")

        if h_boundary is None:
            h_boundary = {'left': 10.0, 'right': 0.0, 'top': 5.0, 'bottom': 5.0}

        h = np.zeros((self.ny, self.nx))

        h[:, 0] = h_boundary.get('left', 0.0)
        h[:, -1] = h_boundary.get('right', 0.0)
        h[0, :] = h_boundary.get('top', 0.0)
        h[-1, :] = h_boundary.get('bottom', 0.0)


        Tx = np.zeros((self.ny, self.nx + 1))
        Ty = np.zeros((self.ny + 1, self.nx))

        Tx[:, 1:-1] = 2.0 * T[:, :-1] * T[:, 1:] / (T[:, :-1] + T[:, 1:] + 1e-20)
        Ty[1:-1, :] = 2.0 * T[:-1, :] * T[1:, :] / (T[:-1, :] + T[1:, :] + 1e-20)

        for it in range(max_iter):
            h_old = h.copy()

            for i in range(1, self.ny - 1):
                for j in range(1, self.nx - 1):

                    coeff = (Tx[i, j] + Tx[i, j+1]) / self.dx**2 + (Ty[i, j] + Ty[i+1, j]) / self.dy**2
                    if coeff < 1e-20:
                        continue

                    rhs = (Tx[i, j] * h[i, j-1] + Tx[i, j+1] * h[i, j+1]) / self.dx**2
                    rhs += (Ty[i, j] * h[i-1, j] + Ty[i+1, j] * h[i+1, j]) / self.dy**2
                    rhs -= self.source[i, j]

                    h_new = rhs / coeff
                    h[i, j] = (1.0 - omega) * h[i, j] + omega * h_new


            diff = np.max(np.abs(h - h_old))
            if diff < tol:
                break

        self.head = h
        return h

    def solve_conjugate_gradient(self, T: np.ndarray,
                                  h_boundary: dict = None,
                                  max_iter: int = None,
                                  tol: float = 1.0e-10) -> np.ndarray:
        if T.shape != (self.ny, self.nx):
            raise ValueError("T 的形状必须与网格匹配")

        if h_boundary is None:
            h_boundary = {'left': 10.0, 'right': 0.0, 'top': 5.0, 'bottom': 5.0}

        if max_iter is None:
            max_iter = self.nx * self.ny

        n = self.nx * self.ny


        def matvec(h_flat: np.ndarray) -> np.ndarray:
            h = h_flat.reshape((self.ny, self.nx))
            Ah = np.zeros_like(h)

            Tx = np.zeros((self.ny, self.nx + 1))
            Ty = np.zeros((self.ny + 1, self.nx))
            Tx[:, 1:-1] = 2.0 * T[:, :-1] * T[:, 1:] / (T[:, :-1] + T[:, 1:] + 1e-20)
            Ty[1:-1, :] = 2.0 * T[:-1, :] * T[1:, :] / (T[:-1, :] + T[1:, :] + 1e-20)

            for i in range(1, self.ny - 1):
                for j in range(1, self.nx - 1):
                    Ah[i, j] = (
                        -(Tx[i, j] * h[i, j-1] + Tx[i, j+1] * h[i, j+1]) / self.dx**2
                        -(Ty[i, j] * h[i-1, j] + Ty[i+1, j] * h[i+1, j]) / self.dy**2
                        + ((Tx[i, j] + Tx[i, j+1]) / self.dx**2
                           + (Ty[i, j] + Ty[i+1, j]) / self.dy**2) * h[i, j]
                    )

            Ah[:, 0] = h[:, 0]
            Ah[:, -1] = h[:, -1]
            Ah[0, :] = h[0, :]
            Ah[-1, :] = h[-1, :]
            return Ah.ravel()


        x = np.zeros(n)
        x = x.reshape((self.ny, self.nx))
        x[:, 0] = h_boundary.get('left', 0.0)
        x[:, -1] = h_boundary.get('right', 0.0)
        x[0, :] = h_boundary.get('top', 0.0)
        x[-1, :] = h_boundary.get('bottom', 0.0)
        x = x.ravel()


        b = np.zeros(n)
        b = b.reshape((self.ny, self.nx))
        b[:, 0] = h_boundary.get('left', 0.0)
        b[:, -1] = h_boundary.get('right', 0.0)
        b[0, :] = h_boundary.get('top', 0.0)
        b[-1, :] = h_boundary.get('bottom', 0.0)
        for i in range(1, self.ny - 1):
            for j in range(1, self.nx - 1):
                b[i, j] = -self.source[i, j]
        b = b.ravel()


        Ax = matvec(x)
        r = b - Ax
        p = r.copy()
        rs_old = np.dot(r, r)

        for it in range(max_iter):
            Ap = matvec(p)
            pAp = np.dot(p, Ap)
            if abs(pAp) < 1e-30:
                break

            alpha = rs_old / pAp
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = np.dot(r, r)

            if np.sqrt(rs_new) < tol:
                break

            beta = rs_new / rs_old
            p = r + beta * p
            rs_old = rs_new

        self.head = x.reshape((self.ny, self.nx))
        return self.head

    def compute_velocity(self, T: np.ndarray,
                         porosity: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:

        pass

    def compute_flow_rate(self, T: np.ndarray) -> dict:
        if self.head is None or np.all(self.head == 0):
            raise RuntimeError("先求解水头场")

        h = self.head


        Q_left = np.sum(T[:, 0] * (h[:, 1] - h[:, 0]) / self.dx * self.dy)

        Q_right = np.sum(T[:, -1] * (h[:, -2] - h[:, -1]) / self.dx * self.dy)

        Q_top = np.sum(T[0, :] * (h[1, :] - h[0, :]) / self.dy * self.dx)

        Q_bottom = np.sum(T[-1, :] * (h[-2, :] - h[-1, :]) / self.dy * self.dx)

        return {
            'Q_left': float(Q_left),
            'Q_right': float(Q_right),
            'Q_top': float(Q_top),
            'Q_bottom': float(Q_bottom),
            'Q_net': float(Q_left + Q_right + Q_top + Q_bottom)
        }
