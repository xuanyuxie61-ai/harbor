
import numpy as np
from typing import Tuple, Optional





class ManipulatorKinematics:

    def __init__(self):


        self.n_dof = 7
        self.mdh = np.array([
            [0.0,       0.0,            0.333,      0.0      ],
            [0.0,      -np.pi/2,        0.0,        0.0      ],
            [0.0,       np.pi/2,        0.316,      0.0      ],
            [0.0825,    np.pi/2,        0.0,        0.0      ],
            [-0.0825,  -np.pi/2,        0.384,      0.0      ],
            [0.0,       np.pi/2,        0.0,        0.0      ],
            [0.088,     np.pi/2,        0.107,      0.0      ],
        ], dtype=float)

        self.link_mass = np.array([2.5, 2.5, 2.0, 2.0, 1.5, 1.5, 1.2], dtype=float)
        self.link_inertia = [
            np.diag([0.01, 0.01, 0.01]),
            np.diag([0.01, 0.01, 0.01]),
            np.diag([0.008, 0.008, 0.008]),
            np.diag([0.008, 0.008, 0.008]),
            np.diag([0.005, 0.005, 0.005]),
            np.diag([0.005, 0.005, 0.005]),
            np.diag([0.003, 0.003, 0.003]),
        ]

    def _mdh_transform(self, a: float, alpha: float, d: float, theta: float) -> np.ndarray:
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        T = np.array([
            [ct,     -st,     0.0,   a],
            [st*ca,  ct*ca,  -sa,  -d*sa],
            [st*sa,  ct*sa,   ca,   d*ca],
            [0.0,    0.0,    0.0,   1.0]
        ], dtype=float)
        return T

    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.size != self.n_dof:
            raise ValueError(f"关节角维度必须为 {self.n_dof}, 得到 {q.size}")

        q = np.clip(q, -np.pi, np.pi)

        T = np.eye(4, dtype=float)
        self._T_list = [T.copy()]
        for i in range(self.n_dof):
            a, alpha, d, theta0 = self.mdh[i]
            theta = theta0 + q[i]
            T_i = self._mdh_transform(a, alpha, d, theta)
            T = T @ T_i
            self._T_list.append(T.copy())
        return T

    def geometric_jacobian(self, q: np.ndarray) -> np.ndarray:
        T_ee = self.forward_kinematics(q)
        p_ee = T_ee[:3, 3]
        J = np.zeros((6, self.n_dof), dtype=float)
        z0 = np.array([0.0, 0.0, 1.0])
        for i in range(self.n_dof):
            T_i = self._T_list[i]
            p_i = T_i[:3, 3]
            z_i = T_i[:3, 2]
            J[:3, i] = np.cross(z_i, p_ee - p_i)
            J[3:, i] = z_i
        return J

    def manipulability_measure(self, q: np.ndarray) -> float:
        J = self.geometric_jacobian(q)
        Jv = J[:3, :]
        try:
            m = np.sqrt(np.linalg.det(Jv @ Jv.T))
        except np.linalg.LinAlgError:
            m = 0.0
        if not np.isfinite(m):
            m = 0.0
        return float(m)






class StiffODEIntegrator:

    def __init__(self, tol: float = 1e-6, max_iter: int = 20):
        self.tol = tol
        self.max_iter = max_iter
        self.gamma = 1.0 - 1.0 / np.sqrt(2.0)

    def _newton_solve(self, f, t, y, h, gamma, k_guess):
        k = k_guess.copy()
        for _ in range(self.max_iter):
            y_stage = y + h * gamma * k
            f_val = f(t, y_stage)

            eps = np.sqrt(np.finfo(float).eps) * (np.linalg.norm(y_stage) + 1.0)
            if eps < 1e-14:
                eps = 1e-14


            residual = k - f_val
            if np.linalg.norm(residual) < self.tol:
                break

            damp = 1.0 / (1.0 + h * gamma * 0.1)
            k = k - damp * residual

            k = np.clip(k, -1e6, 1e6)
        return k

    def integrate(self, f, t_span: Tuple[float, float], y0: np.ndarray,
                  h0: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        t0, tf = t_span
        y0 = np.asarray(y0, dtype=float).reshape(-1)
        if h0 is None:
            h0 = (tf - t0) / 100.0
        h = max(min(h0, tf - t0), 1e-12)
        t = t0
        y = y0.copy()
        t_out = [t]
        y_out = [y.copy()]
        gamma = self.gamma

        while t < tf - 1e-14:
            if t + h > tf:
                h = tf - t

            k1 = self._newton_solve(f, t, y, h, gamma, f(t, y))

            y_stage2 = y + h * (1.0 - gamma) * k1
            k2 = self._newton_solve(f, t + h, y_stage2, h, gamma, f(t + h, y_stage2))

            y_new = y + h * ((1.0 - gamma) * k1 + gamma * k2)

            y_embed = y + h * 0.5 * (k1 + k2)
            err_est = np.linalg.norm(y_new - y_embed) + 1e-20

            fac = 0.9 * (self.tol / err_est) ** 0.5
            fac = max(0.2, min(5.0, fac))
            if err_est > self.tol and h > 1e-12:
                h *= fac
                continue
            t += h
            y = y_new
            t_out.append(t)
            y_out.append(y.copy())
            h *= fac
            h = max(min(h, tf - t), 1e-12)
        return np.array(t_out), np.array(y_out)






def cgne_solve(A: np.ndarray, b: np.ndarray, x0: Optional[np.ndarray] = None,
               max_iter: int = 500, tol: float = 1e-10) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    m, n = A.shape
    if b.size != m:
        raise ValueError(f"b维度{b.size}与A行数{m}不匹配")
    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.asarray(x0, dtype=float).reshape(-1).copy()
        if x.size != n:
            raise ValueError(f"x0维度{x.size}与A列数{n}不匹配")

    r = b - A @ x
    z = A.T @ r
    d = z.copy()
    ztz_old = float(z @ z)
    if ztz_old < tol * tol:
        return x

    for k in range(max_iter):
        Ad = A @ d
        denom = float(Ad @ Ad)
        if abs(denom) < np.finfo(float).eps:
            break
        alpha = ztz_old / denom
        x = x + alpha * d
        r = r - alpha * Ad
        z = A.T @ r
        ztz_new = float(z @ z)
        if np.sqrt(ztz_new) < tol:
            break
        beta = ztz_new / ztz_old
        d = z + beta * d
        ztz_old = ztz_new

        if not np.isfinite(x).all():
            x = np.zeros(n, dtype=float)
            break
    return x


def differential_ik_solver(kin: ManipulatorKinematics, q: np.ndarray,
                           dx_des: np.ndarray, damping: float = 0.01) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(-1)
    dx_des = np.asarray(dx_des, dtype=float).reshape(-1)
    J = kin.geometric_jacobian(q)
    m, n = J.shape

    A_aug = np.vstack([J, damping * np.eye(n)])
    b_aug = np.concatenate([dx_des, np.zeros(n)])
    dq = cgne_solve(A_aug, b_aug, max_iter=300, tol=1e-8)

    max_dq = 2.0
    norm_dq = np.linalg.norm(dq)
    if norm_dq > max_dq:
        dq = dq * (max_dq / norm_dq)
    return dq






def manipulator_dynamics_ode(kin: ManipulatorKinematics):
    n = kin.n_dof

    M_diag = np.array([2.5, 2.5, 2.0, 1.8, 1.2, 1.0, 0.8], dtype=float)

    def f(t: float, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float).reshape(-1)
        if y.size != 2 * n:
            raise ValueError(f"状态向量维度应为{2*n}")
        q = y[:n]
        dq = y[n:]

        q = np.clip(q, -np.pi, np.pi)
        dq = np.clip(dq, -5.0, 5.0)

        g_vec = np.array([0.0, -5.0*np.sin(q[1]), -3.0*np.sin(q[2]),
                          -2.0*np.sin(q[3]), -1.5*np.sin(q[4]),
                          -1.0*np.sin(q[5]), -0.8*np.sin(q[6])], dtype=float)

        Cdq = 0.1 * dq * np.abs(dq)

        tau = -2.0 * dq - 0.5 * q
        ddq = (tau - Cdq - g_vec) / M_diag
        ddq = np.clip(ddq, -20.0, 20.0)
        return np.concatenate([dq, ddq])
    return f
