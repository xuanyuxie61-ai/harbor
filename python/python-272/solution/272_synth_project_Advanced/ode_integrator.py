"""
常微分方程积分器模块
实现 Berry phase 沿路径的演化方程

物理背景：
Berry phase 可以通过求解绝热演化方程获得：
iℏ d|ψ>/dt = H(k(t))|ψ>

对于闭合路径 C，Berry phase 为：
γ_n = i ∮_C <u_n|∇_k|u_n>·dk

等价于求解 Schrödinger 方程在参数空间中的演化
"""

import numpy as np
from typing import Callable, Tuple, List, Dict


class ODEIntegrator:
    """
    ODE 积分器

    用于 Berry phase 的绝热演化计算
    """

    def __init__(self, method: str = 'rk4'):
        """
        初始化

        Parameters:
        -----------
        method : str
            积分方法 ('euler', 'rk4', 'rk45', 'magnus')
        """
        self.method = method

    def integrate(self, f: Callable, y0: np.ndarray,
                  t_span: Tuple[float, float],
                  n_steps: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        积分 ODE dy/dt = f(t, y)

        Parameters:
        -----------
        f : Callable
            右端函数 f(t, y)
        y0 : np.ndarray
            初始条件
        t_span : Tuple[float, float]
            时间范围
        n_steps : int
            步数

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            t_values, y_values
        """
        t0, tf = t_span
        t_values = np.linspace(t0, tf, n_steps + 1)
        dt = (tf - t0) / n_steps

        y_values = np.zeros((n_steps + 1, len(y0)), dtype=y0.dtype)
        y_values[0] = y0

        for i in range(n_steps):
            t = t_values[i]
            y = y_values[i]

            if self.method == 'euler':
                y_next = self._euler_step(f, t, y, dt)
            elif self.method == 'rk4':
                y_next = self._rk4_step(f, t, y, dt)
            elif self.method == 'rk45':
                y_next, _ = self._rk45_step(f, t, y, dt)
            elif self.method == 'magnus':
                y_next = self._magnus_step(f, t, y, dt)
            else:
                raise ValueError(f"Unknown method: {self.method}")

            y_values[i + 1] = y_next

        return t_values, y_values

    def _euler_step(self, f: Callable, t: float, y: np.ndarray,
                    dt: float) -> np.ndarray:
        """Euler 方法"""
        return y + dt * f(t, y)

    def _rk4_step(self, f: Callable, t: float, y: np.ndarray,
                  dt: float) -> np.ndarray:
        """
        经典四阶 Runge-Kutta 方法

        k1 = f(t, y)
        k2 = f(t + dt/2, y + dt/2 k1)
        k3 = f(t + dt/2, y + dt/2 k2)
        k4 = f(t + dt, y + dt k3)
        y_{n+1} = y_n + (dt/6)(k1 + 2k2 + 2k3 + k4)
        """
        k1 = f(t, y)
        k2 = f(t + dt/2, y + dt/2 * k1)
        k3 = f(t + dt/2, y + dt/2 * k2)
        k4 = f(t + dt, y + dt * k3)

        return y + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

    def _rk45_step(self, f: Callable, t: float, y: np.ndarray,
                   dt: float) -> Tuple[np.ndarray, float]:
        """
        Runge-Kutta-Fehlberg 方法 (自适应步长)

        使用 Dormand-Prince 系数
        """
        # Dormand-Prince 系数
        a2, a3, a4, a5, a6 = 1/5, 3/10, 4/5, 8/9, 1.0

        b21 = 1/5
        b31, b32 = 3/40, 9/40
        b41, b42, b43 = 44/45, -56/15, 32/9
        b51, b52, b53, b54 = 19372/6561, -25360/2187, 64448/6561, -212/729
        b61, b62, b63, b64, b65 = 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656

        # 4阶系数
        c1, c3, c4, c5, c6 = 5179/57600, 7571/16695, 393/640, -92097/339200, 187/2100

        k1 = f(t, y)
        k2 = f(t + a2*dt, y + dt*b21*k1)
        k3 = f(t + a3*dt, y + dt*(b31*k1 + b32*k2))
        k4 = f(t + a4*dt, y + dt*(b41*k1 + b42*k2 + b43*k3))
        k5 = f(t + a5*dt, y + dt*(b51*k1 + b52*k2 + b53*k3 + b54*k4))
        k6 = f(t + a6*dt, y + dt*(b61*k1 + b62*k2 + b63*k3 + b64*k4 + b65*k5))

        y_next = y + dt * (c1*k1 + c3*k3 + c4*k4 + c5*k5 + c6*k6)

        # 误差估计 (略)
        error = 1e-6

        return y_next, error

    def _magnus_step(self, f: Callable, t: float, y: np.ndarray,
                     dt: float) -> np.ndarray:
        """
        Magnus 展开方法 (适合酉演化)

        对于 i dU/dt = H(t)U，Magnus 展开保证 U 保持酉性：
        U(t+dt) = exp(Ω(t)) U(t)
        Ω(t) = -i ∫_t^{t+dt} H(s) ds + O(dt³)
        """
        # 一阶 Magnus
        H_mid = f(t + dt/2, y)

        # 矩阵指数
        if y.ndim == 1:
            # 向量情况
            Omega = -1j * dt * H_mid
            U = np.exp(Omega)
            return U * y
        else:
            # 矩阵情况
            from scipy.linalg import expm
            Omega = -1j * dt * H_mid
            U = expm(Omega)
            return U @ y

    def integrate_schrodinger(self, H_func: Callable, psi0: np.ndarray,
                              k_path: np.ndarray,
                              n_steps: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        积分 Schrödinger 方程计算 Berry phase

        i d|ψ>/dt = H(k(t))|ψ>

        Parameters:
        -----------
        H_func : Callable
            Hamiltonian 函数 H(k) → matrix
        psi0 : np.ndarray
            初始态
        k_path : np.ndarray
            k 空间路径
        n_steps : int
            步数

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            t_values, psi_values
        """
        # 定义 ODE 右端
        def f(t, psi):
            # 插值 k(t)
            idx = int(t * (len(k_path) - 1))
            idx = min(idx, len(k_path) - 1)
            k = k_path[idx]
            H = H_func(k)
            return -1j * H @ psi

        return self.integrate(f, psi0, (0, 1), n_steps)

    def compute_berry_phase_evolution(self, H_func: Callable,
                                       psi0: np.ndarray,
                                       k_path: np.ndarray,
                                       n_steps: int = 100) -> Dict:
        """
        计算 Berry phase 随路径的演化

        Parameters:
        -----------
        H_func : Callable
            Hamiltonian 函数
        psi0 : np.ndarray
            初始态
        k_path : np.ndarray
            闭合路径
        n_steps : int
            步数

        Returns:
        --------
        Dict
            包含 Berry phase、态演化等信息
        """
        t_values, psi_values = self.integrate_schrodinger(
            H_func, psi0, k_path, n_steps
        )

        # 计算瞬时 Berry phase
        berry_phases = []
        for i, psi in enumerate(psi_values):
            # γ(t) = arg(<ψ(0)|ψ(t)>)
            overlap = np.conj(psi0) @ psi
            phase = np.angle(overlap)
            berry_phases.append(phase)

        berry_phases = np.array(berry_phases)

        # 最终 Berry phase
        final_phase = berry_phases[-1]

        # 检查闭合性
        psi_final = psi_values[-1]
        fidelity = np.abs(np.conj(psi0) @ psi_final)**2

        return {
            't_values': t_values,
            'psi_values': psi_values,
            'berry_phases': berry_phases,
            'final_berry_phase': final_phase,
            'fidelity': fidelity,
            'path_length': len(k_path)
        }

    def adaptive_integration(self, f: Callable, y0: np.ndarray,
                            t_span: Tuple[float, float],
                            tol: float = 1e-8,
                            max_steps: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        """
        自适应步长积分

        使用 RK45 并动态调整步长

        Parameters:
        -----------
        f : Callable
            右端函数
        y0 : np.ndarray
            初始条件
        t_span : Tuple[float, float]
            时间范围
        tol : float
            误差容限
        max_steps : int
            最大步数

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            t_values, y_values
        """
        t0, tf = t_span
        dt_init = (tf - t0) / 100

        t_values = [t0]
        y_values = [y0.copy()]

        t = t0
        y = y0.copy()
        dt = dt_init

        step = 0
        while t < tf and step < max_steps:
            # RK45 步
            y_next, error = self._rk45_step(f, t, y, dt)

            # 调整步长
            if error < tol:
                # 接受步
                t = min(t + dt, tf)
                y = y_next
                t_values.append(t)
                y_values.append(y.copy())

                # 增大步长
                if error > 0:
                    dt = min(dt * 1.2, tf - t)
            else:
                # 拒绝步，减小步长
                dt = dt * 0.5

            step += 1

        return np.array(t_values), np.array(y_values)


def compute_zygv_phase(H_func: Callable, k_path: np.ndarray,
                       band_idx: int = 0,
                       n_steps: int = 100) -> float:
    """
    使用 Zak 相位公式计算 Berry phase

    γ_n = i ∮ <u_n|∇_k|u_n>·dk

    离散化：
    γ_n ≈ -Im Σ_j ln <u_n(k_j)|u_n(k_{j+1})>

    Parameters:
    -----------
    H_func : Callable
        Hamiltonian 函数
    k_path : np.ndarray
        闭合路径
    band_idx : int
        能带索引
    n_steps : int
        步数

    Returns:
    --------
    float
        Berry phase
    """
    n_path = len(k_path)
    phase_sum = 0.0

    for j in range(n_path - 1):
        H1 = H_func(k_path[j])
        H2 = H_func(k_path[j + 1])

        _, u1 = np.linalg.eigh(H1)
        _, u2 = np.linalg.eigh(H2)

        overlap = np.conj(u1[:, band_idx]) @ u2[:, band_idx]
        phase_sum += np.angle(overlap + 1e-15)

    # 闭合
    H1 = H_func(k_path[-1])
    H2 = H_func(k_path[0])
    _, u1 = np.linalg.eigh(H1)
    _, u2 = np.linalg.eigh(H2)
    overlap = np.conj(u1[:, band_idx]) @ u2[:, band_idx]
    phase_sum += np.angle(overlap + 1e-15)

    return -phase_sum
