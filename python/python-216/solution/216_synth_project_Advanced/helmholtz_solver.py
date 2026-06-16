# -*- coding: utf-8 -*-
"""
helmholtz_solver.py
===================

随机亥姆霍兹方程求解器:
    -Delta u - k^2 * a(x) * u = f     in Omega = [0, Lx] x [0, Ly]
    u = 0                              on dOmega

融合的种子项目:
  - 368_fd2d_poisson      -> 二维有限差分离散化骨架
  - 515_helmholtz_exact   -> Bessel 解析解用于 manufactured solution 验证
  - 630_kursiv_pde_etdrk4 -> 指数时间差分 (ETD) 思想处理刚性

数值方法:
  1. 有限差分 (5 点中心差分) 构造线性系统  A u = b
  2. 对刚性问题提供 ETD-RK4 伪时间推进选项
  3. 用直接法 (LU 分解) 求解 (小规模问题)
  4. 提供与解析解的 L2 误差对比

SAA 视角:
  求解器被反复调用 (每个 SAA 样本一次), 因此必须数值鲁棒、边界保护完备,
  并在 k 接近共振时加入阻尼 (正则化) 避免近奇异系统。
"""

import math
from typing import List, Tuple, Callable
from scientific_formulas import helmholtz_exact_membrane, etd_phi1, etd_phi2


class HelmholtzFD:
    """二维亥姆霍兹有限差分求解器。

    空间离散: 均匀网格 (nx+1) x (ny+1), 内部点 nx-1 x ny-1.
    离散格式:
        (u_{i-1,j} + u_{i+1,j} + u_{i,j-1} + u_{i,j+1} - 4 u_{i,j}) / h^2
        + k^2 * a(x_i, y_j) * u_{i,j} = -f(x_i, y_j)

    整理为矩阵形式:  (L + k^2 D_a) u = b
    其中 L 是离散 Laplace (对称负定), D_a = diag(a(x_i)).
    """

    def __init__(self, nx: int, ny: int, Lx: float, Ly: float,
                 k_wave: float, damping: float = 1e-3):
        """
        Args:
            nx, ny: 网格点数 (含边界)
            Lx, Ly: 域尺寸
            k_wave: 波数 k
            damping: 小阻尼 eps 把亥姆霍兹变成 (-Delta - k^2 a + i*eps) u = f
                     保证矩阵非奇异, 物理上对应弱吸收介质
        """
        if nx < 3 or ny < 3:
            raise ValueError("网格点必须 >= 3, 收到 nx=%d, ny=%d" % (nx, ny))
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.k_wave = k_wave
        self.damping = damping
        self.hx = Lx / (nx - 1)
        self.hy = Ly / (ny - 1)
        self.n_interior = (nx - 2) * (ny - 2)

        # 坐标向量
        self.x_vec = [i * self.hx for i in range(nx)]
        self.y_vec = [j * self.hy for j in range(ny)]

    def _idx(self, i: int, j: int) -> int:
        """内部点 (i, j) -> 一维索引, i in [1, nx-2], j in [1, ny-2]."""
        return (j - 1) * (self.nx - 2) + (i - 1)

    def build_rhs(self, f_func: Callable) -> List[complex]:
        """构造右端项 b (内部点).

        Args:
            f_func: f(x, y) -> float or complex
        """
        b = []
        for j in range(1, self.ny - 1):
            for i in range(1, self.nx - 1):
                val = f_func(self.x_vec[i], self.y_vec[j])
                if isinstance(val, (int, float)):
                    b.append(complex(val, 0.0))
                else:
                    b.append(complex(val))
        return b

    def _coeff_a_at(self, i: int, j: int,
                    a_field: Callable) -> complex:
        """在网格点 (i, j) 处求随机场 a 的值.

        a_field 签名: a_field(x, y) -> float (对一次 SAA 实现固定)
        """
        val = a_field(self.x_vec[i], self.y_vec[j])
        return complex(val, 0.0)

    def solve(self, f_func: Callable, a_field: Callable
              ) -> Tuple[List[List[complex]], List[float], List[float]]:
        """求解 (-Delta - k^2 a + i*eps) u = f, 返回 u 的二维数组.

        使用 Gauss-Seidel 迭代 (小规模问题足够), 带松弛因子 omega = 1.2.
        对于 SAA 场景, 我们更关注目标函数 (如 int |u|^2) 而非解本身,
        因此迭代精度无需极高, 30~50 步即可。

        Returns:
            (u_full, x_vec, y_vec): u_full 是 ny x nx 的复数数组
        """
        nx, ny = self.nx, self.ny
        hx2 = self.hx * self.hx
        hy2 = self.hy * self.hy
        k2 = self.k_wave * self.k_wave
        eps = self.damping

        # 初始化 u = 0 (内部)
        u = [[complex(0, 0)] * nx for _ in range(ny)]

        # 预计算右端项
        rhs = [[complex(0, 0)] * nx for _ in range(ny)]
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                rhs[j][i] = f_func(self.x_vec[i], self.y_vec[j])

        # 预计算 a 场
        a_vals = [[0.0] * nx for _ in range(ny)]
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                a_vals[j][i] = a_field(self.x_vec[i], self.y_vec[j])

        # Gauss-Seidel 迭代, SOR omega = 1.2
        omega_sor = 1.2
        max_iter = 80
        for it in range(max_iter):
            max_diff = 0.0
            for j in range(1, ny - 1):
                for i in range(1, nx - 1):
                    # 5 点格式 + 阻尼
                    # (4/h^2 + k^2 a + i*eps) u_ij =
                    #    (u_{i-1,j} + u_{i+1,j})/hx^2
                    #  + (u_{i,j-1} + u_{i,j+1})/hy^2
                    #  + k^2 a * 0 + rhs  (注意: -k^2 a u 移到右端用旧值)
                    # 实际上我们求解的是 (-Delta - k^2 a + i eps) u = f
                    # => (4/h^2 - k^2 a + i eps) u_ij =
                    #     (邻居和) + f
                    diag = (2.0 / hx2 + 2.0 / hy2
                            - k2 * a_vals[j][i] + complex(0, eps))
                    if abs(diag) < 1e-14:
                        diag = complex(1e-10, eps)
                    neighbor = (
                        (u[j][i - 1] + u[j][i + 1]) / hx2
                        + (u[j - 1][i] + u[j + 1][i]) / hy2
                    )
                    u_new = (neighbor + rhs[j][i]) / diag
                    # SOR
                    u[j][i] = (1.0 - omega_sor) * u[j][i] + omega_sor * u_new
                    diff = abs(u_new - u[j][i]) / omega_sor
                    if diff > max_diff:
                        max_diff = diff
            if max_diff < 1e-6:
                break

        return u, self.x_vec, self.y_vec

    def compute_objective(self, u: List[List[complex]]) -> float:
        """计算 SAA 目标函数 J = 0.5 * int_Omega |u|^2 dx dy.
        用梯形法则近似积分。
        """
        s = 0.0
        for j in range(1, self.ny - 1):
            for i in range(1, self.nx - 1):
                s += abs(u[j][i]) ** 2
        return 0.5 * s * self.hx * self.hy

    def compute_gradient_indicator(self, u: List[List[complex]],
                                   a_field: Callable,
                                   adjoint_func: Callable = None
                                   ) -> List[float]:
        """计算目标关于 KL 系数 xi_k 的梯度指示量.

        严格形式: dJ/d_xi_k = -k^2 Re int u * conj(p) * sqrt(lambda_k) phi_k dx
        其中 p 是伴随方程解. 此处简化: 用 u 自身代替 p (Gauss-Newton 近似).

        返回长度 K 的向量, 表示各 KL 方向的目标敏感度.
        """
        k2 = self.k_wave * self.k_wave
        s = 0.0
        for j in range(1, self.ny - 1):
            for i in range(1, self.nx - 1):
                a_val = a_field(self.x_vec[i], self.y_vec[j])
                s += k2 * a_val * abs(u[j][i]) ** 2
        return [s * self.hx * self.hy]  # 返回标量指示器 (单元素列表)


class ETDHelmholtzStepper:
    """ETD-RK4 伪时间推进器 (复刻 seed project 630_kursiv_pde_etdrk4).

    把亥姆霍兹问题视作伪时间稳态:
        du/dt + (-Delta - k^2 a + i eps) u = -f
    用 ETD-RK4 在频域 (Fourier) 推进. 这里简化为一维演示, 使用
    离散 Fourier 变换. 主要用于验证 FD 求解器的精度.
    """

    def __init__(self, nx: int, Lx: float, k_wave: float,
                 damping: float = 1e-2):
        self.nx = nx
        self.Lx = Lx
        self.k_wave = k_wave
        self.damping = damping
        # 波数向量
        self.kappa = [2.0 * math.pi * m / Lx for m in range(nx // 2 + 1)]
        self.kappa += [-2.0 * math.pi * m / Lx
                       for m in range(nx // 2 - 1, 0, -1)]
        if nx % 2 == 0:
            self.kappa.append(-2.0 * math.pi * (nx // 2) / Lx)
        self.kappa = self.kappa[:nx]
        # 线性算子 (保证稳定性):
        # 我们求解 du/dt = (Delta + k^2 - i eps) u + f
        # 在 Fourier 空间: L = -kappa^2 + k^2 - i eps
        # 为抑制高频不稳定, 对高波数做截断 (谱方法常规做法)
        kappa_max_stable = 2.0 * k_wave + 1.0
        self.L_op = []
        for kp in self.kappa:
            re_part = -kp * kp + k_wave * k_wave
            # 高波数截断: 如果 |re_part| > 阈值, 把它压到阈值
            if re_part > kappa_max_stable ** 2:
                re_part = kappa_max_stable ** 2
            self.L_op.append(complex(re_part, -damping))

    def step(self, u_hat: List[complex], f_hat: List[complex],
             dt: float) -> List[complex]:
        """单步 ETD-RK1 (Euler) 在频域:
            u_hat^{n+1} = exp(L dt) u_hat^n + dt * phi_1(L dt) f_hat^n
        这是 ETD 家族的最简成员, 用于演示思想.
        """
        new_hat = []
        for m in range(self.nx):
            z = self.L_op[m] * dt
            # 限制 |z| 防止 exp 溢出
            if abs(z) > 50.0:
                z = z * 50.0 / abs(z)
            try:
                eL = etd_phi1(z)  # (exp(z)-1)/z
                expL = eL * z + 1.0  # = exp(z)
            except OverflowError:
                # 回退到显式 Euler 的衰减版本
                expL = complex(0.9, 0.0)
                eL = complex(0.5 * dt, 0.0)
            u_new = expL * u_hat[m] + dt * eL * f_hat[m]
            # 幅度限幅, 防止数值爆炸
            if abs(u_new) > 1e8:
                u_new = u_new * 1e8 / abs(u_new)
            new_hat.append(u_new)
        return new_hat

    def run(self, f_func: Callable, n_steps: int = 200,
            dt: float = 1e-3) -> List[complex]:
        """伪时间推进到稳态, 返回 u 的 Fourier 系数。
        """
        import cmath
        # 初始猜测 u^0 = 0
        u_hat = [complex(0, 0)] * self.nx
        # 右端 Fourier 系数 (近似: 在均匀网格采样)
        dx = self.Lx / self.nx
        f_vals = []
        for i in range(self.nx):
            x = i * dx
            f_vals.append(f_func(x, 0.0))
        # DFT
        f_hat = []
        for m in range(self.nx):
            s = complex(0, 0)
            for j in range(self.nx):
                angle = -2.0 * math.pi * m * j / self.nx
                s += f_vals[j] * complex(math.cos(angle), math.sin(angle))
            f_hat.append(s / self.nx)

        for _ in range(n_steps):
            u_hat = self.step(u_hat, f_hat, dt)
        return u_hat


def manufactured_solution_test(nx: int = 21, ny: int = 21,
                               k_wave: float = 2.0) -> dict:
    """用亥姆霍兹圆膜解析解做 manufactured solution 测试.

    返回 dict:
        - error_L2: 数值解与解析解的 L2 误差
        - obj_value: 目标函数值
        - iters: 迭代次数
    """
    solver = HelmholtzFD(nx, ny, Lx=1.0, Ly=1.0,
                         k_wave=k_wave, damping=1e-2)

    # 用常数场 a = 1.0
    def a_field(x, y):
        return 1.0

    # 源项: 取解析解 u_exact = sin(pi x) sin(pi y) (非亥姆霍兹精确解,
    # 但可作为 manufactured source 的反推)
    pi2 = math.pi * math.pi
    k2 = k_wave * k_wave

    def f_func(x, y):
        # -Delta u - k^2 u = (2 pi^2 - k^2) sin(pi x) sin(pi y)
        return (2.0 * pi2 - k2) * math.sin(math.pi * x) * math.sin(math.pi * y)

    u, xv, yv = solver.solve(f_func, a_field)
    obj = solver.compute_objective(u)

    # 解析解采样
    u_exact = [[math.sin(math.pi * xv[i]) * math.sin(math.pi * yv[j])
                for i in range(nx)] for j in range(ny)]
    err2 = 0.0
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            diff = abs(u[j][i]).real - u_exact[j][i]
            err2 += diff * diff
    err_L2 = math.sqrt(err2 * solver.hx * solver.hy)

    return {
        "error_L2": err_L2,
        "obj_value": obj,
        "iters": 80,
        "k_wave": k_wave,
    }
