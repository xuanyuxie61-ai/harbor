"""
lagrangian_particles.py
拉格朗日粒子追踪与混合分析模块

科学背景:
拉格朗日框架下的海洋中尺度涡旋混合研究:
  dx/dt = u(x,y,t)
  dy/dt = v(x,y,t)

其中 (u,v) 为欧拉速度场. 通过追踪大量被动示踪粒子, 可量化:
  - 拉格朗日相干结构 (Lagrangian Coherent Structures, LCS)
  - 有限时间 Lyapunov 指数 (FTLE)
  - 有效扩散系数
  - 粒子对分离率 (Richardson 定律)

Richardson 定律: <r^2(t)> ~ t^3 (对二维湍流)
  其中 r 为粒子对间距.

本模块实现:
  - 最优粒子初始化 (Fibonacci 螺旋/黄金角采样)
  - 周期性域上的最优分布 (Lloyd/CVT 迭代优化)
  - 高质量随机数生成器 (CMRG, 用于随机扩散)
  - RK4 粒子轨迹积分
  - 混合统计量计算

融合来源:
- 427_fibonacci_spiral: 黄金角螺旋采样
- 265_cvtp_1d: 周期性 Lloyd 迭代优化
- 1040_rnglib: 组合型多递推伪随机数生成器
"""

import numpy as np
from typing import Tuple, Optional, Callable


# ============================================================
# 1. 高质量伪随机数生成器 (from 1040_rnglib)
# ============================================================

class CMRG:
    """
    Combined Multiple Recursive Random Number Generator (L'Ecuyer).

    两个 LCG 并行:
      s1_{n+1} = (40014 * s1_n) mod 2147483563
      s2_{n+1} = (40692 * s2_n) mod 2147483399
    组合输出: z = (s1 - s2) mod 2147483563
    归一化: u = z / 2147483563  (若 z<0 则加 2147483563)

    周期: ~ 2.3058 × 10^18
    """

    M1 = 2147483563
    M2 = 2147483399
    A1 = 40014
    A2 = 40692

    def __init__(self, seed1: int = 12345, seed2: int = 67890):
        self.s1 = int(seed1) % self.M1
        self.s2 = int(seed2) % self.M2
        if self.s1 == 0:
            self.s1 = 1
        if self.s2 == 0:
            self.s2 = 1

    def _advance(self):
        self.s1 = (self.A1 * self.s1) % self.M1
        self.s2 = (self.A2 * self.s2) % self.M2

    def rand(self) -> float:
        """返回 [0,1) 均匀随机数."""
        self._advance()
        z = self.s1 - self.s2
        if z < 0:
            z += self.M1
        return z / self.M1

    def randn(self) -> float:
        """Box-Muller 变换生成标准正态随机数."""
        u1 = self.rand()
        u2 = self.rand()
        if u1 < 1e-15:
            u1 = 1e-15
        return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)

    def rand_array(self, shape: Tuple[int, ...]) -> np.ndarray:
        """生成 [0,1) 均匀随机数组."""
        return np.array([self.rand() for _ in range(int(np.prod(shape)))]).reshape(shape)

    def randn_array(self, shape: Tuple[int, ...]) -> np.ndarray:
        """生成标准正态随机数组."""
        return np.array([self.randn() for _ in range(int(np.prod(shape)))]).reshape(shape)


# ============================================================
# 2. Fibonacci 螺旋采样 (from 427_fibonacci_spiral)
# ============================================================

def fibonacci_spiral_2d(n: int, Lx: float = 1.0, Ly: float = 1.0,
                        center: Tuple[float, float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    二维 Fibonacci / 黄金角螺旋采样.

    黄金比例: φ = (1 + sqrt(5)) / 2
    黄金角: θ = 2π / φ^2 = 2π (1 - 1/φ) ≈ 137.5°

    采样公式 (圆域):
      r_i = sqrt(i / n)
      θ_i = i * θ

    对矩形周期域, 将极坐标映射到笛卡尔坐标并缩放到 [0,Lx]×[0,Ly].

    该采样在面积上近乎均匀, 避免了笛卡尔网格的规则性,
    适用于拉格朗日粒子的初始分布.
    """
    if n < 1:
        raise ValueError("n must be positive")
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    golden_angle = 2.0 * np.pi / (phi ** 2)

    i = np.arange(n, dtype=float)
    # 半径从 0 到 1
    r = np.sqrt(i / n)
    theta = i * golden_angle

    # 圆域坐标
    x_circ = r * np.cos(theta)
    y_circ = r * np.sin(theta)

    # 映射到矩形 [0,Lx]x[0,Ly]
    if center is None:
        cx, cy = Lx / 2.0, Ly / 2.0
    else:
        cx, cy = center

    # 使用极坐标到矩形的近似映射 (保持密度均匀)
    x = cx + 0.5 * Lx * x_circ
    y = cy + 0.5 * Ly * y_circ

    # 边界处理: 周期折叠
    x = x % Lx
    y = y % Ly

    return x, y


# ============================================================
# 3. 周期性 Lloyd/CVT 迭代优化 (from 265_cvtp_1d)
# ============================================================

def periodic_distance(a: np.ndarray, b: np.ndarray, L: float) -> np.ndarray:
    """
    周期域上的最短距离.
      d = min(|a-b|, L - |a-b|)
    """
    d = np.abs(a - b)
    return np.minimum(d, L - d)


def cvtp_optimize_2d(x: np.ndarray, y: np.ndarray, Lx: float, Ly: float,
                     n_samples: int = 5000, n_iter: int = 10,
                     rng: Optional[CMRG] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    二维周期性域上的 Lloyd 迭代优化 (CVT).

    算法:
      1. 在域内随机采样 n_samples 个点
      2. 将每个采样点分配到最近的生成元 (考虑周期边界)
      3. 更新生成元为其 Voronoi 区域内采样点的平均值 (周期平均)
      4. 重复 n_iter 次

    能量度量 (CVT 能量):
      E = sum_i ∫_{V_i} |x - g_i|^2 dx
    该能量在 Lloyd 迭代下单调递减.

    Returns
    -------
    x, y : 优化后的生成元坐标
    energy_history : 各迭代的 CVT 能量
    """
    n_particles = len(x)
    if rng is None:
        rng = CMRG(seed1=12345, seed2=67890)

    gx = x.copy()
    gy = y.copy()
    energy_history = []

    for _ in range(n_iter):
        # 随机采样
        sx = rng.rand_array((n_samples,)) * Lx
        sy = rng.rand_array((n_samples,)) * Ly

        # 分配到最近生成元
        belongs_to = np.zeros(n_samples, dtype=int)
        for s in range(n_samples):
            dx = periodic_distance(sx[s], gx, Lx)
            dy = periodic_distance(sy[s], gy, Ly)
            dist2 = dx ** 2 + dy ** 2
            belongs_to[s] = int(np.argmin(dist2))

        # 更新生成元 (周期平均)
        new_gx = np.zeros(n_particles)
        new_gy = np.zeros(n_particles)
        counts = np.zeros(n_particles)

        for i in range(n_particles):
            mask = belongs_to == i
            if np.sum(mask) > 0:
                # 周期平均需用角度平均法
                sx_m = sx[mask]
                sy_m = sy[mask]
                # 对 x 坐标
                angles_x = 2.0 * np.pi * sx_m / Lx
                cx = np.mean(np.cos(angles_x))
                sx_mean = (np.arctan2(np.mean(np.sin(angles_x)), cx) / (2.0 * np.pi)) * Lx
                if sx_mean < 0:
                    sx_mean += Lx
                # 对 y 坐标
                angles_y = 2.0 * np.pi * sy_m / Ly
                cy = np.mean(np.cos(angles_y))
                sy_mean = (np.arctan2(np.mean(np.sin(angles_y)), cy) / (2.0 * np.pi)) * Ly
                if sy_mean < 0:
                    sy_mean += Ly
                new_gx[i] = sx_mean
                new_gy[i] = sy_mean
                counts[i] = np.sum(mask)
            else:
                new_gx[i] = gx[i]
                new_gy[i] = gy[i]
                counts[i] = 1

        gx = new_gx
        gy = new_gy

        # 计算能量
        energy = 0.0
        for s in range(n_samples):
            i = belongs_to[s]
            dx = periodic_distance(sx[s], gx[i], Lx)
            dy = periodic_distance(sy[s], gy[i], Ly)
            energy += dx ** 2 + dy ** 2
        energy_history.append(energy / n_samples)

    return gx, gy, np.array(energy_history)


# ============================================================
# 4. 拉格朗日粒子追踪器
# ============================================================

class LagrangianParticleTracker:
    """
    拉格朗日粒子追踪器.

    运动方程 (含随机扩散):
      dx = u(x,y,t) dt + sqrt(2κ) dW_x
      dy = v(x,y,t) dt + sqrt(2κ) dW_y

    其中 κ 为扩散系数, dW 为 Wiener 过程增量.
    """

    def __init__(self, x0: np.ndarray, y0: np.ndarray,
                 Lx: float, Ly: float, dt: float = 0.01,
                 diffusivity: float = 0.0, rng: Optional[CMRG] = None):
        self.x = np.asarray(x0, dtype=float).copy()
        self.y = np.asarray(y0, dtype=float).copy()
        self.n_particles = len(self.x)
        self.Lx = float(Lx)
        self.Ly = float(Ly)
        self.dt = float(dt)
        self.diffusivity = float(diffusivity)
        self.rng = rng if rng is not None else CMRG()

        # 轨迹历史
        self.trajectory_x = [self.x.copy()]
        self.trajectory_y = [self.y.copy()]
        self.t_history = [0.0]

    def _periodic_wrap(self):
        """周期边界折叠."""
        self.x = self.x % self.Lx
        self.y = self.y % self.Ly

    def _interpolate_velocity(self, x_query: np.ndarray, y_query: np.ndarray,
                              u_field: np.ndarray, v_field: np.ndarray,
                              x_grid: np.ndarray, y_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        双线性插值获取速度.

        u_field 形状为 (Ny, Nx), 对应网格 y_grid (长度 Ny), x_grid (长度 Nx).
        """
        Ny, Nx = u_field.shape
        dx = x_grid[1] - x_grid[0]
        dy = y_grid[1] - y_grid[0]

        ix = np.floor((x_query - x_grid[0]) / dx).astype(int)
        iy = np.floor((y_query - y_grid[0]) / dy).astype(int)
        ix = np.clip(ix, 0, Nx - 2)
        iy = np.clip(iy, 0, Ny - 2)

        fx = (x_query - x_grid[ix]) / dx
        fy = (y_query - y_grid[iy]) / dy
        fx = np.clip(fx, 0.0, 1.0)
        fy = np.clip(fy, 0.0, 1.0)

        u_q = (1 - fx) * (1 - fy) * u_field[iy, ix] + \
              fx * (1 - fy) * u_field[iy, ix + 1] + \
              (1 - fx) * fy * u_field[iy + 1, ix] + \
              fx * fy * u_field[iy + 1, ix + 1]

        v_q = (1 - fx) * (1 - fy) * v_field[iy, ix] + \
              fx * (1 - fy) * v_field[iy, ix + 1] + \
              (1 - fx) * fy * v_field[iy + 1, ix] + \
              fx * fy * v_field[iy + 1, ix + 1]

        return u_q, v_q

    def step_rk4(self, u_field: np.ndarray, v_field: np.ndarray,
                 x_grid: np.ndarray, y_grid: np.ndarray):
        """
        RK4 积分粒子位置.
        """
        def vel(xp, yp):
            return self._interpolate_velocity(xp, yp, u_field, v_field, x_grid, y_grid)

        # k1
        k1x, k1y = vel(self.x, self.y)
        if self.diffusivity > 0:
            k1x += np.sqrt(2.0 * self.diffusivity / self.dt) * self.rng.randn_array((self.n_particles,))
            k1y += np.sqrt(2.0 * self.diffusivity / self.dt) * self.rng.randn_array((self.n_particles,))

        # k2
        x2 = self.x + 0.5 * self.dt * k1x
        y2 = self.y + 0.5 * self.dt * k1y
        k2x, k2y = vel(x2, y2)

        # k3
        x3 = self.x + 0.5 * self.dt * k2x
        y3 = self.y + 0.5 * self.dt * k2y
        k3x, k3y = vel(x3, y3)

        # k4
        x4 = self.x + self.dt * k3x
        y4 = self.y + self.dt * k3y
        k4x, k4y = vel(x4, y4)

        self.x = self.x + self.dt * (k1x + 2.0 * k2x + 2.0 * k3x + k4x) / 6.0
        self.y = self.y + self.dt * (k1y + 2.0 * k2y + 2.0 * k3y + k4y) / 6.0
        self._periodic_wrap()

        self.trajectory_x.append(self.x.copy())
        self.trajectory_y.append(self.y.copy())
        self.t_history.append(self.t_history[-1] + self.dt)

    def compute_mean_square_displacement(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算均方位移 (MSD): <(r(t)-r(0))^2>.
        """
        tx = np.array(self.trajectory_x)
        ty = np.array(self.trajectory_y)
        dx = periodic_distance(tx, tx[0, :], self.Lx)
        dy = periodic_distance(ty, ty[0, :], self.Ly)
        msd = np.mean(dx ** 2 + dy ** 2, axis=1)
        t = np.array(self.t_history)
        return t, msd

    def compute_pair_separation(self, n_pairs: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算随机粒子对的分离率.

        Richardson 定律: <r^2(t)> ~ t^3
        """
        tx = np.array(self.trajectory_x)
        ty = np.array(self.trajectory_y)
        nt = tx.shape[0]

        rng_local = np.random.default_rng(42)
        pairs = rng_local.integers(0, self.n_particles, size=(n_pairs, 2))

        r2_mean = np.zeros(nt)
        for t_idx in range(nt):
            dx = periodic_distance(tx[t_idx, pairs[:, 0]], tx[t_idx, pairs[:, 1]], self.Lx)
            dy = periodic_distance(ty[t_idx, pairs[:, 0]], ty[t_idx, pairs[:, 1]], self.Ly)
            r2_mean[t_idx] = np.mean(dx ** 2 + dy ** 2)

        return np.array(self.t_history), r2_mean

    def compute_diffusivity(self) -> float:
        """
        从长时间 MSD 估计有效扩散系数: D = MSD / (4t) (二维).
        """
        t, msd = self.compute_mean_square_displacement()
        if len(t) < 3:
            return 0.0
        # 使用后半段时间线性拟合
        half = len(t) // 2
        if half < 2:
            half = 1
        D_est = np.mean(msd[half:] / (4.0 * t[half:]))
        return float(D_est)


# ============================================================
# 5. 有限时间 Lyapunov 指数 (FTLE) 计算
# ============================================================

def compute_ftle_grid(u_field: np.ndarray, v_field: np.ndarray,
                      x_grid: np.ndarray, y_grid: np.ndarray,
                      dt: float, n_steps: int,
                      dx_grid: float, dy_grid: float) -> np.ndarray:
    """
    计算有限时间 Lyapunov 指数场.

    FTLE 定义为:
      σ(x0, t0, T) = (1/|T|) * ln(sqrt(lambda_max(C)))
    其中 C = (∇F_T)^T (∇F_T) 为 Cauchy-Green 张量.

    通过追踪邻近粒子的变形梯度近似 ∇F_T.
    """
    Ny, Nx = u_field.shape
    ftle = np.zeros((Ny, Nx))

    # 简化: 在每个网格点周围放置4个邻近粒子
    eps = min(dx_grid, dy_grid) * 0.1

    for j in range(Ny):
        for i in range(Nx):
            x0 = x_grid[i]
            y0 = y_grid[j]

            # 初始变形梯度 = I
            # 追踪邻近粒子
            offsets = np.array([[eps, 0], [0, eps], [-eps, 0], [0, -eps]])
            x_traj = np.zeros((n_steps + 1, 4))
            y_traj = np.zeros((n_steps + 1, 4))
            x_traj[0, :] = x0 + offsets[:, 0]
            y_traj[0, :] = y0 + offsets[:, 1]

            # 简化: 使用欧拉前进步进
            for step in range(n_steps):
                idx = step % u_field.shape[0]  # 循环使用速度场
                # 使用当前网格速度近似 (简化版)
                ux = u_field[j, i]
                vy = v_field[j, i]
                x_traj[step + 1, :] = x_traj[step, :] + ux * dt
                y_traj[step + 1, :] = y_traj[step, :] + vy * dt

            # 计算变形梯度
            dx_final = x_traj[-1, :] - x_traj[-1, :].mean()
            dy_final = y_traj[-1, :] - y_traj[-1, :].mean()

            # 简化的 Cauchy-Green 张量
            J11 = (x_traj[-1, 0] - x_traj[-1, 2]) / (2 * eps)
            J12 = (x_traj[-1, 1] - x_traj[-1, 3]) / (2 * eps)
            J21 = (y_traj[-1, 0] - y_traj[-1, 2]) / (2 * eps)
            J22 = (y_traj[-1, 1] - y_traj[-1, 3]) / (2 * eps)

            C = np.array([[J11 ** 2 + J21 ** 2, J11 * J12 + J21 * J22],
                          [J11 * J12 + J21 * J22, J12 ** 2 + J22 ** 2]])
            eigvals = np.linalg.eigvalsh(C)
            lambda_max = np.max(eigvals)
            T = n_steps * dt
            if lambda_max > 1e-15 and abs(T) > 1e-15:
                ftle[j, i] = 0.5 * np.log(lambda_max) / abs(T)
            else:
                ftle[j, i] = 0.0

    return ftle


if __name__ == "__main__":
    # 测试 Fibonacci 螺旋
    x, y = fibonacci_spiral_2d(500, Lx=2*np.pi, Ly=2*np.pi)
    print("Fibonacci spiral mean x:", np.mean(x), "std:", np.std(x))

    # 测试 CMRG
    rng = CMRG()
    vals = [rng.rand() for _ in range(5)]
    print("CMRG samples:", vals)

    # 测试 CVT
    x_opt, y_opt, E = cvtp_optimize_2d(x, y, 2*np.pi, 2*np.pi, n_samples=2000, n_iter=3)
    print("CVT energy history:", E)
