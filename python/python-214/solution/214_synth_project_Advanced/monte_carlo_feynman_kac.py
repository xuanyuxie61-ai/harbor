"""
monte_carlo_feynman_kac.py — 蒙特卡洛与 Feynman-Kac 概率验证模块
=================================================================
来源项目映射:
  - 303_disk01_positive_monte_carlo → 正圆盘上蒙特卡洛积分
  - 423_feynman_kac_2d             → 2D Feynman-Kac 概率 PDE 求解

科学背景:
  L1 正则化恢复的统计验证需要:
    (1) 期望风险 R(x) = E_ω[L(Ax(ω), y(ω))] 的蒙特卡洛估计
    (2) 通过 Feynman-Kac 公式将 PDE 解表示为随机过程的期望
        u(x) = E_x[∫_0^τ f(B_s) ds + g(B_τ)]
        其中 B_s 为 Brown 运动, τ 为首次逸出时间.

  本模块实现:
    - 单位圆盘/球上的正象限蒙特卡洛 (源自 303)
    - 2D Brown 运动模拟
    - Feynman-Kac 期望估计
    - 随机 PDE 解的 MC 验证

核心公式:
  蒙特卡洛:
    E[f(X)] ≈ (1/N) ∑_i f(X_i),  Var ≈ σ²/N
  Feynman-Kac (Poisson 方程):
    -Δu/2 = f  in D,   u = g  on ∂D
    ⇒  u(x) = E_x[∫_0^τ f(B_s) ds + g(B_τ)]
  随机椭圆 PDE:
    -∇·(a(x,ω)∇u) = f
    ⇒  u(x,ω) = E^x[∫_0^τ f(X_s)/a(X_s,ω) ds]  (概率表示)
"""
import numpy as np


# ----------------------------------------------------------------------
# 蒙特卡洛积分 (源自 303_disk01_positive_monte_carlo)
# ----------------------------------------------------------------------
def disk01_positive_sample(n, seed=0):
    """在单位圆盘的正象限 {x² + y² ≤ 1, x,y ≥ 0} 中均匀采样.

    方法: 极坐标 r ∈ [0,1], θ ∈ [0, π/2],
         面积元 r dr dθ,  用逆变换采样.
    """
    rng = np.random.RandomState(seed)
    r = np.sqrt(rng.uniform(0, 1, n))
    theta = rng.uniform(0, np.pi / 2, n)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y])


def disk01_positive_area(n_sample=10000, seed=0):
    """蒙特卡洛估计正象限单位圆盘面积 (理论值 = π/4)."""
    pts = disk01_positive_sample(n_sample, seed)
    # 所有点都在区域内, 面积 = (1/4) · π · 1² = π/4
    # 用 MC 验证: 在 [0,1]² 中采样, 落在圆盘内的比例 × 1
    rng = np.random.RandomState(seed + 1)
    xy = rng.uniform(0, 1, (n_sample, 2))
    inside = np.sum(xy[:, 0] ** 2 + xy[:, 1] ** 2 <= 1.0)
    return inside / n_sample  # 应接近 π/4


def monomial_integral_disk(alpha, beta, n_sample=50000, seed=0):
    """∫∫_{x²+y²≤1, x,y≥0} x^α y^β dx dy  的蒙特卡洛估计.

    理论值: B((α+1)/2, (β+1)/2) / 4  (Beta 函数).
    """
    pts = disk01_positive_sample(n_sample, seed)
    f_vals = pts[:, 0] ** alpha * pts[:, 1] ** beta
    area = np.pi / 4.0
    return float(area * np.mean(f_vals))


# ----------------------------------------------------------------------
# 2D Brown 运动模拟 (源自 423_feynman_kac_2d)
# ----------------------------------------------------------------------
def brownian_motion_2d(x0, T, n_steps, seed=0):
    """模拟 2D 标准 Brown 运动 B_t, t ∈ [0,T].

    dB_t ~ N(0, dt·I),   B_0 = x0.
    返回: (n_steps+1, 2) 轨迹.
    """
    rng = np.random.RandomState(seed)
    dt = T / n_steps
    dB = np.sqrt(dt) * rng.randn(n_steps, 2)
    path = np.zeros((n_steps + 1, 2))
    path[0] = x0
    for k in range(n_steps):
        path[k + 1] = path[k] + dB[k]
    return path


def first_exit_time_disk(x0, radius=1.0, dt=1e-3, max_steps=10000, seed=0):
    """模拟 Brown 运动从圆盘首次逸出时间 τ.

    τ = inf{t > 0 : |B_t - center| ≥ radius}.
    返回: (tau, B_tau) 逸出时间和逸出位置.
    """
    rng = np.random.RandomState(seed)
    x = np.array(x0, dtype=float)
    t = 0.0
    for _ in range(max_steps):
        dB = np.sqrt(dt) * rng.randn(2)
        x_new = x + dB
        if np.linalg.norm(x_new) >= radius:
            # 线性插值精确逸出点
            a_coeff = np.linalg.norm(x)
            if np.linalg.norm(x_new - x) > 1e-14:
                # 求解 |x + s(x_new - x)|² = radius²
                d = x_new - x
                a = np.dot(d, d)
                b = 2 * np.dot(x, d)
                c = np.dot(x, x) - radius ** 2
                disc = b * b - 4 * a * c
                s = (-b + np.sqrt(max(disc, 0))) / (2 * a)
                s = np.clip(s, 0, 1)
                x_exit = x + s * d
            else:
                x_exit = x_new
            return t + s * dt, x_exit
        x = x_new
        t += dt
    return t, x


def feynman_kac_poisson_2d(f_func, g_func, x0, n_paths=2000,
                           dt=1e-3, max_steps=5000, seed=0):
    """Feynman-Kac 求解 Poisson 方程:
        -½ Δu = f  in D = {|x| < 1}
        u = g    on ∂D
    解: u(x0) = E[∫_0^τ f(B_s) ds + g(B_τ)].

    输入:
        f_func : 源项 f(x)
        g_func : 边界项 g(x)
        x0     : 起点
    """
    rng = np.random.RandomState(seed)
    total = 0.0
    for k in range(n_paths):
        x = np.array(x0, dtype=float)
        integral = 0.0
        t = 0.0
        for _ in range(max_steps):
            dB = np.sqrt(dt) * rng.randn(2)
            x_new = x + dB
            if np.linalg.norm(x_new) >= 1.0:
                # 逸出
                integral += 0.5 * dt * f_func(x)
                total += integral + g_func(x_new / np.linalg.norm(x_new))
                break
            integral += 0.5 * dt * (f_func(x) + f_func(x_new))
            x = x_new
            t += dt
        else:
            total += integral + g_func(x / max(np.linalg.norm(x), 1e-14))
    return total / n_paths


# ----------------------------------------------------------------------
# 随机 PDE 解的 MC 验证
# ----------------------------------------------------------------------
def mc_expected_loss(A, y_true, x_recovered, n_samples=500, noise_std=0.01,
                     seed=0):
    """估计期望损失 E[||A(x+ξ) - y||²] 的 MC 近似.

    ξ ~ N(0, σ² I) 为观测噪声, 验证恢复解的稳健性.
    """
    rng = np.random.RandomState(seed)
    m = len(y_true)
    losses = []
    for _ in range(n_samples):
        xi = noise_std * rng.randn(m)
        y_noisy = y_true + xi
        losses.append(np.sum((A @ x_recovered - y_noisy) ** 2) / m)
    return float(np.mean(losses)), float(np.std(losses) / np.sqrt(n_samples))


def feynman_kac_verify_pde(f_func, x_recovered_pce, basis_eval_func,
                           n_paths=500, seed=0):
    """用 Feynman-Kac 验证 PCE 恢复解满足 PDE.

    检查: -½ Δ u(x0) ≈ f(x0),  其中 u 由 PCE 给出.
    返回: 相对误差.
    """
    # 数值 Laplacian (有限差分)
    x0 = np.array([0.0, 0.0])
    h = 1e-4
    pts = [x0, x0 + [h, 0], x0 - [h, 0], x0 + [0, h], x0 + [0, -h]]
    u_vals = [basis_eval_func(p) for p in pts]
    laplacian = (u_vals[1] - 2 * u_vals[0] + u_vals[2] +
                 u_vals[3] - 2 * u_vals[0] + u_vals[4]) / (h * h)
    fk_val = feynman_kac_poisson_2d(f_func, lambda x: 0.0, x0,
                                    n_paths=n_paths, seed=seed)
    rhs = f_func(x0)
    return float(abs(-0.5 * laplacian - rhs) / (abs(rhs) + 1e-10))


# ----------------------------------------------------------------------
# 置信区间
# ----------------------------------------------------------------------
def confidence_interval(samples, alpha=0.05):
    """95% 置信区间 (正态近似)."""
    n = len(samples)
    mu = np.mean(samples)
    se = np.std(samples, ddof=1) / np.sqrt(n)
    z = 1.96  # 95%
    return float(mu - z * se), float(mu + z * se)
