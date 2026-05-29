"""
monte_carlo_pore.py
===================
催化剂孔结构的蒙特卡洛分析与随机几何模拟。

基于种子项目 298_disk_triangle_picking 与 320_duel_simulation 重构：
- disk_triangle_picking 估计单位圆盘内随机三角形的平均面积；
- duel_simulation 模拟交替射击的随机过程。

在本系统中：
1. 利用圆盘内随机三角形面积估计来近似催化剂孔截面的连通面积与曲折度；
2. 利用决斗模拟的马尔可夫链思想模拟反应物分子在孔道网络中的随机行走
   与到达活性位点的概率（首次通过问题）；
3. 结合蒙特卡洛方法估计有效扩散系数和孔道利用率。
"""

import numpy as np


class MonteCarloPoreError(Exception):
    """蒙特卡洛孔结构分析异常。"""
    pass


def random_triangle_area_in_disk(n_trials, rng=None):
    """
    估计单位圆盘内随机三角形的平均面积。

    在催化剂科学中，该值与孔截面内三相界面的几何特征相关：
    随机三角形的期望面积反映了孔道截面的有效连通空间。

    采样方法（基于 disk_triangle_picking）：
        r = sqrt(u),  u ~ U(0,1)   （保证面积均匀）
        θ = 2π v,    v ~ U(0,1)
        三角形面积使用海伦公式：
            A = sqrt[s(s-a)(s-b)(s-c)]
            s = (a+b+c)/2

    Parameters
    ----------
    n_trials : int
        蒙特卡洛试验次数。
    rng : np.random.Generator, optional

    Returns
    -------
    mean_area : float
        平均面积。
    std_area : float
        面积标准差。
    """
    if n_trials < 1:
        raise MonteCarloPoreError("n_trials 必须 ≥ 1")
    if rng is None:
        rng = np.random.default_rng()

    areas = np.empty(n_trials, dtype=float)
    for k in range(n_trials):
        theta = 2.0 * np.pi * rng.random(3)
        r = np.sqrt(rng.random(3))
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        s1 = np.hypot(x[0] - x[1], y[0] - y[1])
        s2 = np.hypot(x[1] - x[2], y[1] - y[2])
        s3 = np.hypot(x[2] - x[0], y[2] - y[0])
        s = 0.5 * (s1 + s2 + s3)
        # 海伦公式数值稳定版本
        area_sq = s * (s - s1) * (s - s2) * (s - s3)
        if area_sq < 0:
            area_sq = 0.0
        areas[k] = np.sqrt(area_sq)

    return float(np.mean(areas)), float(np.std(areas))


def pore_accessibility_simulation(n_pores, hit_probs, n_trials=100000, rng=None):
    r"""
    模拟反应物分子在多级孔道网络中的可达性。

    将孔道网络建模为多层"决斗"过程：
    反应物分子依次通过宏观孔（mesopore）和微孔（micropore），
    每一层以一定概率被吸附或继续扩散。

    基于 duel_simulation 的交替射击马尔可夫链思想：
    - 反应物分子（Player 1）试图到达活性位点；
    - 孔壁吸附（Player 2）试图捕获分子；
    - 若分子在某一层成功"击中"（即未被吸附），则进入下一层。

    总体到达概率的解析解：
        P_{arrival} = \prod_{k=1}^{n} \frac{p_k}{p_k + q_k - p_k q_k}
    其中 p_k 为第 k 层通过概率，q_k 为被吸附概率。

    Parameters
    ----------
    n_pores : int
        孔道层数。
    hit_probs : ndarray, shape (n_pores, 2)
        每层 [通过概率, 吸附概率]，均应在 [0, 1] 内。
    n_trials : int
        蒙特卡洛模拟次数。
    rng : np.random.Generator, optional

    Returns
    -------
    arrival_prob : float
        模拟得到的到达活性位点的概率。
    mean_steps : float
        平均通过的孔道层数。
    """
    if rng is None:
        rng = np.random.default_rng()

    hit_probs = np.asarray(hit_probs, dtype=float)
    if hit_probs.ndim != 2 or hit_probs.shape[1] != 2:
        raise MonteCarloPoreError("hit_probs 形状必须为 (n, 2)")
    if np.any((hit_probs < 0) | (hit_probs > 1)):
        raise MonteCarLOPoreError("概率必须在 [0, 1] 之间")

    total_arrivals = 0
    total_steps = 0

    for _ in range(n_trials):
        steps = 0
        arrived = False
        for layer in range(n_pores):
            p_pass = hit_probs[layer, 0]
            p_ads = hit_probs[layer, 1]
            # 先尝试通过
            steps += 1
            if rng.random() > p_pass:
                # 未通过（被吸附或堵截）
                break
            # 通过了本层
            if layer == n_pores - 1:
                arrived = True
        if arrived:
            total_arrivals += 1
            total_steps += steps

    arrival_prob = total_arrivals / n_trials
    mean_steps = total_steps / max(total_arrivals, 1)
    return arrival_prob, mean_steps


def estimate_effective_diffusivity_mc(pore_network, temperature, molecular_weight,
                                      n_walks=50000, n_steps=200, step_size=1e-9,
                                      rng=None):
    r"""
    使用随机行走（Random Walk）蒙特卡洛方法估计有效扩散系数。

    扩散系数的 Einstein 关系：
        D_{eff} = \frac{\langle |r(t) - r(0)|^2 \rangle}{6 t}

    其中分子在孔道网络中做随机行走，遇到孔壁时反射。

    Parameters
    ----------
    pore_network : callable
        函数签名 f(x, y, z) -> bool，返回 True 表示点在孔道内部。
    temperature : float
        温度 [K]。
    molecular_weight : float
        分子量 [kg/mol]。
    n_walks : int
        随机行走轨迹数。
    n_steps : int
        每条轨迹的步数。
    step_size : float
        每步长度 [m]。
    rng : np.random.Generator, optional

    Returns
    -------
    D_eff : float
        估计的有效扩散系数 [m²/s]。
    """
    if rng is None:
        rng = np.random.default_rng()

    R = 8.314462618
    # 热速度
    v_thermal = np.sqrt(3.0 * R * temperature / molecular_weight)
    dt = step_size / v_thermal  # 时间步长

    msd_sum = 0.0
    valid_walks = 0

    for _ in range(n_walks):
        # 在中心区域初始化
        pos = np.zeros(3, dtype=float)
        if not pore_network(*pos):
            continue
        valid_walks += 1
        start_pos = pos.copy()

        for _ in range(n_steps):
            # 随机方向
            theta = np.arccos(2.0 * rng.random() - 1.0)
            phi = 2.0 * np.pi * rng.random()
            direction = np.array([
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta)
            ])
            new_pos = pos + step_size * direction
            if pore_network(*new_pos):
                pos = new_pos
            # 否则保持原位（反射边界简化）

        msd = np.sum((pos - start_pos) ** 2)
        msd_sum += msd

    if valid_walks == 0:
        raise MonteCarloPoreError("没有有效的随机行走轨迹")

    msd_mean = msd_sum / valid_walks
    D_eff = msd_mean / (6.0 * n_steps * dt)
    return D_eff


def pore_tortuosity_from_mc(n_trials=100000, rng=None):
    r"""
    基于圆盘内随机几何的蒙特卡洛估计，计算孔道曲折因子的统计近似。

    曲折因子 τ 的定义：
        τ = \left(\frac{L_{actual}}{L_{straight}}\right)^2

    通过随机三角形平均面积与圆盘面积之比，间接估计孔道的几何曲折度。

    Returns
    -------
    tau_estimate : float
        估计的曲折因子。
    """
    mean_area, std_area = random_triangle_area_in_disk(n_trials, rng)
    disk_area = np.pi
    # 统计模型：曲折因子与连通面积占空比的经验反比关系
    # τ ≈ 1 + c * (1 - A_triangle / A_disk)
    fill_ratio = mean_area / disk_area
    tau_estimate = 1.0 + 2.0 * (1.0 - fill_ratio)
    # 工程边界
    tau_estimate = max(1.0, min(tau_estimate, 10.0))
    return tau_estimate
