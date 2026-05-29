"""
data_generator.py — 超大规模异构科学数据生成器
===============================================
融合来源:
  - 181_circle_monte_carlo (圆周蒙特卡洛采样与Gamma函数积分)
  - 1086_sir_ode (SIR传染病ODE模型)
  - 700_logistic_bifurcation (Logistic映射混沌动力学)
  - 020_artery_pde (动脉血管PDE模型)
  - 343_euler (Euler法数值积分)

模拟高能物理实验（如LHC-like探测器）产生的异构事件数据流。
数据量远超内存容量，键值为复合物理量（时间戳+动量+能量），
需在外排序后供物理分析流水线使用。
"""

import math
import random
from typing import List, Tuple


def circle_monte_carlo_integrand(theta: float, e1: int, e2: int) -> float:
    """
    单位圆上的单项式采样值：
        f(θ; e1, e2) = (cos θ)^{e1} · (sin θ)^{e2}

    其精确积分为：
        I(e1, e2) = ∫_0^{2π} cos^{e1}(θ) sin^{e2}(θ) dθ
                  = 2 · Γ((e1+1)/2) · Γ((e2+1)/2) / Γ((e1+e2+2)/2)
    当 e1 或 e2 为奇数时积分值为零。
    """
    return (math.cos(theta) ** e1) * (math.sin(theta) ** e2)


def exact_circle_integral(e1: int, e2: int) -> float:
    """
    利用Gamma函数解析求圆周积分值。
    """
    if e1 % 2 == 1 or e2 % 2 == 1:
        return 0.0
    # 使用递推避免直接调用gamma函数的大数问题
    # Γ(n+1/2) = (2n)! / (4^n · n!) · √π
    def gamma_half(k: int) -> float:
        # k = (e+1)/2, e为偶数时k为半整数
        # 例如 e=0 -> k=0.5 -> Γ(0.5)=√π
        # e=2 -> k=1.5 -> Γ(1.5)=0.5√π
        n = int(k - 0.5)
        if n < 0:
            return math.sqrt(math.pi)
        result = math.sqrt(math.pi)
        for i in range(1, n + 1):
            result *= (i - 0.5)
        return result

    k1 = (e1 + 1) / 2.0
    k2 = (e2 + 1) / 2.0
    k12 = (e1 + e2 + 2) / 2.0
    return 2.0 * gamma_half(k1) * gamma_half(k2) / gamma_half(k12)


def generate_monte_carlo_keys(n: int, seed: int = 42) -> List[float]:
    """
    基于圆周蒙特卡洛采样生成n个物理事件的方向相关键值。

    利用方向角 θ 的均匀分布生成事件方向，键值编码为：
        key_i = (cos θ_i)^2 · (sin θ_i)^2 + ε_i
    其中 ε_i 为小量数值噪声。
    """
    random.seed(seed)
    keys = []
    for _ in range(n):
        theta = random.uniform(0.0, 2.0 * math.pi)
        val = circle_monte_carlo_integrand(theta, 2, 2)
        # 添加数值噪声模拟探测器分辨率
        noise = random.gauss(0.0, 0.001)
        keys.append(max(val + noise, 0.0))
    return keys


class SIRDataFlow:
    """
    SIR 模型模拟数据在分布式探测节点间的传播动态。

    状态向量 y = [S, I, R]，动力学方程：
        dS/dt = -α·S·I/N + γ·R
        dI/dt =  α·S·I/N - β·I
        dR/dt =  β·I - γ·R

    映射关系：
        S(t): 空闲计算节点数
        I(t): 正在处理数据（感染态）的节点数
        R(t): 已完成处理进入恢复态的节点数
        N = S + I + R: 总节点数（守恒）
    """

    def __init__(self, alpha: float, beta: float, gamma: float, N: float):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.N = N

    def rhs(self, t: float, y: List[float]) -> List[float]:
        S, I, R = y
        # 边界保护
        S = max(S, 0.0)
        I = max(I, 0.0)
        R = max(R, 0.0)
        dSdt = -self.alpha * S * I / self.N + self.gamma * R
        dIdt = self.alpha * S * I / self.N - self.beta * I
        dRdt = self.beta * I - self.gamma * R
        return [dSdt, dIdt, dRdt]

    def simulate_euler(self, S0: float, I0: float, R0: float,
                       t_end: float, n_steps: int) -> Tuple[List[float], List[float], List[float], List[float]]:
        """
        显式Euler法积分SIR系统。

        时间步长 h = t_end / n_steps，截断误差 O(h)。
        """
        h = t_end / n_steps
        t_vals = [i * h for i in range(n_steps + 1)]
        S, I, R = S0, I0, R0
        S_vals, I_vals, R_vals = [S0], [I0], [R0]
        for i in range(n_steps):
            t = t_vals[i]
            dS, dI, dR = self.rhs(t, [S, I, R])
            S += h * dS
            I += h * dI
            R += h * dR
            # 非负裁剪与守恒修正
            S = max(S, 0.0)
            I = max(I, 0.0)
            R = max(R, 0.0)
            total = S + I + R
            if total > 1e-15 and abs(total - self.N) > 1e-6:
                scale = self.N / total
                S *= scale
                I *= scale
                R *= scale
            S_vals.append(S)
            I_vals.append(I)
            R_vals.append(R)
        return t_vals, S_vals, I_vals, R_vals

    def basic_reproduction_number(self) -> float:
        """
        基本再生数 R0 = α / β。

        若 R0 > 1，数据传播呈指数增长；若 R0 < 1，传播自然消退。
        在外排序中，R0 > 1 对应高并发写入场景，需增大缓冲区。
        """
        if self.beta < 1e-15:
            return float('inf')
        return self.alpha / self.beta


def logistic_chaotic_sequence(n: int, r: float, x0: float = 0.5) -> List[float]:
    """
    Logistic 映射生成混沌伪随机序列：
        x_{n+1} = r · x_n · (1 - x_n)

    参数 r 控制动力学行为：
        r < 3:   收敛到不动点
        3 < r < 3.57: 周期倍增分岔
        r > 3.57: 混沌区域（对初值敏感依赖）

    该序列用于生成具有长程相关性的数据键值，模拟高能物理中
    簇射事件的自相似结构。
    """
    seq = [x0]
    x = x0
    for _ in range(n - 1):
        x = r * x * (1.0 - x)
        # 数值边界保护
        if x <= 0.0:
            x = 1e-12
        elif x >= 1.0:
            x = 1.0 - 1e-12
        seq.append(x)
    return seq


def generate_heterogeneous_dataset(
    total_records: int,
    memory_limit: int,
    seed: int = 199
) -> List[Tuple[float, ...]]:
    """
    生成异构科学数据集，模拟LHC-like粒子物理实验输出。

    每条记录为多元组 (timestamp, px, py, pz, energy, noise_key)，其中：
        timestamp: 事件时间戳（由SIR模型驱动的节点负载决定）
        px, py: 横向动量分量（由圆周MC采样生成）
        pz: 纵向动量（由Logistic混沌序列生成）
        energy: 能量沉积（由动脉PDE模型生成）
        noise_key: 综合噪声键值（用于外排序测试）

    数据量 total_records 远大于 memory_limit，强制触发外排序路径。
    """
    random.seed(seed)

    # 1. SIR模型生成时间戳负载模式
    sir = SIRDataFlow(alpha=0.3, beta=0.1, gamma=0.05, N=1000.0)
    _, S_vals, I_vals, _ = sir.simulate_euler(
        S0=990.0, I0=10.0, R0=0.0, t_end=100.0, n_steps=total_records
    )

    # 2. Logistic混沌序列生成长程相关键
    logistic_seq = logistic_chaotic_sequence(total_records, r=3.9, x0=0.314159)

    # 3. 圆周MC生成方向采样
    mc_keys = generate_monte_carlo_keys(total_records, seed=seed + 1)

    # 4. 动脉PDE驱动的能量项（简化为正弦调制）
    omega = 2.0 * math.pi / 50.0
    energy_base = 100.0

    dataset = []
    for i in range(total_records):
        # 时间戳由SIR的感染节点数驱动（峰值对应高事件率）
        timestamp = 1000.0 * i / total_records + 50.0 * math.sin(0.1 * i) + 10.0 * I_vals[i] / sir.N

        # 动量分量
        px = mc_keys[i] * math.cos(2.0 * math.pi * logistic_seq[i])
        py = mc_keys[i] * math.sin(2.0 * math.pi * logistic_seq[i])
        pz = logistic_seq[i] * 100.0

        # 能量沉积（带脉动调制）
        energy = energy_base + 30.0 * math.sin(omega * i) + 5.0 * random.gauss(0.0, 1.0)
        energy = max(energy, 0.0)

        # 综合排序键：时间戳主导 + 能量次要
        composite_key = timestamp + 0.01 * energy + 0.0001 * random.gauss(0.0, 1.0)

        record = (composite_key, timestamp, px, py, pz, energy, logistic_seq[i])
        dataset.append(record)

    return dataset
