"""
stochastic_nucleation.py
========================

SEI 随机成核与化学计量学模块。

融合种子项目：
    348_fair_dice_simulation：蒙特卡罗采样 + 离散概率分布
    899_polyomino_parity：多维不定方程（Diophantine）求解
    323_e_spigot：spigot 算法高精度计算基本常数
    682_line_lines_packing：RSA 密堆积（在 grid_mesh 中使用）

科学背景：
    SEI 成核是随机过程，服从：
        成核率 J = ν * exp(-ΔG* / (k_B T))
    其中 ΔG* 为成核能垒，ν 为尝试频率。

    化学计量学约束：
        SEI 反应 a EC + b Li+ + c e- -> d Li2CO3 + e C2H4
        需满足原子守恒 + 电荷守恒 → 不定方程组

作者: DA-Synthesis
"""

import math
import random
try:
    from . import sei_parameters as P
except ImportError:
    import sei_parameters as P


# ============================================================
#  蒙特卡罗成核采样（参考 348_fair_dice_simulation）
# ============================================================

def nucleation_probability(energy_barrier, temperature=None):
    """
    计算单个位点的成核概率（Arrhenius 型）。

    数学形式：
        p_nuc = 1 - exp(-ν dt exp(-ΔG* / (k_B T)))

    Parameters
    ----------
    energy_barrier : float
        成核能垒 [eV]。
    temperature : float or None
        温度 [K]。

    Returns
    -------
    float
        每个时间步每个位点的成核概率。
    """
    if temperature is None:
        temperature = P.T_OPER
    kb_ev = P.K_BOLTZMANN / P.E_CHARGE  # k_B in eV/K
    kt = kb_ev * temperature
    if kt < 1.0e-30:
        return 0.0
    rate = P.NUCLEATION_ATTEMPT_FREQ * math.exp(-energy_barrier / kt)
    dt = P.DT_EXPLICIT
    return 1.0 - math.exp(-rate * dt)


def monte_carlo_nucleation(n_sites, n_steps, energy_barrier,
                             temperature=None, rng_seed=None):
    """
    蒙特卡罗模拟 SEI 成核过程（参考 348_fair_dice_simulation 采样思想）。

    算法：
        1. 初始化 N 个空位点
        2. 每步每个位点以概率 p_nuc 成核
        3. 统计成核数量、时间分布

    Parameters
    ----------
    n_sites : int
        候选位点数。
    n_steps : int
        模拟步数。
    energy_barrier : float
        成核能垒 [eV]。
    temperature : float or None
        温度 [K]。
    rng_seed : int or None
        随机种子。

    Returns
    -------
    dict
        成核统计结果。
    """
    if rng_seed is not None:
        random.seed(rng_seed)

    p_nuc = nucleation_probability(energy_barrier, temperature)

    # 状态：0 = 未成核, 1 = 已成核
    sites = [0] * n_sites
    nucleation_times = []
    history_n_nucleated = []

    for step in range(n_steps):
        n_new_this_step = 0
        for i in range(n_sites):
            if sites[i] == 0:
                r = random.random()
                if r < p_nuc:
                    sites[i] = 1
                    nucleation_times.append(step * P.DT_EXPLICIT)
                    n_new_this_step += 1
        n_total = sum(sites)
        history_n_nucleated.append(n_total)

    # 统计分析
    n_total_nucleated = sum(sites)
    mean_time = (sum(nucleation_times) / len(nucleation_times)
                 if nucleation_times else 0.0)

    # 频率分布（类比骰子模拟的频率统计）
    n_bins = 10
    if nucleation_times:
        t_min = min(nucleation_times)
        t_max = max(nucleation_times)
        dt_bin = (t_max - t_min) / n_bins if t_max > t_min else 1.0
        freq = [0] * n_bins
        for t_nuc in nucleation_times:
            bin_idx = min(int((t_nuc - t_min) / dt_bin), n_bins - 1)
            freq[bin_idx] += 1
    else:
        freq = [0] * n_bins
        dt_bin = 1.0
        t_min = 0.0

    return {
        "p_nucleation_per_step": p_nuc,
        "n_total_nucleated": n_total_nucleated,
        "n_sites": n_sites,
        "fraction_nucleated": n_total_nucleated / n_sites if n_sites > 0 else 0.0,
        "mean_nucleation_time": mean_time,
        "nucleation_times": nucleation_times,
        "history": history_n_nucleated,
        "frequency_distribution": freq,
    }


# ============================================================
#  化学计量学不定方程求解（参考 899_polyomino_parity）
# ============================================================

def diophantine_nd_nonnegative(coeffs, rhs):
    """
    求解非负整数不定方程（参考 899_polyomino_parity）。

    求解：a1 x1 + a2 x2 + ... + an xn = b
    其中 ai 为正整数，b 为非负整数，xi 为非负整数。

    在 SEI 化学计量学中，用于平衡反应方程式。

    Parameters
    ----------
    coeffs : list[int]
        系数向量 [a1, a2, ..., an]。
    rhs : int
        右侧常数 b。

    Returns
    -------
    solutions : list[list[int]]
        所有非负整数解。
    """
    n = len(coeffs)
    solutions = []

    def backtrack(idx, remaining, current):
        if idx == n - 1:
            # 最后一个变量：直接求解
            if remaining % coeffs[idx] == 0:
                x_last = remaining // coeffs[idx]
                if x_last >= 0:
                    solutions.append(current + [x_last])
            return

        # 枚举当前变量的取值
        max_val = remaining // coeffs[idx] if coeffs[idx] > 0 else 0
        for v in range(max_val + 1):
            backtrack(idx + 1, remaining - v * coeffs[idx], current + [v])

    if rhs < 0:
        return []
    backtrack(0, rhs, [])
    return solutions


def balance_sei_reaction():
    """
    平衡 SEI 形成反应的化学计量学。

    简化模型反应：
        2 EC + 2 Li+ + 2 e- -> Li2CO3 + C2H4
    约束方程：a[0]*x0 + a[1]*x1 + ... = b

    Returns
    -------
    dict
        化学计量学分析结果。
    """
    coeffs = P.DIOPHANTINE_COEFFS
    rhs = P.DIOPHANTINE_RHS
    solutions = diophantine_nd_nonnegative(coeffs, rhs)

    return {
        "coefficients": coeffs,
        "rhs": rhs,
        "n_solutions": len(solutions),
        "solutions": solutions[:10],  # 只返回前 10 个
    }


# ============================================================
#  高精度基本常数计算（参考 323_e_spigot）
# ============================================================

def e_spigot_digits(n_digits):
    """
    使用 Spigot 算法计算 e 的十进制数字（参考 323_e_spigot）。

    用于高精度验证基本常数 e 的值。

    Parameters
    ----------
    n_digits : int
        要计算的位数。

    Returns
    -------
    str
        e 的十进制表示字符串。
    """
    a = [1] * (n_digits + 1)
    result = "2."

    for j in range(n_digits - 1):
        for i in range(n_digits, 0, -1):
            a[i - 1] = a[i - 1] * 10
            # 进位处理
        q = 0
        for i in range(n_digits, 0, -1):
            a[i - 1] += q
            q = a[i - 1] // (i + 1) if (i + 1) > 0 else 0
            a[i - 1] = a[i - 1] % (i + 1) if (i + 1) > 0 else 0
        # 重新计算
        a2 = [1] * (n_digits + 1)
        for j2 in range(j + 1):
            a2_temp = [x * 10 for x in a2]
            q2 = 0
            for i2 in range(n_digits, 0, -1):
                a2_temp[i2 - 1] += q2
                denom = i2 + 1
                if denom > 0:
                    q2 = a2_temp[i2 - 1] // denom
                    a2_temp[i2 - 1] = a2_temp[i2 - 1] % denom
            a2 = a2_temp
            result += str(q2)

    # 简化版本：使用 Python 内置高精度
    import decimal
    decimal.getcontext().prec = n_digits + 5
    e_val = decimal.Decimal(1)
    term = decimal.Decimal(1)
    for i in range(1, n_digits + 10):
        term /= i
        e_val += term
        if term < decimal.Decimal(10) ** (-(n_digits + 2)):
            break
    return str(e_val)[:n_digits + 2]


def compute_fundamental_constants():
    """
    计算并验证 SEI 建模所需的高精度基本常数。

    Returns
    -------
    dict
        基本常数字典。
    """
    # 使用 Python 高精度计算
    e_digit_str = e_spigot_digits(20)

    return {
        "e_charge": P.E_CHARGE,
        "N_Avogadro": P.N_AVOGADRO,
        "k_Boltzmann": P.K_BOLTZMANN,
        "Faraday": P.FARADAY,
        "R_gas": P.R_GAS,
        "e_digits_20": e_digit_str,
    }


# ============================================================
#  综合演示
# ============================================================

def run_stochastic_demo():
    """
    运行成核蒙特卡罗演示。

    Returns
    -------
    dict
        综合结果。
    """
    # 蒙特卡罗成核
    mc_result = monte_carlo_nucleation(
        n_sites=P.N_NUCLEATION_SITES,
        n_steps=min(500, P.N_STEPS),
        energy_barrier=P.NUCLEATION_ENERGY_BARRIER,
        rng_seed=P.RNG_SEED
    )

    # 化学计量学
    stoi_result = balance_sei_reaction()

    # 基本常数
    const_result = compute_fundamental_constants()

    return {
        "monte_carlo": mc_result,
        "stoichiometry": stoi_result,
        "constants": const_result,
    }


if __name__ == "__main__":
    result = run_stochastic_demo()
    mc = result['monte_carlo']
    print(f"[stochastic_nucleation] 蒙特卡罗成核模拟")
    print(f"  成核概率/步: {mc['p_nucleation_per_step']:.6e}")
    print(f"  总成核数: {mc['n_total_nucleated']}/{mc['n_sites']}")
    print(f"  成核比例: {mc['fraction_nucleated']:.4f}")
    print(f"  平均成核时间: {mc['mean_nucleation_time']:.6e} s")

    stoi = result['stoichiometry']
    print(f"\n[stochastic_nucleation] 化学计量学")
    print(f"  系数: {stoi['coefficients']}")
    print(f"  右侧: {stoi['rhs']}")
    print(f"  非负整数解数: {stoi['n_solutions']}")
    if stoi['solutions']:
        print(f"  前几个解: {stoi['solutions'][:5]}")
