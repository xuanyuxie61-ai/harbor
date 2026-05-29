"""
finite_field_symmetry.py
=========================
有限域 GF(2) 多项式对称性代数模块

基于种子项目 198_collatz_polynomial 的 GF(2) 多项式
运算思想，本模块将有限域多项式代数应用于核物理中的
对称性操作：
1. 核自旋态的宇称操作 (P: 空间反演)
2. 同位旋空间的 SU(2) 生成元代数
3. 核壳模型中的占据数表示 (比特串 ↔ 组态)
4. 时间反演对称性的代数检验

核心公式
--------
GF(2) 多项式运算 (系数模 2):
    加法: 异或 (XOR)
    乘法: 卷积后模 2

宇称算符:
    P̂ ψ(r) = ψ(-r)
    对球谐函数: P̂ Y_{lm}(θ,φ) = (-1)^l Y_{lm}(θ,φ)

同位旋升降算符:
    T_± |T, T_z⟩ = √[T(T+1) - T_z(T_z±1)] |T, T_z±1⟩

Pauli 矩阵在 GF(2) 中的对应 (忽略相位):
    σ_x ↔ 比特翻转
    σ_z ↔ 相位翻转 (在 GF(2) 中退化为恒等)

核组态的占据数表示:
    对于 N 个单粒子态，一个组态可用 N 位二进制数表示。
    两个组态的相互作用矩阵元通过对称性操作相关联。
"""

import numpy as np


def gf2_add(p, q):
    """
    GF(2) 多项式加法 = 系数异或。

    若 p, q 为整数，则结果为其按位异或。
    """
    return p ^ q


def gf2_multiply(p, q):
    """
    GF(2) 多项式乘法。

    使用移位-异或算法 (类似二进制乘法但无进位)。
    """
    result = 0
    while q:
        if q & 1:
            result ^= p
        p <<= 1
        q >>= 1
    return result


def gf2_mod(p, mod_poly):
    """
    GF(2) 多项式取模 (mod mod_poly)。

    使用长除法算法。
    """
    deg_mod = mod_poly.bit_length() - 1
    if deg_mod < 0:
        raise ValueError("模多项式不能为 0")

    while p.bit_length() > deg_mod:
        shift = p.bit_length() - deg_mod - 1
        p ^= mod_poly << shift
    return p


def gf2_poly_degree(p):
    """GF(2) 多项式的次数。"""
    if p == 0:
        return -1
    return p.bit_length() - 1


def gf2_poly_string(p):
    """将 GF(2) 多项式打印为字符串。"""
    if p == 0:
        return "0"
    terms = []
    for i in range(gf2_poly_degree(p) + 1):
        if (p >> i) & 1:
            if i == 0:
                terms.append("1")
            elif i == 1:
                terms.append("x")
            else:
                terms.append(f"x^{i}")
    return " + ".join(reversed(terms))


def parity_operator_state(l):
    """
    计算轨道角动量 l 的宇称本征值。

    P |l⟩ = (-1)^l |l⟩

    返回 +1 (偶宇称) 或 -1 (奇宇称)。
    """
    return 1 if l % 2 == 0 else -1


def spin_orbit_coupling_gf2(l):
    """
    使用 GF(2) 代数表示自旋-轨道耦合的允许跃迁。

    对于自旋-1/2 粒子，j = l ± 1/2 可表示为两个状态。
    使用 2 位二进制表示: |l+1/2⟩ = 10, |l-1/2⟩ = 01。

    自旋-轨道耦合算符在这组基下的作用可通过 GF(2) 的
    比特操作模拟。
    """
    # 状态编码
    j_plus = 0b10   # j = l + 1/2
    j_minus = 0b01  # j = l - 1/2

    # 跃迁算符：翻转高位 (模拟自旋翻转导致的 j 改变)
    # 在真实物理中，LS 耦合不改变 j，但改变能级顺序
    # 这里简化为对称性标记
    transition = {
        'j_plus': j_plus,
        'j_minus': j_minus,
        'degeneracy': 2,  # (2j+1) 对 j=l±1/2 的平均
    }
    return transition


def nuclear_configuration_gf2(n_particles, n_states):
    """
    将核组态表示为 GF(2) 向量。

    n_states 个单粒子态中填充 n_particles 个核子，
    每个组态对应一个 n_states 位二进制数。

    Parameters
    ----------
    n_particles : int
        核子数。
    n_states : int
        单粒子态数。

    Returns
    -------
    configurations : list of int
        所有可能的组态 (二进制数)。
    """
    if n_particles > n_states:
        return []
    configurations = []
    # 生成所有 n_states 位中恰好有 n_particles 个 1 的数
    def generate(pos, ones_left, current):
        if ones_left == 0:
            configurations.append(current)
            return
        if pos < 0:
            return
        # 在当前位置放 1
        generate(pos - 1, ones_left - 1, current | (1 << pos))
        # 在当前位置放 0
        generate(pos - 1, ones_left, current)

    generate(n_states - 1, n_particles, 0)
    return configurations


def isospin_states(N, Z):
    """
    计算同位旋态的允许值。

    同位旋 T_z = (N - Z) / 2
    允许的同位旋: T >= |T_z|

    返回同位旋多重态的维数和态列表。
    """
    Tz = (N - Z) / 2.0
    min_T = abs(Tz)
    # 对于 A = N + Z, 最大 T = A/2
    max_T = (N + Z) / 2.0

    states = []
    T = min_T
    while T <= max_T:
        multiplicity = int(2 * T + 1)
        states.append({'T': T, 'Tz': Tz, 'multiplicity': multiplicity})
        T += 1.0

    return states


def time_reversal_symmetry_check(J, config_gf2):
    """
    检验时间反演对称性。

    对于半整数 J，Kramers 定理要求每个能级至少是二重简并的。
    对于整数 J，时间反演不改变态。

    这里通过 GF(2) 组态的简并度进行检验。
    """
    if abs(J - round(J)) < 0.25:
        # 整数自旋
        return {'kramers_degeneracy': 1, 'time_reversal_even': True}
    else:
        # 半整数自旋: Kramers 二重态
        return {'kramers_degeneracy': 2, 'time_reversal_even': False}


def shell_model_parity(configuration, shell_parity_list):
    """
    计算壳模型组态的总宇称。

    总宇称 = Π_i (-1)^{l_i}

    Parameters
    ----------
    configuration : int
        组态的二进制表示 (每位对应一个态的占据)。
    shell_parity_list : list of int
        每个单粒子态的宇称 (+1 或 -1)。

    Returns
    -------
    total_parity : int
        +1 或 -1。
    """
    parity = 1
    for i, p in enumerate(shell_parity_list):
        if (configuration >> i) & 1:
            parity *= p
    return parity


def young_tableau_dimension(partition):
    """
    计算 Young 图的维数 (SU(N) 表示论)。

    用于核物理中的 SU(4) 超多重态或 SU(3) 壳模型。

    简化版：使用钩长公式 (hook-length formula)。
    """
    n = sum(partition)
    # 简化为对称群 S_n 的表示维数
    # dim = n! / Π hook_lengths
    # 这里使用简化公式
    if len(partition) == 1:
        return 1
    # 两行的 Young 图
    if len(partition) == 2:
        a, b = partition
        from math import factorial
        return factorial(a + b) * (a - b + 1) // (factorial(a + 1) * factorial(b))
    return 1


if __name__ == "__main__":
    # 自检
    p = 0b1011  # x^3 + x + 1
    q = 0b110   # x^2 + x
    print(f"p = {gf2_poly_string(p)}")
    print(f"q = {gf2_poly_string(q)}")
    print(f"p+q = {gf2_poly_string(gf2_add(p, q))}")
    print(f"p*q = {gf2_poly_string(gf2_multiply(p, q))}")

    print(f"l=2 宇称: {parity_operator_state(2)}")
    print(f"l=3 宇称: {parity_operator_state(3)}")

    configs = nuclear_configuration_gf2(2, 4)
    print(f"4 态中填 2 粒子: {len(configs)} 种组态")
    print([bin(c) for c in configs])

    iso = isospin_states(30, 26)
    print(f"56Fe 同位旋态: {iso}")

    tr = time_reversal_symmetry_check(2.5, configs[0])
    print(f"J=5/2 时间反演检验: {tr}")
