"""
monomial_symmetry.py — 多项式对称化与置换不变可观测量
=====================================================
本模块实现多变量多项式的置换对称化,
用于构建顶夸克质量测量中的置换不变可观测量。

数学基础:
  在 tt̄ → (b l⁺ ν)(b̄ l⁻ ν̄) 中,
  两个 b 喷注不可区分, 导致 2! = 2 种 jet assignment。
  更一般地, 对于 n 个不可区分粒子, 有 n! 种置换。

  对称化操作: 对多项式系数进行群平均:
    c_sym[z] = (1/|S_d|) Σ_{π∈S_d} c[π(z)]

  其中 S_d 是 d 个变量的对称群, |S_d| = d!。

  不变基构造:
  - 初等对称多项式:
    e₁ = Σ x_i,  e₂ = Σ_{i<j} x_i x_j,  e₃ = x₁x₂x₃
  - 幂和:
    p_k = Σ x_i^k
  - Newton 恒等式联系两者:
    k e_k = Σ_{j=1}^k (-1)^{j-1} e_{k-j} p_j

  在 tt̄ 事件中的不可区分性:
    - 2 个 b-jets: 对称化 over S₂
    - 2 个轻子: 对称化 over S₂ (对双轻道)
    - 多个 jets: 选择最佳 assignment (组合优化)

  轨道代表元:
    等价类 [z] = {π(z) : π ∈ S_d}
    代表元选为非降序排列 z₁ ≤ z₂ ≤ ... ≤ z_d

  映射种子项目:
    - 776_monomial_symmetrize: 多项式系数对称化算法
"""

import numpy as np
from itertools import permutations


def vector_rank(d, b, a):
    """
    计算 d 维 b 基向量的线性索引 (混合进制编码)。

    rank = 1 + Σ_{i=1}^{d} (a_i - 1) × b^{d-i}

    例如: d=3, b=2, a=[1,2,1] → rank = 1 + 0×4 + 1×2 + 0×1 = 3

    参数:
        d: 维度
        b: 基数
        a: 向量 (各分量 ∈ {1, ..., b})

    返回:
        线性索引 (1-based)
    """
    rank = 0
    for i in range(d):
        rank = rank * b + (int(a[i]) - 1)
    return rank + 1  # 1-based


def next_permutation(a):
    """
    生成下一个字典序排列 (Knuth Algorithm L)。

    步骤:
    1. 找最大 i 使得 a[i] < a[i+1]
    2. 找最大 j 使得 a[i] < a[j]
    3. 交换 a[i] 和 a[j]
    4. 翻转 a[i+1:]

    参数:
        a: 排列 (就地修改)

    返回:
        True 如果存在下一个排列, False 如果已到最后一个
    """
    n = len(a)
    a = list(a)

    # Step 1
    i = n - 2
    while i >= 0 and a[i] >= a[i + 1]:
        i -= 1
    if i < 0:
        return False, a

    # Step 2
    j = n - 1
    while a[j] <= a[i]:
        j -= 1

    # Step 3
    a[i], a[j] = a[j], a[i]

    # Step 4
    a[i + 1:] = reversed(a[i + 1:])

    return True, a


def generate_all_permutations(d):
    """
    生成 d 个元素的所有排列。

    参数:
        d: 元素个数

    返回:
        perms: 所有排列的列表, 每个排列是 tuple
    """
    return list(permutations(range(d)))


def symmetrize_coefficients(coefficients, d, b):
    """
    对多项式系数进行对称化。

    输入: 系数数组 c[rank], rank = 0, ..., b^d - 1
    输出: 对称化后的系数 c_sym

    算法:
    1. 对每个等价类代表元 z (非降序排列):
       a. 生成 z 的所有排列 π(z)
       b. 计算多重性 m = |{π(z)}|
       c. 平均: c_sym[π(z)] = (1/m) Σ c[π(z)] 对所有排列
    2. 对所有等价类重复

    映射种子项目:
      - 776_monomial_symmetrize: 轨道平均 + 散射

    参数:
        coefficients: 系数数组 (长度 b^d)
        d: 变量数
        b: 每变量的阶数

    返回:
        symmetrized: 对称化后的系数数组
    """
    total = b**d
    if len(coefficients) != total:
        raise ValueError(f"系数数组长度应为 {total}, 得到 {len(coefficients)}")

    c = np.array(coefficients, dtype=float).copy()
    c_sym = np.zeros(total)
    visited = np.zeros(total, dtype=bool)

    # 枚举所有代表元 (非降序排列)
    for combo in _generate_non_decreasing(d, b):
        # 将组合转换为索引向量 (1-based)
        z = [x + 1 for x in combo]

        # 生成所有排列
        perm_indices = []
        for perm in set(permutations(z)):
            rank = vector_rank(d, b, list(perm)) - 1  # 0-based
            perm_indices.append(rank)

        # 平均
        if len(perm_indices) > 0:
            avg = np.mean([c[pi] for pi in perm_indices])
            for pi in perm_indices:
                c_sym[pi] = avg
                visited[pi] = True

    return c_sym


def _generate_non_decreasing(d, b):
    """
    生成所有非降序 d 元组, 每个分量 ∈ {0, ..., b-1}。

    这些是 S_d 轨道的代表元。
    """
    if d == 0:
        yield ()
        return
    if d == 1:
        for v in range(b):
            yield (v,)
        return

    for first in range(b):
        for rest in _generate_non_decreasing(d - 1, b):
            if len(rest) == 0 or rest[0] >= first:
                yield (first,) + rest
            else:
                break


def elementary_symmetric_polynomials(x):
    """
    计算初等对称多项式 e_k(x₁, ..., x_n)。

    e₀ = 1
    e₁ = x₁ + x₂ + ... + x_n
    e₂ = Σ_{i<j} x_i x_j
    e_k = Σ_{i₁<...<i_k} x_{i₁} ... x_{i_k}
    e_n = x₁ x₂ ... x_n

    使用递推 (Vieta 公式):
      e_k(x₁,...,x_n) = e_k(x₁,...,x_{n-1}) + x_n e_{k-1}(x₁,...,x_{n-1})

    参数:
        x: 变量数组

    返回:
        e: 数组 [e₀, e₁, ..., e_n]
    """
    n = len(x)
    e = np.zeros(n + 1)
    e[0] = 1.0

    for xi in x:
        # 反向更新以避免使用未更新的值
        for k in range(n, 0, -1):
            e[k] = e[k] + xi * e[k - 1]

    return e


def power_sums(x, max_order=None):
    """
    计算幂和 p_k = Σ x_i^k。

    参数:
        x: 变量数组
        max_order: 最大阶数 (默认 len(x))

    返回:
        p: 数组 [p₁, p₂, ..., p_max_order]
    """
    n = len(x)
    if max_order is None:
        max_order = n

    p = np.zeros(max_order)
    for k in range(max_order):
        p[k] = np.sum(np.array(x)**(k + 1))

    return p


def newton_identities_to_elementary(power_sum_array):
    """
    使用 Newton 恒等式从幂和计算初等对称多项式。

    k e_k = Σ_{j=1}^{k} (-1)^{j-1} e_{k-j} p_j

    参数:
        power_sum_array: [p₁, p₂, ..., p_n]

    返回:
        e: [e₀=1, e₁, e₂, ..., e_n]
    """
    n = len(power_sum_array)
    e = np.zeros(n + 1)
    e[0] = 1.0

    for k in range(1, n + 1):
        s = 0.0
        for j in range(1, k + 1):
            sign = (-1.0)**(j - 1)
            s += sign * e[k - j] * power_sum_array[j - 1]
        e[k] = s / k

    return e


def ttbar_event_symmetrize(jet_energies, lepton_energies=None):
    """
    对 tt̄ 事件的可观测量进行对称化。

    对于两个 b-jet:
      - 对称可观测量: E_b1 + E_b2, E_b1 × E_b2
      - 反对称可观测量: |E_b1 - E_b2|

    映射种子项目:
      - 776_monomial_symmetrize: 置换对称化

    参数:
        jet_energies: [E_b1, E_b2] b-jet 能量
        lepton_energies: [E_l1, E_l2] 轻子能量 (可选)

    返回:
        symmetric_obs: 对称化可观测量字典
    """
    result = {}

    # b-jet 对称可观测量
    e = elementary_symmetric_polynomials(jet_energies)
    result['sum_Eb'] = e[1]  # e₁ = E_b1 + E_b2
    result['prod_Eb'] = e[2] if len(e) > 2 else jet_energies[0] * jet_energies[1]
    result['diff_Eb'] = abs(jet_energies[0] - jet_energies[1])
    result['max_Eb'] = max(jet_energies)
    result['min_Eb'] = min(jet_energies)

    # 幂和
    ps = power_sums(jet_energies)
    result['p1_Eb'] = ps[0]  # = sum
    result['p2_Eb'] = ps[1] if len(ps) > 1 else jet_energies[0]**2

    if lepton_energies is not None and len(lepton_energies) >= 2:
        e_l = elementary_symmetric_polynomials(lepton_energies)
        result['sum_El'] = e_l[1]
        result['prod_El'] = e_l[2] if len(e_l) > 2 else lepton_energies[0] * lepton_energies[1]
        result['diff_El'] = abs(lepton_energies[0] - lepton_energies[1])

    return result


def invariant_mass_symmetric(particles_4vec, pair_indices):
    """
    对所有不可区分粒子对计算对称 ne变质量。

    对于 n 个不可区分粒子, ne变质量:
      M_{ij}² = (p_i + p_j)² = m_i² + m_j² + 2(E_i E_j - p⃗_i·p⃗_j)

    对称化: 取所有配对的最小/最大/平均值

    参数:
        particles_4vec: (N, 4) 四动量 [E, px, py, pz]
        pair_indices: 要配对的粒子索引对列表

    返回:
        inv_masses: ne变质量数组
    """
    inv_masses = []
    for (i, j) in pair_indices:
        pi = particles_4vec[i]
        pj = particles_4vec[j]
        # M² = (E_i + E_j)² - (px_i + px_j)² - (py_i + py_j)² - (pz_i + pz_j)²
        M2 = ((pi[0] + pj[0])**2 - (pi[1] + pj[1])**2 -
              (pi[2] + pj[2])**2 - (pi[3] + pj[3])**2)
        inv_masses.append(np.sqrt(max(M2, 0.0)))

    return np.array(inv_masses)
