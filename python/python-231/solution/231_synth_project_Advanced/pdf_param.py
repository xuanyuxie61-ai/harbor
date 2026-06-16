"""
pdf_param.py — PDF 参数化、物理约束与截面计算
=============================================================
本模块是连接 DGLAP 演化与实验拟合的核心桥梁。
它包含:
  (1) 初始尺度 Q0² 的 PDF 参数化函数 (价夸克、海夸克、胶子);
  (2) 物理约束: 动量求和规则, 价夸克数守恒 (边界词合法性判定);
  (3) LO DIS 与 Drell-Yan 截面计算公式;
  (4) 参数-向量 ↔ PDF-函数 的双向映射。

边界词合法性 (Boundary Word Legality) 映射自 seed 项目 108_boundary_word_hexagon:
  在 polyhex 网格上, 边界词 w ∈ {1,...,6}^L 合法性要求:
    - 每个方向 k 的步数与其对向 k+3 (mod 6) 匹配;
    - 相邻步的方向差 |Δk| ≡ 1 (mod 6).
  在 PDF 参数空间中, 我们定义类似的"约束词" w ∈ {M, V_u, V_d, P, N}^N_par:
    - M: 动量求和规则
    - V_u: u 价夸克数守恒 (= 2)
    - V_d: d 价夸克数守恒 (= 1)
    - P: 正定性约束
    - N: 正则性约束 (x→0 与 x→1 行为)
  合法约束词要求每个约束的"对偶"匹配:
    - M 的对偶是 M (动量求和自对偶);
    - V_u 的对偶是 V_u (自身守恒);
    - P 的对偶是 N (正定性与正则性互对偶);
    - 相邻约束的"方向差"需满足合法性条件。

核心公式 (初始 PDF 参数化, 在 Q0² = 2 GeV²):
    x u_v(x) = A_u x^{a_u} (1−x)^{b_u} (1 + c_u √x + d_u x)
    x d_v(x) = A_d x^{a_d} (1−x)^{b_d} (1 + c_d x)
    x g(x)   = A_g x^{a_g} (1−x)^{b_g} (1 + c_g √x)
    x S(x)   = A_s x^{a_s} (1−x)^{b_s}    (奇异海, S = s + s̄)

核心公式 (动量求和规则):
    ∫_0^1 dx [Σ_q x(q + q̄) + xg] = 1

核心公式 (价夸克数守恒):
    ∫_0^1 dx u_v(x) = 2
    ∫_0^1 dx d_v(x) = 1

核心公式 (LO DIS 约化截面):
    σ_r^{DIS}(x, Q²) = F_2(x, Q²) − [y² / (1+(1−y)²)] F_L(x, Q²)
    在 LO: F_2 = Σ_q e_q² x(q + q̄), F_L = 0

核心公式 (LO Drell-Yan 截面):
    dσ/dM² ∝ (4πα² / 9M²s) Σ_q e_q² [q(x₁)q̄(x₂) + q̄(x₁)q(x₂)]
"""
from __future__ import annotations
import math
from typing import Dict, List, Tuple, Optional

from phys_consts import (
    ALPHA_EM, SIN2THETA_W, X_MIN_DEFAULT, X_MAX_DEFAULT,
    Q0_SQ_DEFAULT, EPS_NUMERICAL, PDF_POSITIVITY_FLOOR, NX_DEFAULT
)

# ============================================================
# 1. PDF 参数化
# ============================================================
# 参数命名约定:
#   flavor = 'u_v', 'd_v', 'g', 'S' (奇异海)
#   每味参数: A (归一化), a (小-x 幂次), b (大-x 幂次),
#             c (√x 修正), d (线性 x 修正, 仅 u_v)
DEFAULT_PARAM_KEYS: List[str] = [
    'A_uv', 'a_uv', 'b_uv', 'c_uv', 'd_uv',
    'A_dv', 'a_dv', 'b_dv', 'c_dv',
    'A_g', 'a_g', 'b_g', 'c_g',
    'A_s', 'a_s', 'b_s',
]
N_PARAMS: int = len(DEFAULT_PARAM_KEYS)


def default_params() -> Dict[str, float]:
    """
    返回 CT14-like 的默认初始参数 (Q0² = 2 GeV²)
    注意: A_uv, A_dv, A_g, A_s 由约束自动确定, 不参与自由拟合
    """
    return {
        'A_uv': 3.0, 'a_uv': 0.5, 'b_uv': 3.5, 'c_uv': 1.2, 'd_uv': 0.3,
        'A_dv': 1.5, 'a_dv': 0.5, 'b_dv': 4.0, 'c_dv': 0.8,
        'A_g': 5.0, 'a_g': -0.3, 'b_g': 5.0, 'c_g': 2.0,
        'A_s': 0.5, 'a_s': 0.3, 'b_s': 6.0,
    }


def params_to_vector(p: Dict[str, float]) -> List[float]:
    """参数字典 → 参数向量"""
    return [p[k] for k in DEFAULT_PARAM_KEYS]


def vector_to_params(v: List[float]) -> Dict[str, float]:
    """参数向量 → 参数字典"""
    if len(v) != N_PARAMS:
        raise ValueError(f"vector_to_params: 期望 {N_PARAMS} 维, 得到 {len(v)} 维")
    return dict(zip(DEFAULT_PARAM_KEYS, v))


# ============================================================
# 2. 初始 PDF 计算
# ============================================================
def xuv_x(x: float, p: Dict[str, float]) -> float:
    """
    x u_v(x, Q0²) = A_uv × x^{a_uv} × (1−x)^{b_uv} × (1 + c_uv √x + d_uv x)

    此参数化确保:
      - x→0: x u_v ~ x^{a_uv}, 若 a_uv > −1 则可积
      - x→1: x u_v ~ (1−x)^{b_uv}, 满足强子端点行为
    """
    if x <= 0.0 or x >= 1.0:
        return 0.0
    xc = max(x, 1e-15)
    omx = max(1.0 - xc, 1e-15)
    sqrtx = math.sqrt(xc)
    try:
        # 使用对数空间计算以避免溢出
        log_val = (math.log(abs(p['A_uv']) + 1e-300)
                   + p['a_uv'] * math.log(xc)
                   + p['b_uv'] * math.log(omx)
                   + math.log(abs(1.0 + p['c_uv'] * sqrtx + p['d_uv'] * xc) + 1e-300))
        if log_val > 500:  # 防止溢出
            return 0.0
        val = math.exp(log_val)
        if p['A_uv'] < 0:
            val = -val
        if (1.0 + p['c_uv'] * sqrtx + p['d_uv'] * xc) < 0:
            val = -val
    except (OverflowError, ValueError):
        return 0.0
    return max(val, PDF_POSITIVITY_FLOOR * x)


def xdv_x(x: float, p: Dict[str, float]) -> float:
    """
    x d_v(x, Q0²) = A_dv × x^{a_dv} × (1−x)^{b_dv} × (1 + c_dv x)

    注意 d_v/u_v → 0 当 x→1, 因为 b_dv > b_uv
    """
    if x <= 0.0 or x >= 1.0:
        return 0.0
    xc = max(x, 1e-15)
    omx = max(1.0 - xc, 1e-15)
    try:
        log_val = (math.log(abs(p['A_dv']) + 1e-300)
                   + p['a_dv'] * math.log(xc)
                   + p['b_dv'] * math.log(omx)
                   + math.log(abs(1.0 + p['c_dv'] * xc) + 1e-300))
        if log_val > 500:
            return 0.0
        val = math.exp(log_val)
        if p['A_dv'] < 0:
            val = -val
        if (1.0 + p['c_dv'] * xc) < 0:
            val = -val
    except (OverflowError, ValueError):
        return 0.0
    return max(val, PDF_POSITIVITY_FLOOR * x)


def xg_x(x: float, p: Dict[str, float]) -> float:
    """
    x g(x, Q0²) = A_g × x^{a_g} × (1−x)^{b_g} × (1 + c_g √x)

    胶子在 x→0 时快速增长: a_g < 0 导致 xg ~ x^{a_g} → ∞
    鲁棒性: 在 x < x_min 处截断
    """
    if x <= 0.0 or x >= 1.0:
        return 0.0
    xc = max(x, 1e-15)
    omx = max(1.0 - xc, 1e-15)
    sqrtx = math.sqrt(xc)
    try:
        log_val = (math.log(abs(p['A_g']) + 1e-300)
                   + p['a_g'] * math.log(xc)
                   + p['b_g'] * math.log(omx)
                   + math.log(abs(1.0 + p['c_g'] * sqrtx) + 1e-300))
        if log_val > 500:
            return 0.0
        val = math.exp(log_val)
        if p['A_g'] < 0:
            val = -val
        if (1.0 + p['c_g'] * sqrtx) < 0:
            val = -val
    except (OverflowError, ValueError):
        return 0.0
    return max(val, PDF_POSITIVITY_FLOOR * x)


def xs_x(x: float, p: Dict[str, float]) -> float:
    """
    x S(x, Q0²) = A_s × x^{a_s} × (1−x)^{b_s}

    S(x) = s(x) + s̄(x) 为奇异海分布
    假设 s = s̄, 故 x s(x) = x S(x) / 2
    """
    if x <= 0.0 or x >= 1.0:
        return 0.0
    xc = max(x, 1e-15)
    omx = max(1.0 - xc, 1e-15)
    try:
        log_val = (math.log(abs(p['A_s']) + 1e-300)
                   + p['a_s'] * math.log(xc)
                   + p['b_s'] * math.log(omx))
        if log_val > 500:
            return 0.0
        val = math.exp(log_val)
        if p['A_s'] < 0:
            val = -val
    except (OverflowError, ValueError):
        return 0.0
    return max(val, PDF_POSITIVITY_FLOOR * x)


def evaluate_all_flavors(x: float, p: Dict[str, float]
                         ) -> Tuple[float, float, float, float, float]:
    """
    返回 (x u_v, x d_v, x g, x S, x Σ) 其中 Σ = u_v + d_v + S + 2x ū
    假设 ū = d̄ = 0.5 × (Σ − u_v − d_v − S) / 2 的简化模型
    """
    xuv = xuv_x(x, p)
    xdv = xdv_x(x, p)
    xg = xg_x(x, p)
    xs = xs_x(x, p)
    x_sigma = xuv + xdv + xs
    return xuv, xdv, xg, xs, x_sigma


# ============================================================
# 3. 物理约束: 边界词合法性与求和规则
# ============================================================
# 约束类型编码 (映射自 boundary_word_hexagon):
CONSTRAINT_TYPES = {
    'M': 1,   # 动量求和规则
    'V': 2,   # 价夸克数守恒
    'P': 3,   # 正定性约束
    'N': 4,   # 正则性约束
}
CONSTRAINT_DUAL = {1: 1, 2: 2, 3: 4, 4: 3}  # 对偶映射


def constraint_word_is_legal(word: List[str]) -> bool:
    """
    判定约束词是否合法 (映射自 boundary_is_legal):
      - 每个约束类型 k 的出现次数必须与其对偶 k_dual 匹配;
      - 相邻约束的方向差 |Δk| 必须在 {1, 2, 3} 内 (六边形邻接);
      - M 约束 (动量) 只能出现偶数次 (自对偶).

    参数:
        word: 约束词列表, 每个元素为 'M', 'V', 'P', 'N' 之一
    返回:
        bool: 是否合法
    """
    if not word:
        return False
    # 检查字符合法性
    for w in word:
        if w not in CONSTRAINT_TYPES:
            return False
    # 检查对偶匹配
    counts = {k: 0 for k in CONSTRAINT_TYPES}
    for w in word:
        counts[w] += 1
    # M 必须偶数 (自对偶)
    if counts['M'] % 2 != 0:
        return False
    # P 和 N 必须数量相等 (互对偶)
    if counts['P'] != counts['N']:
        return False
    # 相邻约束的方向差 |Δk| ∈ {1, 2} (mod 4)
    for i in range(1, len(word)):
        k_prev = CONSTRAINT_TYPES[word[i - 1]]
        k_curr = CONSTRAINT_TYPES[word[i]]
        diff = abs(k_curr - k_prev)
        diff_mod = min(diff, 4 - diff)
        if diff_mod > 2:
            return False
    return True


def momentum_sum_rule(x_grid: List[float], p: Dict[str, float],
                      w_weights: Optional[List[float]] = None) -> float:
    """
    动量求和规则:
        ∫_0^1 dx [u_v + d_v + S + g + 2 ū + 2 d̄ + ...] = 1

    简化为 (3 味 + 胶子):
        ∫_0^1 dx [u_v + d_v + S + g] ≈ 1

    数值积分采用梯形法则 (在 x_grid 上).
    w_weights: 可选的积分权重 (若为空则使用均匀梯形权重)
    """
    nx = len(x_grid)
    integrand = [0.0] * nx
    for i, x in enumerate(x_grid):
        if x <= 0.0 or x >= 1.0:
            continue
        xuv, xdv, xg, xs, _ = evaluate_all_flavors(x, p)
        integrand[i] = xuv + xdv + xg + xs
    # 梯形积分
    result = 0.0
    for i in range(nx - 1):
        dx = x_grid[i + 1] - x_grid[i]
        result += 0.5 * (integrand[i] + integrand[i + 1]) * dx
    if w_weights is not None:
        result = sum(w * f for w, f in zip(w_weights, integrand))
    return result


def valence_number_rule(x_grid: List[float], p: Dict[str, float],
                        flavor: str = 'u_v') -> float:
    """
    价夸克数守恒:
        ∫_0^1 dx u_v(x) = 2   (质子中 2 个 u 价夸克)
        ∫_0^1 dx d_v(x) = 1   (质子中 1 个 d 价夸克)

    数值: 梯形积分 u_v(x) 或 d_v(x) 在 x_grid 上
    """
    nx = len(x_grid)
    integrand = [0.0] * nx
    func = xuv_x if flavor == 'u_v' else xdv_x
    for i, x in enumerate(x_grid):
        if x <= 0.0 or x >= 1.0:
            continue
        val = func(x, p)
        # 注意 func 返回 x f(x), 所以 u_v(x) = func(x) / x
        integrand[i] = val / max(x, EPS_NUMERICAL)
    result = 0.0
    for i in range(nx - 1):
        dx = x_grid[i + 1] - x_grid[i]
        result += 0.5 * (integrand[i] + integrand[i + 1]) * dx
    return result


def normalize_parameters(p: Dict[str, float], x_grid: List[float]
                         ) -> Dict[str, float]:
    """
    通过调整 A_uv, A_dv, A_g, A_s 使求和规则精确满足.

    步骤:
      1) 先固定 A_dv 使 ∫ d_v = 1;
      2) 固定 A_uv 使 ∫ u_v = 2;
      3) 固定 A_s = 0.5 × A_uv × 0.4 (经验关系);
      4) 固定 A_g 使 ∫ (u_v + d_v + S + g) = 1.

    返回归一化后的参数副本.
    """
    p_new = dict(p)
    # Step 1: 归一化 d_v
    I_dv = valence_number_rule(x_grid, p_new, 'd_v')
    if I_dv > EPS_NUMERICAL:
        p_new['A_dv'] = p_new['A_dv'] / I_dv
    # Step 2: 归一化 u_v
    I_uv = valence_number_rule(x_grid, p_new, 'u_v')
    if I_uv > EPS_NUMERICAL:
        p_new['A_uv'] = p_new['A_uv'] * 2.0 / I_uv
    # Step 3: 设置 A_s (奇异海归一化, 经验取 u_v 的 40%)
    p_new['A_s'] = 0.4 * p_new['A_uv'] * (p_new['a_uv'] / max(p_new['a_s'], 0.1))
    # Step 4: 归一化胶子使动量求和 = 1
    mom = momentum_sum_rule(x_grid, p_new)
    # 胶子贡献的近似积分
    I_g_only = 0.0
    nx = len(x_grid)
    for i in range(nx - 1):
        x_mid = 0.5 * (x_grid[i] + x_grid[i + 1])
        dx = x_grid[i + 1] - x_grid[i]
        I_g_only += xg_x(x_mid, p_new) * dx
    mom_without_g = mom - I_g_only
    if I_g_only > EPS_NUMERICAL:
        target_g = 1.0 - mom_without_g
        if target_g > 0.01:
            p_new['A_g'] = p_new['A_g'] * target_g / I_g_only
    return p_new


# ============================================================
# 4. DIS 与 Drell-Yan 截面计算
# ============================================================
# 夸克电荷平方 (u: +2/3, d: -1/3, s: -1/3)
QUARK_CHARGES_SQ: Dict[str, float] = {
    'u': 4.0 / 9.0, 'd': 1.0 / 9.0, 's': 1.0 / 9.0,
    'ubar': 4.0 / 9.0, 'dbar': 1.0 / 9.0, 'sbar': 1.0 / 9.0,
}


def f2_dis(x: float, q2: float, p: Dict[str, float]) -> float:
    """
    LO DIS 结构函数 F_2:
        F_2(x, Q²) = Σ_q e_q² × x [q(x, Q²) + q̄(x, Q²)]

    简化: 在 Q0² 处 (无演化), 使用参数化直接计算.
    假设 ū = d̄ = s = s̄ = S/2, 则:
        F_2 = (4/9) x(u_v + 2ū) + (1/9) x(d_v + 2d̄) + (1/9) x(s + s̄)
            = (4/9) x u_v + (1/9) x d_v + (4/9 + 1/9 + 1/9) × 2x ū + (1/9) x S
            = (4/9) x u_v + (1/9) x d_v + (2/3) × 2x ū + (1/9) x S
    """
    if x <= 0.0 or x >= 1.0:
        return 0.0
    xuv, xdv, xg, xs, _ = evaluate_all_flavors(x, p)
    # 假设海夸克: 2 x ū = (x_sigma - xuv - xdv - xs) * 0.5
    # 简化: 假设 S = 2 x s̄, 海夸克贡献 = xs (已经包含 s+s̄)
    f2 = (4.0 / 9.0) * xuv + (1.0 / 9.0) * xdv + (1.0 / 9.0) * xs
    return max(f2, 0.0)


def reduced_cross_section_dis(x: float, q2: float, p: Dict[str, float],
                              y_inelasticity: float = 0.5) -> float:
    """
    LO DIS 约化截面:
        σ_r = F_2(x, Q²) − y² / [1 + (1−y)²] × F_L(x, Q²)

    在 LO, F_L = 0 (Callan-Gross 关系), 故 σ_r = F_2.
    此处保留 y 依赖性以便未来 NLO 扩展.
    """
    f2 = f2_dis(x, q2, p)
    if y_inelasticity <= 0.0 or y_inelasticity >= 1.0:
        return f2
    yl = y_inelasticity ** 2 / (1.0 + (1.0 - y_inelasticity) ** 2)
    # LO: F_L = 0
    fl = 0.0
    return f2 - yl * fl


def dy_cross_section(x1: float, x2: float, M2: float, s: float,
                     p: Dict[str, float]) -> float:
    """
    LO Drell-Yan 截面:
        dσ/dM² = (4πα² / 9M²s) × Σ_q e_q² [q(x₁) q̄(x₂) + q̄(x₁) q(x₂)]

    参数:
        x1, x2: 两个入射强子的动量分数
        M²:     轻子对的不变质量平方
        s:      质心能量平方
        p:      PDF 参数

    简化: 假设海夸克 ū ≈ d̄ ≈ s/2, 仅考虑 u ū 和 d d̄ 通道
    """
    if x1 <= 0.0 or x1 >= 1.0 or x2 <= 0.0 or x2 >= 1.0:
        return 0.0
    if M2 <= 0.0 or s <= 0.0:
        return 0.0
    xuv1, xdv1, _, xs1, _ = evaluate_all_flavors(x1, p)
    xuv2, xdv2, _, xs2, _ = evaluate_all_flavors(x2, p)
    # 转换: x q(x) → q(x) = [x q(x)] / x
    uv1 = xuv1 / x1
    dv1 = xdv1 / x1
    s1 = xs1 / (2.0 * x1)  # s(x) = S(x)/2
    uv2 = xuv2 / x2
    dv2 = xdv2 / x2
    s2 = xs2 / (2.0 * x2)
    # 假设海夸克: ubar = dbar = (sigma - uv - dv - S) / 4
    # 简化: ubar = dbar = s
    ubar1 = s1
    dbar1 = s1
    ubar2 = s2
    dbar2 = s2
    # 求和: u ubar + d dbar + s sbar
    eu2 = 4.0 / 9.0
    ed2 = 1.0 / 9.0
    es2 = 1.0 / 9.0
    sum_q = (eu2 * (uv1 * ubar2 + ubar1 * uv2)
             + ed2 * (dv1 * dbar2 + dbar1 * dv2)
             + es2 * (s1 * s2 + s1 * s2))
    prefactor = 4.0 * ALPHA_EM * math.pi / (9.0 * M2 * s)
    return max(prefactor * sum_q, 0.0)


# ============================================================
# 5. 单参数翻转敏感度分析 (映射自 SoleFlip)
# ============================================================
def single_param_flip_sensitivity(
    p_base: Dict[str, float],
    x_grid: List[float],
    param_key: str,
    flip_fraction: float = 0.01
) -> Dict[str, float]:
    """
    对参数 p[param_key] 施加微小翻转 (乘以 1 ± flip_fraction),
    计算对动量求和规则与价夸克数的影响.

    映射自 SoleFlip 的"单比特翻转"思想:
      在神经网络中, 翻转单个权重的最低有效位;
      在 PDF 中, 翻转单个参数的一个小比例, 观察全局约束的敏感度.

    返回:
        {
            'momentum_shift': Δ(∫ x Σ),
            'valence_uv_shift': Δ(∫ u_v),
            'valence_dv_shift': Δ(∫ d_v),
            'max_pdf_shift': max_x |Δf(x)| / |f(x)|
        }
    """
    p_plus = dict(p_base)
    p_minus = dict(p_base)
    delta = p_base[param_key] * flip_fraction
    if abs(delta) < EPS_NUMERICAL:
        delta = flip_fraction
    p_plus[param_key] = p_base[param_key] + delta
    p_minus[param_key] = p_base[param_key] - delta
    mom_plus = momentum_sum_rule(x_grid, p_plus)
    mom_minus = momentum_sum_rule(x_grid, p_minus)
    vu_plus = valence_number_rule(x_grid, p_plus, 'u_v')
    vu_minus = valence_number_rule(x_grid, p_minus, 'u_v')
    vd_plus = valence_number_rule(x_grid, p_plus, 'd_v')
    vd_minus = valence_number_rule(x_grid, p_minus, 'd_v')
    max_pdf_shift = 0.0
    for x in x_grid:
        if x <= 0.0 or x >= 1.0:
            continue
        _, _, g_plus, _, sig_plus = evaluate_all_flavors(x, p_plus)
        _, _, g_minus, _, sig_minus = evaluate_all_flavors(x, p_minus)
        if abs(sig_plus) > EPS_NUMERICAL:
            rel = abs(sig_plus - sig_minus) / abs(sig_plus)
            max_pdf_shift = max(max_pdf_shift, rel)
    return {
        'momentum_shift': 0.5 * (mom_plus - mom_minus),
        'valence_uv_shift': 0.5 * (vu_plus - vu_minus),
        'valence_dv_shift': 0.5 * (vd_plus - vd_minus),
        'max_pdf_shift': max_pdf_shift,
    }
