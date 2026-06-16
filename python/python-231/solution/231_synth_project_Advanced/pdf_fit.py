"""
pdf_fit.py — PDF 全局 χ² 拟合引擎
=============================================================
本模块实现 PDF 参数的全局最小化:
    min_{p} χ²(p) = Σ_i [(d_i − t_i(p))² / σ_i²]

使用:
  (1) 梯度下降 + Armijo 线搜索;
  (2) 有限差分梯度;
  (3) 参数约束投影 (映射自 boundary_word_hexagon 的合法性判定);
  (4) 单参数翻转敏感度分析 (映射自 SoleFlip);
  (5) 稀疏攻击恢复 (映射自 Secure-Data-Reconstruction).

核心公式 (χ² 函数):
    χ²(p) = Σ_{i=1}^{N_data} [(d_i − t_i(p))² / σ_i²]
    + λ_M (∫ x Σ dx − 1)²
    + λ_Vu (∫ u_v dx − 2)²
    + λ_Vd (∫ d_v dx − 1)²

核心公式 (有限差分梯度):
    ∂χ²/∂p_k ≈ [χ²(p + h e_k) − χ²(p − h e_k)] / (2h)

核心公式 (Armijo 线搜索):
    找到最小整数 m ≥ 0 使:
    χ²(p − 2^{-m} α ∇χ²) ≤ χ²(p) − c × 2^{-m} α ||∇χ²||²

核心公式 (单参数翻转敏感度, 映射自 SoleFlip):
    翻转参数 p_k 的一个"比特": p_k → p_k (1 ± ε)
    敏感度: S_k = |χ²(p_flipped) − χ²(p)| / χ²(p)

核心公式 (稀疏攻击恢复, 映射自 H∞ control):
    min ||w − Hg||₁  subject to ||g||₀ ≤ k
    识别异常数据点并恢复.
"""
from __future__ import annotations
import math
from typing import Dict, List, Tuple, Optional

from phys_consts import (
    EPS_NUMERICAL, CHI2_CONVERGENCE_TOL
)
from pdf_param import (
    N_PARAMS, default_params, params_to_vector, vector_to_params,
    evaluate_all_flavors, f2_dis, dy_cross_section,
    momentum_sum_rule, valence_number_rule, normalize_parameters,
    constraint_word_is_legal, single_param_flip_sensitivity
)
from experimental_data import DataSet


# ============================================================
# 1. χ² 函数计算
# ============================================================
def compute_theory_value(x: float, q2: float, type_id: str,
                         params: Dict[str, float]) -> float:
    """计算单个数据点的理论值"""
    try:
        if type_id == 'DIS':
            val = f2_dis(x, q2, params)
        elif type_id == 'DY':
            # DY: q2 字段存储 M², 使用固定 s
            val = dy_cross_section(x, 0.05, q2, 1.96e4, params)
        else:
            val = 0.0
        # clamp to prevent overflow
        return max(-1e10, min(1e10, val))
    except (OverflowError, ValueError):
        return 0.0


def compute_chi_squared(params: Dict[str, float], data: DataSet,
                        lambda_m: float = 100.0,
                        lambda_v: float = 50.0,
                        x_grid: Optional[List[float]] = None
                        ) -> float:
    """
    完整的 χ² 函数:
        χ² = χ²_data + λ_M × (momentum − 1)²
            + λ_Vu × (N_u − 2)² + λ_Vd × (N_d − 1)²

    参数:
        params:    PDF 参数
        data:      实验数据集
        lambda_m:  动量约束权重
        lambda_v:  价夸克约束权重
        x_grid:    求和规则积分网格 (若为 None 则使用 20 点对数网格)
    """
    # 数据项
    chi2_data = 0.0
    for pt in data.points:
        try:
            theory = compute_theory_value(pt.x, pt.q2, pt.type_id, params)
            if pt.uncertainty > EPS_NUMERICAL:
                residual = (pt.value - theory) / pt.uncertainty
                residual = max(-1e6, min(1e6, residual))
                chi2_data += residual * residual
        except (OverflowError, ValueError):
            chi2_data += 1e10
        if chi2_data > 1e20:
            break

    # 约束项
    if x_grid is None:
        x_grid = [math.exp(math.log(1e-3) + i * (math.log(0.99) - math.log(1e-3)) / 19)
                  for i in range(20)]
    try:
        mom = momentum_sum_rule(x_grid, params)
        n_u = valence_number_rule(x_grid, params, 'u_v')
        n_d = valence_number_rule(x_grid, params, 'd_v')
        # 防止溢出: clamp residuals
        mom_res = max(-1e6, min(1e6, mom - 1.0))
        vu_res = max(-1e6, min(1e6, n_u - 2.0))
        vd_res = max(-1e6, min(1e6, n_d - 1.0))
        chi2_constraint = (lambda_m * mom_res * mom_res
                           + lambda_v * vu_res * vu_res
                           + lambda_v * vd_res * vd_res)
    except (OverflowError, ValueError):
        chi2_constraint = 1e20
    # 防止 chi2_data 也溢出
    chi2_data = min(chi2_data, 1e20)
    return min(chi2_data + chi2_constraint, 1e22)


def compute_chi2_data_only(params: Dict[str, float], data: DataSet) -> float:
    """仅计算数据 χ² (不含约束)"""
    chi2 = 0.0
    for pt in data.points:
        try:
            theory = compute_theory_value(pt.x, pt.q2, pt.type_id, params)
            if pt.uncertainty > EPS_NUMERICAL:
                residual = (pt.value - theory) / pt.uncertainty
                residual = max(-1e6, min(1e6, residual))
                chi2 += residual * residual
        except (OverflowError, ValueError):
            chi2 += 1e10
        if chi2 > 1e20:
            break
    return min(chi2, 1e20)


# ============================================================
# 2. 梯度计算 (有限差分)
# ============================================================
def compute_gradient(params: Dict[str, float], data: DataSet,
                     h: float = 1e-4, x_grid: Optional[List[float]] = None
                     ) -> List[float]:
    """
    χ² 对参数的有限差分梯度:
        ∂χ²/∂p_k ≈ [χ²(p + h e_k) − χ²(p − h e_k)] / (2h)

    使用中心差分, O(h²) 精度.
    """
    grad = []
    keys = list(params.keys())
    for k in keys:
        p_plus = dict(params)
        p_minus = dict(params)
        delta = max(abs(params[k]) * h, h)
        p_plus[k] = params[k] + delta
        p_minus[k] = params[k] - delta
        # 投影到合法范围
        p_plus = _project_params(p_plus)
        p_minus = _project_params(p_minus)
        try:
            chi2_plus = compute_chi_squared(p_plus, data, x_grid=x_grid)
            chi2_minus = compute_chi_squared(p_minus, data, x_grid=x_grid)
            g = (chi2_plus - chi2_minus) / (2.0 * delta)
            # 梯度截断防止过大
            g = max(-1e10, min(1e10, g))
        except (OverflowError, ValueError):
            g = 0.0
        grad.append(g)
    return grad


def gradient_norm(grad: List[float]) -> float:
    """梯度 L2 范数"""
    return math.sqrt(sum(g * g for g in grad))


# ============================================================
# 3. 梯度下降拟合 (含 Armijo 线搜索)
# ============================================================
def armijo_line_search(
    params: Dict[str, float], grad: List[float], data: DataSet,
    alpha_init: float = 0.1, c: float = 1e-4, rho: float = 0.5,
    max_iter: int = 20, x_grid: Optional[List[float]] = None
) -> float:
    """
    Armijo 线搜索:
        找到最小 m 使:
        χ²(p − 2^m α ∇χ²) ≤ χ²(p) − c × 2^m α ||∇χ²||²

    参数:
        params:     当前参数
        grad:       当前梯度
        data:       数据集
        alpha_init: 初始步长
        c:          Armijo 常数 (充分下降参数)
        rho:        收缩因子
        max_iter:   最大搜索次数
        x_grid:     求和规则网格
    返回:
        满足 Armijo 条件的步长
    """
    chi2_current = compute_chi_squared(params, data, x_grid=x_grid)
    grad_sq = sum(g * g for g in grad)
    alpha = alpha_init
    for _ in range(max_iter):
        # 试探步
        p_trial = dict(params)
        keys = list(params.keys())
        for i, k in enumerate(keys):
            p_trial[k] = params[k] - alpha * grad[i]
        chi2_trial = compute_chi_squared(p_trial, data, x_grid=x_grid)
        # Armijo 条件
        if chi2_trial <= chi2_current - c * alpha * grad_sq:
            return alpha
        alpha *= rho
    return alpha


def fit_pdf_params(
    data: DataSet,
    init_params: Optional[Dict[str, float]] = None,
    max_iter: int = 50,
    tol: float = CHI2_CONVERGENCE_TOL,
    lr: float = 0.05,
    x_grid: Optional[List[float]] = None,
    verbose: bool = False
) -> Tuple[Dict[str, float], List[float], int]:
    """
    PDF 参数全局拟合:
        min_p χ²(p) = Σ [(d_i − t_i(p))/σ_i]² + constraints

    算法: 梯度下降 + Armijo 线搜索 + 约束投影

    参数:
        data:        实验数据集
        init_params: 初始参数猜测 (若为 None 则使用默认)
        max_iter:    最大迭代次数
        tol:         收敛容差 (||∇χ²|| < tol)
        lr:          学习率
        x_grid:      求和规则积分网格
        verbose:     打印进度
    返回:
        (best_params, chi2_history, n_iter)
    """
    if init_params is None:
        init_params = default_params()
    params = dict(init_params)

    # 初始归一化
    if x_grid is None:
        x_grid = [math.exp(math.log(1e-3) + i * (math.log(0.99) - math.log(1e-3)) / 19)
                  for i in range(20)]
    params = normalize_parameters(params, x_grid)

    chi2_history = []
    best_chi2 = float('inf')
    best_params = dict(params)

    for iteration in range(max_iter):
        # 计算梯度
        grad = compute_gradient(params, data, x_grid=x_grid)
        # 梯度裁剪防止发散
        gn = gradient_norm(grad)
        max_grad_norm = 1e6
        if gn > max_grad_norm:
            scale = max_grad_norm / gn
            grad = [g * scale for g in grad]
            gn = max_grad_norm
        chi2 = compute_chi_squared(params, data, x_grid=x_grid)
        chi2_history.append(chi2)

        if verbose:
            chi2_data = compute_chi2_data_only(params, data)
            print(f"  Iter {iteration:3d}: χ² = {chi2:.4f} "
                  f"(data: {chi2_data:.4f}), ||∇χ²|| = {gn:.6f}")

        # 记录最优
        if chi2 < best_chi2:
            best_chi2 = chi2
            best_params = dict(params)

        # 收敛检查
        if gn < tol:
            if verbose:
                print(f"  收敛: ||∇χ²|| = {gn:.2e} < {tol:.2e}")
            break

        # Armijo 线搜索
        alpha = armijo_line_search(params, grad, data, alpha_init=lr,
                                   x_grid=x_grid)
        # 更新参数
        keys = list(params.keys())
        for i, k in enumerate(keys):
            params[k] = params[k] - alpha * grad[i]

        # 约束投影: 确保参数在物理范围内
        params = _project_params(params)

        # 周期性归一化 (每 5 步)
        if iteration % 5 == 4:
            params = normalize_parameters(params, x_grid)

    return best_params, chi2_history, len(chi2_history)


def _project_params(p: Dict[str, float]) -> Dict[str, float]:
    """
    参数约束投影:
      - A_* > 0 (归一化必须为正)
      - a_* ∈ [−0.5, 1.5] (小-x 幂次, 收紧以避免溢出)
      - b_* ∈ [1.0, 12] (大-x 幂次)
      - c_*, d_* ∈ [−2, 2] (修正系数)

    映射自 boundary_word_hexagon 的合法性约束思想:
      参数空间中的"合法区域"类似 polyhex 网格上的合法边界词.
    """
    p_new = dict(p)
    # 归一化参数范围
    for k in ['A_uv', 'A_dv', 'A_g', 'A_s']:
        p_new[k] = max(0.01, min(p_new[k], 50.0))
    for k in ['a_uv', 'a_dv', 'a_g', 'a_s']:
        p_new[k] = max(-0.5, min(1.5, p_new[k]))
    for k in ['b_uv', 'b_dv', 'b_g', 'b_s']:
        p_new[k] = max(1.0, min(12.0, p_new[k]))
    for k in ['c_uv', 'c_dv', 'c_g']:
        p_new[k] = max(-2.0, min(2.0, p_new[k]))
    if 'd_uv' in p_new:
        p_new['d_uv'] = max(-2.0, min(2.0, p_new['d_uv']))
    return p_new


# ============================================================
# 4. 单参数翻转敏感度 (映射自 SoleFlip)
# ============================================================
def soleflip_sensitivity_analysis(
    params: Dict[str, float], data: DataSet,
    x_grid: Optional[List[float]] = None,
    flip_fraction: float = 0.05
) -> Dict[str, Dict[str, float]]:
    """
    单参数翻转敏感度分析 (映射自 SoleFlip 的"单比特翻转"思想):

    对每个参数 p_k:
      1) 翻转: p_k → p_k × (1 + flip_fraction)
      2) 计算 Δχ²_k = |χ²(p_flipped) − χ²(p)|
      3) 计算约束偏移: Δmomentum, Δvalence
      4) 排序: 找出最敏感的参数 (类比 SoleFlip 的"脆弱神经元")

    返回:
        Dict[param_key] = {
            'delta_chi2': χ² 变化,
            'momentum_shift': 动量约束偏移,
            'valence_shift': 价夸克约束偏移,
            'sensitivity_rank': 敏感度排名
        }
    """
    if x_grid is None:
        x_grid = [math.exp(math.log(1e-3) + i * (math.log(0.99) - math.log(1e-3)) / 19)
                  for i in range(20)]

    chi2_base = compute_chi_squared(params, data, x_grid=x_grid)
    results = {}

    for k in params.keys():
        p_flip = dict(params)
        p_flip[k] = params[k] * (1.0 + flip_fraction)
        chi2_flip = compute_chi_squared(p_flip, data, x_grid=x_grid)
        delta_chi2 = abs(chi2_flip - chi2_base)
        # 物理约束偏移
        constraint_sens = single_param_flip_sensitivity(
            params, x_grid, k, flip_fraction)
        results[k] = {
            'delta_chi2': delta_chi2,
            'momentum_shift': abs(constraint_sens['momentum_shift']),
            'valence_uv_shift': abs(constraint_sens['valence_uv_shift']),
            'valence_dv_shift': abs(constraint_sens['valence_dv_shift']),
        }

    # 排序
    sorted_keys = sorted(results.keys(),
                         key=lambda k: results[k]['delta_chi2'], reverse=True)
    for rank, k in enumerate(sorted_keys):
        results[k]['sensitivity_rank'] = rank + 1

    return results


# ============================================================
# 5. 稀疏异常恢复 (映射自 Secure-Data-Reconstruction)
# ============================================================
def sparse_outlier_detection(
    params: Dict[str, float], data: DataSet,
    threshold_sigma: float = 3.0
) -> Tuple[List[int], List[float]]:
    """
    稀疏异常数据点检测与恢复 (映射自 H∞ 控制的稀疏攻击恢复):

    核心思想:
      残差向量 r_i = (d_i − t_i) / σ_i
      若 |r_i| > threshold_sigma, 则 d_i 为异常值.
      使用 L1 范数最小化识别稀疏异常:
        min ||r − Hg||₁  s.t. ||g||₀ ≤ k
      简化: 直接按 |r_i| 排序, 取 top-k 为异常.

    返回:
        (outlier_indices, outlier_residuals)
    """
    residuals = []
    for i, pt in enumerate(data.points):
        theory = compute_theory_value(pt.x, pt.q2, pt.type_id, params)
        if pt.uncertainty > EPS_NUMERICAL:
            r = (pt.value - theory) / pt.uncertainty
        else:
            r = 0.0
        residuals.append((i, abs(r)))

    # 排序找异常
    residuals.sort(key=lambda x: x[1], reverse=True)
    outlier_indices = [idx for idx, r in residuals if r > threshold_sigma]
    outlier_residuals = [r for _, r in residuals if r > threshold_sigma]
    return outlier_indices, outlier_residuals


def robust_chi_squared(
    params: Dict[str, float], data: DataSet,
    huber_delta: float = 3.0
) -> float:
    """
    鲁棒 χ² (Huber 损失):
        ρ(r) = r²/2                  if |r| ≤ δ
             = δ(|r| − δ/2)         if |r| > δ

    对异常值不敏感, 映射自 H∞ 控制的鲁棒思想.
    """
    chi2 = 0.0
    for pt in data.points:
        theory = compute_theory_value(pt.x, pt.q2, pt.type_id, params)
        if pt.uncertainty > EPS_NUMERICAL:
            r = abs((pt.value - theory) / pt.uncertainty)
        else:
            continue
        if r <= huber_delta:
            chi2 += 0.5 * r * r
        else:
            chi2 += huber_delta * (r - 0.5 * huber_delta)
    return chi2
