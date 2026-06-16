"""
experimental_data.py — 实验数据生成与蒙特卡洛不确定性 (映射自 craps_simulation)
=============================================================
本模块为 PDF 全局拟合生成合成实验数据, 并包含蒙特卡洛方法
用于不确定性传播. 映射自 craps_simulation 的蒙特卡洛概率模拟思想.

包含:
  (1) DIS 数据生成 (F_2 测量, 含统计与系统误差);
  (2) Drell-Yan 数据生成 (截面测量);
  (3) 蒙特卡洛采样与概率估计;
  (4) 数据相关矩阵与协方差矩阵.

核心公式 (DIS 数据模拟):
    F_2^{data}(x_i, Q²_i) = F_2^{true}(x_i, Q²_i) × (1 + δ_stat × N(0,1))
                                            × (1 + δ_syst × N(0,1))
    σ_i² = (δ_stat × F_2^{true})² + (δ_syst × F_2^{true})²

核心公式 (Drell-Yan 数据模拟):
    (dσ/dM²)^{data} = (dσ/dM²)^{true} × (1 + δ × N(0,1))

核心公式 (蒙特卡洛估计, 映射自 craps_probability):
    p ≈ N_win / N_total
    σ_p = √[p(1−p)/N_total]
"""
from __future__ import annotations
import math
import random
from typing import Dict, List, Tuple, Optional

from phys_consts import EPS_NUMERICAL, Q0_SQ_DEFAULT
from pdf_param import f2_dis, dy_cross_section, evaluate_all_flavors

# ============================================================
# 1. 数据点定义
# ============================================================
class DataPoint:
    """单个实验数据点"""
    __slots__ = ['x', 'q2', 'value', 'uncertainty', 'type_id', 'experiment']

    def __init__(self, x: float, q2: float, value: float,
                 uncertainty: float, type_id: str = 'DIS',
                 experiment: str = 'synthetic'):
        self.x = x
        self.q2 = q2
        self.value = value
        self.uncertainty = uncertainty
        self.type_id = type_id
        self.experiment = experiment


class DataSet:
    """实验数据集容器"""
    def __init__(self, points: Optional[List[DataPoint]] = None):
        self.points: List[DataPoint] = points if points is not None else []

    def __len__(self) -> int:
        return len(self.points)

    def add(self, pt: DataPoint):
        self.points.append(pt)

    def values(self) -> List[float]:
        return [p.value for p in self.points]

    def uncertainties(self) -> List[float]:
        return [p.uncertainty for p in self.points]

    def by_type(self, type_id: str) -> 'DataSet':
        return DataSet([p for p in self.points if p.type_id == type_id])


# ============================================================
# 2. 合成数据生成
# ============================================================
def generate_dis_data(
    params: Dict[str, float],
    x_values: Optional[List[float]] = None,
    q2_values: Optional[List[float]] = None,
    delta_stat: float = 0.02,
    delta_syst: float = 0.03,
    seed: int = 42
) -> DataSet:
    """
    生成合成 DIS F_2 数据.

    参数:
        params:     PDF 参数
        x_values:   x 值列表 (若为 None 则使用默认)
        q2_values:  Q² 值列表 (若为 None 则使用默认)
        delta_stat: 统计相对误差 (2% 默认)
        delta_syst: 系统相对误差 (3% 默认)
        seed:       随机种子

    数据点生成流程:
        1) 计算 F_2^{true}(x_i, Q²_i) (使用给定参数)
        2) 添加统计涨落: F_2 × (1 + δ_stat × N(0,1))
        3) 添加系统涨落: F_2 × (1 + δ_syst × N(0,1))
        4) 不确定度: σ_i = F_2 × √(δ_stat² + δ_syst²)

    返回:
        DataSet 包含所有 DIS 数据点
    """
    rng = random.Random(seed)
    if x_values is None:
        x_values = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85]
    if q2_values is None:
        q2_values = [5.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 10000.0]

    ds = DataSet()
    for x in x_values:
        for q2 in q2_values:
            f2_true = f2_dis(x, q2, params)
            if f2_true <= 0.0:
                continue
            # 统计与系统涨落
            stat_fluct = delta_stat * rng.gauss(0, 1)
            syst_fluct = delta_syst * rng.gauss(0, 1)
            f2_obs = f2_true * (1.0 + stat_fluct) * (1.0 + syst_fluct)
            f2_obs = max(f2_obs, EPS_NUMERICAL)
            uncertainty = f2_true * math.sqrt(delta_stat ** 2 + delta_syst ** 2)
            ds.add(DataPoint(x, q2, f2_obs, uncertainty, 'DIS', 'HERA_synthetic'))
    return ds


def generate_dy_data(
    params: Dict[str, float],
    x1_values: Optional[List[float]] = None,
    x2_values: Optional[List[float]] = None,
    M2_values: Optional[List[float]] = None,
    s_cm2: float = 1.96e4,  # sqrt(s) = 140 GeV (固定靶)
    delta_rel: float = 0.05,
    seed: int = 123
) -> DataSet:
    """
    生成合成 Drell-Yan 数据.

    参数:
        params:    PDF 参数
        x1_values: 入射强子 1 的 x 值
        x2_values: 入射强子 2 的 x 值
        M2_values: 轻子对不变质量平方
        s_cm2:     质心能量平方
        delta_rel: 相对误差
        seed:      随机种子

    Drell-Yan 截面:
        dσ/dM² = (4πα² / 9M²s) Σ_q e_q² [q(x₁)q̄(x₂) + q̄(x₁)q(x₂)]
    """
    rng = random.Random(seed)
    if x1_values is None:
        x1_values = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
    if x2_values is None:
        x2_values = [0.01, 0.02, 0.05, 0.1]
    if M2_values is None:
        M2_values = [20.0, 50.0, 100.0, 200.0]

    ds = DataSet()
    for x1 in x1_values:
        for x2 in x2_values:
            for M2 in M2_values:
                dy_true = dy_cross_section(x1, x2, M2, s_cm2, params)
                if dy_true <= 0.0:
                    continue
                fluct = delta_rel * rng.gauss(0, 1)
                dy_obs = dy_true * (1.0 + fluct)
                dy_obs = max(dy_obs, EPS_NUMERICAL)
                uncertainty = dy_true * delta_rel
                # 使用 x1 作为代表 x
                ds.add(DataPoint(x1, M2, dy_obs, uncertainty, 'DY', 'FNAL_synthetic'))
    return ds


def combine_datasets(datasets: List[DataSet]) -> DataSet:
    """合并多个数据集"""
    combined = DataSet()
    for ds in datasets:
        for pt in ds.points:
            combined.add(pt)
    return combined


# ============================================================
# 3. 蒙特卡洛概率估计 (映射自 craps_simulation)
# ============================================================
def mc_probability_estimate(n_trials: int, success_func,
                            seed: int = 42) -> Tuple[float, float]:
    """
    通用蒙特卡洛概率估计 (映射自 craps_probability):
        p ≈ N_success / N_trials
        σ_p = √[p(1−p)/N_trials]

    在 PDF 语境中, 可用于估计:
      - PDF 在给定 x 处为负的概率;
      - 动量求和规则违反的概率;
      - χ²/dof > 阈值的概率.

    参数:
        n_trials:     蒙特卡洛试验次数
        success_func: 返回 True/False 的判定函数
        seed:         随机种子
    返回:
        (p, sigma_p): 概率估计与不确定度
    """
    rng = random.Random(seed)
    n_success = 0
    for _ in range(n_trials):
        if success_func(rng):
            n_success += 1
    p = n_success / n_trials
    sigma_p = math.sqrt(p * (1 - p) / max(n_trials, 1))
    return p, sigma_p


def mc_replica_generation(
    data: DataSet, seed: int = 42
) -> DataSet:
    """
    生成一个蒙特卡洛副本:
        d_i^{replica} = d_i + σ_i × N(0,1)

    每个副本代表一组可能的"真实数据", 用于后续的 PDF 不确定性分析.

    参数:
        data: 原始数据集
        seed: 随机种子
    返回:
        新的 DataSet, 值为原始值加上随机涨落
    """
    rng = random.Random(seed)
    new_ds = DataSet()
    for pt in data.points:
        fluct = pt.uncertainty * rng.gauss(0, 1)
        new_val = max(pt.value + fluct, EPS_NUMERICAL)
        new_ds.add(DataPoint(pt.x, pt.q2, new_val, pt.uncertainty,
                             pt.type_id, pt.experiment))
    return new_ds


# ============================================================
# 4. 协方差矩阵
# ============================================================
def compute_covariance_matrix(data: DataSet) -> List[List[float]]:
    """
    计算数据协方差矩阵 (对角近似):
        C_{ij} = δ_{ij} × σ_i²

    对于更复杂的分析, 可以包含非对角项 (系统误差相关).
    """
    n = len(data)
    cov = [[0.0] * n for _ in range(n)]
    for i in range(n):
        cov[i][i] = data.points[i].uncertainty ** 2
    return cov


def compute_correlation_matrix(cov: List[List[float]]) -> List[List[float]]:
    """
    计算相关矩阵:
        ρ_{ij} = C_{ij} / √(C_{ii} × C_{jj})

    对角协方差矩阵的相关矩阵为单位矩阵.
    """
    n = len(cov)
    corr = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dii = cov[i][i]
            djj = cov[j][j]
            if dii > EPS_NUMERICAL and djj > EPS_NUMERICAL:
                corr[i][j] = cov[i][j] / math.sqrt(dii * djj)
            else:
                corr[i][j] = 1.0 if i == j else 0.0
    return corr


def text_matrix_display(matrix: List[List[float]], n_show: int = 8,
                        label: str = 'M') -> str:
    """
    矩阵的文本显示 (映射自 box_fill 的矩阵热图思想).

    使用 ASCII 字符表示矩阵元素的相对大小:
      ' ' (空格): |M_ij| < 0.1
      '·' (中点): 0.1 ≤ |M_ij| < 0.3
      '-' (短横): 0.3 ≤ |M_ij| < 0.5
      '+' (加号): 0.5 ≤ |M_ij| < 0.7
      '#' (井号): |M_ij| ≥ 0.7

    参数:
        matrix: N×N 矩阵
        n_show: 显示的最大行列数
        label:  矩阵名称
    返回:
        字符串表示
    """
    n = len(matrix)
    show_n = min(n, n_show)
    chars = ' ·-+#'
    lines = [f"  {label} ({n}×{n}, 显示前 {show_n}×{show_n}):"]
    for i in range(show_n):
        row_str = "  "
        for j in range(show_n):
            val = abs(matrix[i][j])
            idx = min(int(val / 0.2), 4)
            row_str += chars[idx]
        lines.append(row_str)
    return '\n'.join(lines)
