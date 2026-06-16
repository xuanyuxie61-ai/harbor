"""
spin_glass_couplings.py
=======================

自旋玻璃交换耦合常数 J_{ij} 的生成与统计。

物理背景
--------
Edwards-Anderson 模型中, 最近邻耦合 J_{ij} 是 quenched disorder
(淬火无序), 从某一概率分布中抽取后固定。常用两种分布:

1. 高斯分布 (Gaussian):
   P(J) = (1 / sqrt(2*pi*J_var)) * exp(-J^2 / (2*J_var))
   其中 J_var = <J^2> 是方差, 控制无序强度

2. 双模分布 (Bimodal / +-J):
   P(J) = 0.5 * delta(J - J0) + 0.5 * delta(J + J0)
   耦合等概率取 +J0 或 -J0

高斯分布在理论处理中更便利 (配分函数可解析积分),
而 +-J 模型在数值模拟中能更清晰地展现相变特征。

本模块核心算法来源于 seed project:
- 817_normal_dataset: 多维高斯随机数生成
  (映射为耦合常数的多维联合高斯采样)
- 229_cube_arbq_rule: 任意阶高斯求积公式
  (映射为对耦合分布的数值积分)
"""

import numpy as np
from typing import Dict, Tuple, Literal, Optional
from spin_lattice_geometry import CubicLattice3D


# =====================================================================
#  耦合分布类型
# =====================================================================

CouplingDistribution = Literal["gaussian", "bimodal"]


# =====================================================================
#  耦合常数生成器
# =====================================================================

class CouplingGenerator:
    """
    生成 Edwards-Anderson 自旋玻璃的耦合常数。

    参数
    ----
    distribution : str, "gaussian" 或 "bimodal"
        耦合分布类型
    J_var : float, default 1.0
        高斯分布的方差 <J^2> (即 J 的标准差 sigma = sqrt(J_var))
    J0 : float, default 1.0
        双模分布的 |J| 值
    rng : numpy.random.Generator or None
        随机数生成器 (用于可复现实验)
    """

    def __init__(self,
                 distribution: CouplingDistribution = "gaussian",
                 J_var: float = 1.0,
                 J0: float = 1.0,
                 rng: Optional[np.random.Generator] = None):
        if distribution not in ("gaussian", "bimodal"):
            raise ValueError(f"不支持的分布: {distribution}")
        if J_var <= 0:
            raise ValueError(f"J_var 必须 > 0, 当前 J_var={J_var}")
        if J0 <= 0:
            raise ValueError(f"J0 必须 > 0, 当前 J0={J0}")
        self.distribution = distribution
        self.J_var = J_var
        self.J0 = J0
        self.rng = rng if rng is not None else np.random.default_rng()

    def generate(self, lattice: CubicLattice3D) -> Dict[Tuple[int, int], float]:
        """
        生成一组耦合常数 {J_{ij}}。

        返回
        ----
        couplings : dict {(i, j): J_ij}
            键 (i, j) 的耦合常数, i < j
        """
        bonds = lattice.get_bond_list()
        n_bond = len(bonds)
        if n_bond == 0:
            return {}

        if self.distribution == "gaussian":
            J_array = self._sample_gaussian(n_bond)
        elif self.distribution == "bimodal":
            J_array = self._sample_bimodal(n_bond)
        else:
            raise ValueError(f"未实现的分布: {self.distribution}")

        couplings = {}
        for k, (i, j) in enumerate(bonds):
            couplings[(i, j)] = float(J_array[k])
        return couplings

    def _sample_gaussian(self, n: int) -> np.ndarray:
        """
        高斯采样: J ~ N(0, J_var)

        对应种子项目 817_normal_dataset 中:
            x = mu' + A * z
        其中 A 是方差矩阵的 Cholesky 分解。
        在标量情况下退化为 J = sqrt(J_var) * z, z ~ N(0,1)
        """
        sigma = np.sqrt(self.J_var)
        return self.rng.normal(loc=0.0, scale=sigma, size=n)

    def _sample_bimodal(self, n: int) -> np.ndarray:
        """
        双模采样: J = +J0 或 -J0, 各概率 0.5

        对应: P(J) = 0.5*delta(J-J0) + 0.5*delta(J+J0)
        """
        signs = self.rng.choice([-1.0, +1.0], size=n)
        return self.J0 * signs

    # -----------------------------------------------------------------
    #  统计性质
    # -----------------------------------------------------------------

    def theoretical_mean(self) -> float:
        """理论均值 <J>"""
        return 0.0

    def theoretical_variance(self) -> float:
        """理论方差 <J^2> - <J>^2"""
        if self.distribution == "gaussian":
            return self.J_var
        else:
            return self.J0 ** 2

    def theoretical_skewness(self) -> float:
        """理论偏度 <(J-<J>)^3> / sigma^3"""
        return 0.0  # 两种分布关于 0 对称

    def theoretical_kurtosis_excess(self) -> float:
        """
        理论超额峰度:
        高斯分布: 0
        双模分布: -2 (platykurtic)
        """
        if self.distribution == "gaussian":
            return 0.0
        else:
            return -2.0


# =====================================================================
#  耦合矩阵 (稀疏表示)
# =====================================================================

def couplings_to_matrix(couplings: Dict[Tuple[int, int], float],
                        N: int) -> np.ndarray:
    """
    将字典形式的耦合转为对称矩阵 J_matrix[N, N]。

    J_matrix[i, j] = J[i, j] = J[j, i]
    J_matrix[i, i] = 0 (无自相互作用)
    """
    J_matrix = np.zeros((N, N), dtype=np.float64)
    for (i, j), Jval in couplings.items():
        J_matrix[i, j] = Jval
        J_matrix[j, i] = Jval
    return J_matrix


def couplings_sparse_diagonal(couplings: Dict[Tuple[int, int], float],
                              N: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    提取耦合矩阵的对角优势 (diagonal dominance):
        d_i = sum_{j in nbr(i)} |J_{ij}|

    返回
    ----
    diag_dom : ndarray (N,)
        每个格点的对角优势值
    off_diag_sum : ndarray (N,)
        每个格点的非对角绝对值之和 (等于 diag_dom)
    """
    diag_dom = np.zeros(N, dtype=np.float64)
    for (i, j), Jval in couplings.items():
        diag_dom[i] += abs(Jval)
        diag_dom[j] += abs(Jval)
    return diag_dom, diag_dom.copy()


# =====================================================================
#  耦合统计验证
# =====================================================================

def verify_coupling_statistics(couplings: Dict[Tuple[int, int], float],
                               generator: CouplingGenerator,
                               tolerance: float = 0.5) -> Dict[str, float]:
    """
    验证生成的耦合常数是否符合目标分布的统计特性。

    计算:
    - 样本均值 <J>_sample
    - 样本方差 <J^2>_sample - <J>^2_sample
    - 样本偏度
    - 样本超额峰度

    返回
    ----
    stats : dict
        样本统计量
    """
    J_vals = np.array(list(couplings.values()))
    n = len(J_vals)
    if n < 4:
        return {"sample_mean": 0.0, "sample_var": 0.0}

    mean_J = np.mean(J_vals)
    var_J = np.var(J_vals, ddof=1)
    std_J = np.sqrt(var_J) if var_J > 0 else 1e-15

    # 偏度: gamma_1 = <(J - <J>)^3> / sigma^3
    skew_J = np.mean(((J_vals - mean_J) / std_J) ** 3)

    # 超额峰度: gamma_2 = <(J - <J>)^4> / sigma^4 - 3
    kurt_J = np.mean(((J_vals - mean_J) / std_J) ** 4) - 3.0

    theo_mean = generator.theoretical_mean()
    theo_var = generator.theoretical_variance()

    return {
        "sample_mean": float(mean_J),
        "sample_var": float(var_J),
        "sample_skewness": float(skew_J),
        "sample_kurtosis_excess": float(kurt_J),
        "theoretical_mean": theo_mean,
        "theoretical_var": theo_var,
        "mean_deviation": abs(mean_J - theo_mean),
        "var_deviation": abs(var_J - theo_var),
        "n_samples": n,
    }


# =====================================================================
#  有效耦合场 (effective coupling field)
# =====================================================================

def effective_local_field(spins: np.ndarray,
                          couplings: Dict[Tuple[int, int], float],
                          site: int,
                          lattice: CubicLattice3D) -> float:
    """
    计算作用在格点 site 上的有效局部场:
        h_i^{eff} = sum_{j in nbr(i)} J_{ij} * S_j

    这是 Langevin 动力学中驱动自旋弛豫的力。
    在有限差分格式中, 这对应于离散拉普拉斯算子的推广:
        h_i^{eff} = sum_{mu=1}^{d} [J_{i,i+e_mu} * S_{i+e_mu} + J_{i,i-e_mu} * S_{i-e_mu}]
    """
    h_eff = 0.0
    for nbr in lattice.neighbor_table[site]:
        if nbr < 0:
            continue
        key = (min(site, nbr), max(site, nbr))
        J = couplings.get(key, 0.0)
        h_eff += J * spins.flat[nbr]
    return h_eff
