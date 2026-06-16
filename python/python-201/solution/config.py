"""
config.py — 全局配置模块
=========================
管理多项式混沌展开 (PCE) 的全部参数: 概率测度、基函数截断、
稀疏网格层级、Cahn-Hilliard 物理参数、自适应细化容差等。

映射种子项目:
  - 096_bisection_min: 二分法搜索最优多项式阶数
  - 680_line_grid: 一维网格生成 → 随机空间配置
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict


# ============================================================
#  概率测度配置 (每个随机变量一个测度)
# ============================================================
@dataclass
class MeasureConfig:
    """单变量概率测度配置"""
    name: str                    # 变量名称 (如 "kappa", "gamma")
    measure_type: str            # "gauss" | "uniform" | "beta"
    params: Dict[str, float]     # 测度参数
    # gauss: {"mu": float, "sigma": float}
    # uniform: {"a": float, "b": float}
    # beta: {"alpha": float, "beta": float, "loc": float, "scale": float}

    @property
    def mean(self) -> float:
        if self.measure_type == "gauss":
            return self.params["mu"]
        elif self.measure_type == "uniform":
            return 0.5 * (self.params["a"] + self.params["b"])
        elif self.measure_type == "beta":
            a, b = self.params["alpha"], self.params["beta"]
            return self.params["loc"] + self.params["scale"] * a / (a + b)
        raise ValueError(f"Unknown measure: {self.measure_type}")

    @property
    def variance(self) -> float:
        if self.measure_type == "gauss":
            return self.params["sigma"] ** 2
        elif self.measure_type == "uniform":
            a, b = self.params["a"], self.params["b"]
            return (b - a) ** 2 / 12.0
        elif self.measure_type == "beta":
            a, b = self.params["alpha"], self.params["beta"]
            s = self.params["scale"]
            return s ** 2 * a * b / ((a + b) ** 2 * (a + b + 1))
        raise ValueError(f"Unknown measure: {self.measure_type}")

    @property
    def support(self) -> Tuple[float, float]:
        if self.measure_type == "gauss":
            mu, sigma = self.params["mu"], self.params["sigma"]
            return (mu - 4 * sigma, mu + 4 * sigma)
        elif self.measure_type == "uniform":
            return (self.params["a"], self.params["b"])
        elif self.measure_type == "beta":
            loc = self.params["loc"]
            scale = self.params["scale"]
            return (loc, loc + scale)
        raise ValueError(f"Unknown measure: {self.measure_type}")


# ============================================================
#  多项式基函数配置
# ============================================================
@dataclass
class BasisConfig:
    """PCE 基函数配置"""
    max_degree: int = 5                    # 最大多项式阶数 p
    truncation: str = "total_order"        # "total_order" | "hyperbolic"
    hyperbolic_q: float = 0.5             # 双曲交叉参数 q ∈ (0,1]
    # 总阶数截断: |i|_1 = i_1 + ... + i_d <= p
    # 双曲截断:   (sum i_k^q)^(1/q) <= p

    @property
    def num_basis_estimate(self) -> int:
        """估计基函数数量 (对 total_order)"""
        # 实际数量由 PolynomialBasis 计算
        return self.max_degree + 1


# ============================================================
#  多元素随机空间分解配置
# ============================================================
@dataclass
class MultiElementConfig:
    """多元素 gPC (ME-gPC) 配置"""
    initial_elements: int = 2              # 初始元素个数
    max_elements: int = 32                 # 最大元素个数
    local_degree: int = 3                  # 每个元素内的局部多项式阶数
    refinement_tolerance: float = 1e-4     # 自适应细化容差
    refinement_indicator: str = "residual" # "residual" | "spectral_decay"
    max_refinement_levels: int = 8         # 最大细化层数


# ============================================================
#  稀疏网格配置
# ============================================================
@dataclass
class SparseGridConfig:
    """Smolyak 稀疏网格配置"""
    level: int = 4                         # 稀疏网格层级
    growth_rule: str = "linear"            # "linear" | "exponential"
    quadrature_type: str = "gauss-patterson"  # "gauss" | "gauss-patterson" | "clenshaw-curtis"


# ============================================================
#  Cahn-Hilliard 物理参数
# ============================================================
@dataclass
class CahnHilliardConfig:
    """随机 Cahn-Hilliard 方程参数"""
    # 计算域
    Lx: float = 1.0                        # x 方向域长度
    Ly: float = 1.0                        # y 方向域长度
    nx: int = 64                           # x 方向网格点数
    ny: int = 64                           # y 方向网格点数

    # 物理参数 (标称值)
    kappa_mean: float = 0.01               # 梯度能量系数 κ (界面张力)
    kappa_std: float = 0.002               # κ 的不确定性标准差
    gamma_mean: float = 1.0                # 双阱势参数 γ
    gamma_std: float = 0.1                 # γ 的不确定性标准差
    mobility_mean: float = 1.0             # 迁移率 M
    mobility_std: float = 0.15             # M 的不确定性标准差

    # 时间积分
    dt: float = 1e-4                       # 时间步长
    n_steps: int = 500                     # 时间步数
    theta: float = 0.5                     # theta 方法参数 (0.5=Crank-Nicolson)

    # 初始条件
    c0_mean: float = 0.0                   # 初始浓度均值
    c0_noise_std: float = 0.05             # 初始扰动标准差
    random_seed: int = 42                  # 随机种子

    # 边界条件
    bc_type: str = "periodic"              # "periodic" | "no_flux"


# ============================================================
#  自适应细化配置
# ============================================================
@dataclass
class RefinementConfig:
    """自适应细化控制"""
    enable: bool = True
    max_iterations: int = 10
    error_tolerance: float = 1e-5
    coarsening_tolerance: float = 1e-7
    min_element_size: float = 1e-6
    max_element_size: float = 1.0


# ============================================================
#  神经网络闭合模型配置
# ============================================================
@dataclass
class NeuralClosureConfig:
    """神经闭合模型配置"""
    hidden_dim: int = 64
    n_layers: int = 3
    learning_rate: float = 1e-3
    n_epochs: int = 200
    batch_size: int = 32
    regularization: float = 1e-4           # L2 正则化系数
    activation: str = "tanh"               # "tanh" | "relu" | "sigmoid"


# ============================================================
#  多保真度配置
# ============================================================
@dataclass
class MultiFidelityConfig:
    """多保真度控制变量配置"""
    n_high_fidelity: int = 20              # 高保真样本数
    n_low_fidelity: int = 100              # 低保真样本数
    correlation_model: str = "linear"      # "linear" | "nonlinear"


# ============================================================
#  MCMC / 贝叶斯配置
# ============================================================
@dataclass
class MCMCConfig:
    """MCMC 贝叶斯推断配置"""
    n_walkers: int = 32
    n_burnin: int = 200
    n_samples: int = 500
    proposal_scale: float = 0.1
    target_acceptance: float = 0.234       # 最优接受率 (高维)
    random_seed: int = 123


# ============================================================
#  全局配置聚合
# ============================================================
@dataclass
class GlobalConfig:
    """聚合所有子配置"""
    measures: List[MeasureConfig] = field(default_factory=list)
    basis: BasisConfig = field(default_factory=BasisConfig)
    multi_element: MultiElementConfig = field(default_factory=MultiElementConfig)
    sparse_grid: SparseGridConfig = field(default_factory=SparseGridConfig)
    cahn_hilliard: CahnHilliardConfig = field(default_factory=CahnHilliardConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    neural_closure: NeuralClosureConfig = field(default_factory=NeuralClosureConfig)
    multifidelity: MultiFidelityConfig = field(default_factory=MultiFidelityConfig)
    mcmc: MCMCConfig = field(default_factory=MCMCConfig)


def create_default_config() -> GlobalConfig:
    """
    创建默认的全局配置。

    随机变量:
      - κ (梯度能量系数): 高斯分布
      - γ (双阱势参数): 均匀分布
      - M (迁移率): 高斯分布 (截断至正值)
    """
    config = GlobalConfig()

    # 三个随机输入参数
    config.measures = [
        MeasureConfig(
            name="kappa",
            measure_type="gauss",
            params={"mu": 0.01, "sigma": 0.002}
        ),
        MeasureConfig(
            name="gamma",
            measure_type="uniform",
            params={"a": 0.8, "b": 1.2}
        ),
        MeasureConfig(
            name="mobility",
            measure_type="gauss",
            params={"mu": 1.0, "sigma": 0.15}
        ),
    ]

    # PCE 基函数
    config.basis = BasisConfig(
        max_degree=4,
        truncation="total_order",
    )

    # 多元素
    config.multi_element = MultiElementConfig(
        initial_elements=2,
        max_elements=16,
        local_degree=3,
        refinement_tolerance=1e-4,
    )

    # 稀疏网格
    config.sparse_grid = SparseGridConfig(
        level=4,
        growth_rule="linear",
        quadrature_type="gauss-patterson",
    )

    # Cahn-Hilliard
    config.cahn_hilliard = CahnHilliardConfig(
        nx=32, ny=32,
        dt=1e-4, n_steps=200,
    )

    # 自适应细化
    config.refinement = RefinementConfig(
        enable=True,
        max_iterations=5,
        error_tolerance=1e-4,
    )

    # 神经闭合
    config.neural_closure = NeuralClosureConfig(
        hidden_dim=32,
        n_layers=2,
        n_epochs=100,
    )

    # 多保真度
    config.multifidelity = MultiFidelityConfig(
        n_high_fidelity=15,
        n_low_fidelity=80,
    )

    # MCMC
    config.mcmc = MCMCConfig(
        n_walkers=24,
        n_burnin=100,
        n_samples=300,
    )

    return config


def estimate_total_basis_dim(config: GlobalConfig) -> int:
    """
    估计 PCE 基函数的总维度。

    对 total_order 截断，d 维随机空间，阶数 p:
        P = C(d + p, p) = (d+p)! / (d! * p!)

    对 multi-element, 乘以元素个数。
    """
    d = len(config.measures)
    p = config.basis.max_degree
    # 组合数 C(d+p, p)
    from math import comb
    P = comb(d + p, p)
    return P
