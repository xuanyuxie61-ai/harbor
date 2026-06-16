"""
multi_element.py — 多元素广义多项式混沌 (ME-gPC) 模块
=======================================================
实现随机空间的多元素分解和局部多项式混沌展开。
每个元素内使用独立的 gPC 展开, 元素之间通过界面条件耦合。

核心公式:
  随机空间分解:
    Ξ = ∪_{e=1}^{E} Ξ_e,  Ξ_e ∩ Ξ_{e'} = ∅

  局部展开:
    u(ξ) ≈ Σ_{k=0}^{P_e} û_k^{(e)} Ψ_k^{(e)}(ξ),  ξ ∈ Ξ_e

  界面连续性:
    u(ξ_e^+) = u(ξ_{e+1}^-)

  自适应细化:
    基于局部误差指示器 η_e 选择待细化元素

映射种子项目:
  - 1434_zombie_ode: 耦合 ODE 系统 → 元素间耦合
  - 1246_Gaulios: 有限-无穷视界耦合 → 多元素耦合
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from config import (MeasureConfig, BasisConfig, MultiElementConfig,
                    GlobalConfig)
from polynomial_basis import OrthogonalPolynomialBasis
from measure import gauss_quadrature, cdf_eval


class StochasticElement:
    """
    随机空间中的一个元素。

    每个元素 Ξ_e = [a_e, b_e]^d 拥有独立的局部 PCE 展开。

    属性:
        element_id:    元素编号
        bounds:        shape (d, 2), 各维度的 [a, b]
        local_basis:   局部基函数
        coefficients:  PCE 系数, shape (P,) 或 (n_field, P)
        volume:        元素的概率体积
    """

    def __init__(self, element_id: int,
                 bounds: np.ndarray,
                 measures: List[MeasureConfig],
                 local_degree: int):
        self.element_id = element_id
        self.bounds = bounds.copy()  # (d, 2)
        self.n_dim = len(measures)

        # 创建局部测度 (映射到局部坐标)
        self.local_measures = self._create_local_measures(measures)

        # 创建局部基函数
        local_config = BasisConfig(
            max_degree=local_degree,
            truncation="total_order",
        )
        self.local_basis = OrthogonalPolynomialBasis(
            local_config, self.local_measures)

        # 系数 (初始为零)
        self.coefficients: Optional[np.ndarray] = None

        # 计算概率体积
        self.volume = self._compute_volume(measures)

    def _create_local_measures(self,
                               global_measures: List[MeasureConfig]
                               ) -> List[MeasureConfig]:
        """
        创建局部测度: 将全局测度限制在元素域内并重新归一化。

        对均匀测度: 直接截断到 [a_e, b_e]
        对高斯测度: 使用条件分布 (截断正态)
        """
        local_measures = []
        for dim in range(self.n_dim):
            gm = global_measures[dim]
            a_e, b_e = self.bounds[dim]

            if gm.measure_type == "uniform":
                # 直接截断
                lm = MeasureConfig(
                    name=f"{gm.name}_elem{self.element_id}_d{dim}",
                    measure_type="uniform",
                    params={"a": a_e, "b": b_e},
                )
            elif gm.measure_type == "gauss":
                # 截断高斯: 用均匀近似 (简化)
                lm = MeasureConfig(
                    name=f"{gm.name}_elem{self.element_id}_d{dim}",
                    measure_type="uniform",
                    params={"a": a_e, "b": b_e},
                )
            else:
                lm = MeasureConfig(
                    name=f"{gm.name}_elem{self.element_id}_d{dim}",
                    measure_type="uniform",
                    params={"a": a_e, "b": b_e},
                )
            local_measures.append(lm)

        return local_measures

    def _compute_volume(self,
                        global_measures: List[MeasureConfig]) -> float:
        """计算元素的概率体积 (在概率测度下)"""
        vol = 1.0
        for dim in range(self.n_dim):
            a_e, b_e = self.bounds[dim]
            gm = global_measures[dim]
            cdf_a = float(cdf_eval(gm, np.array([a_e]))[0])
            cdf_b = float(cdf_eval(gm, np.array([b_e]))[0])
            vol *= (cdf_b - cdf_a)
        return max(vol, 1e-15)

    def set_coefficients(self, coeffs: np.ndarray):
        """设置 PCE 系数"""
        self.coefficients = coeffs.copy()

    def evaluate_mean(self) -> float:
        """
        计算元素的局部均值。

        E[u|Ξ_e] = û_0^{(e)} (因为 Ψ_0 = 1 且归一化)
        """
        if self.coefficients is None:
            return 0.0
        if self.coefficients.ndim == 1:
            return float(self.coefficients[0])
        return float(np.mean(self.coefficients[:, 0]))

    def evaluate_variance(self) -> float:
        """
        计算元素的局部方差。

        Var[u|Ξ_e] = Σ_{k=1}^{P} (û_k^{(e)})^2
        """
        if self.coefficients is None or len(self.coefficients) < 2:
            return 0.0
        if self.coefficients.ndim == 1:
            return float(np.sum(self.coefficients[1:] ** 2))
        return float(np.sum(self.coefficients[:, 1:] ** 2, axis=1).mean())


class MultiElementGalerkin:
    """
    多元素广义多项式混沌 (ME-gPC) 管理器。

    管理随机空间的多元素分解, 协调各元素的局部 PCE 展开。

    映射 1434_zombie_ode: 将 SIR 模型的分段行为
    推广为随机空间的分段多项式逼近。

    映射 1246_Gaulios: 将有限-无穷horizon耦合
    推广为元素间的界面耦合。
    """

    def __init__(self, config: GlobalConfig):
        self.config = config
        self.me_config = config.multi_element
        self.measures = config.measures
        self.n_dim = len(config.measures)

        # 初始元素列表
        self.elements: List[StochasticElement] = []
        self._initialize_elements()

    def _initialize_elements(self):
        """初始化均匀分解的随机空间元素"""
        n_elem = self.me_config.initial_elements
        local_deg = self.me_config.local_degree

        # 仅对第一维度进行初始分解 (可扩展到多维)
        supports = [m.support for m in self.measures]

        if self.n_dim == 1:
            a, b = supports[0]
            elem_width = (b - a) / n_elem
            for e in range(n_elem):
                a_e = a + e * elem_width
                b_e = a + (e + 1) * elem_width
                bounds = np.array([[a_e, b_e]])
                elem = StochasticElement(
                    e, bounds, self.measures, local_deg)
                self.elements.append(elem)
        else:
            # 多维: 使用张量积分解
            # 为简化, 仅沿第一维度分解
            a, b = supports[0]
            elem_width = (b - a) / n_elem
            for e in range(n_elem):
                bounds = np.zeros((self.n_dim, 2))
                bounds[0] = [a + e * elem_width,
                             a + (e + 1) * elem_width]
                for d in range(1, self.n_dim):
                    bounds[d] = supports[d]
                elem = StochasticElement(
                    e, bounds, self.measures, local_deg)
                self.elements.append(elem)

    @property
    def n_elements(self) -> int:
        return len(self.elements)

    @property
    def total_basis_size(self) -> int:
        """所有元素的总基函数数量"""
        return sum(e.local_basis.n_basis for e in self.elements)

    def get_element_bounds(self) -> List[np.ndarray]:
        """获取所有元素的边界"""
        return [e.bounds.copy() for e in self.elements]

    def refine_element(self, elem_idx: int,
                       split_dim: Optional[int] = None):
        """
        对指定元素进行二分细化。

        将元素 Ξ_e 沿维度 split_dim 对半分为两个子元素。

        参数:
            elem_idx: 待细化元素的索引
            split_dim: 细化维度 (None 则选择最宽维度)
        """
        if elem_idx < 0 or elem_idx >= len(self.elements):
            return

        elem = self.elements[elem_idx]

        if split_dim is None:
            # 选择最宽的维度
            widths = elem.bounds[:, 1] - elem.bounds[:, 0]
            split_dim = int(np.argmax(widths))

        # 检查最小尺寸
        width = elem.bounds[split_dim, 1] - elem.bounds[split_dim, 0]
        if width < self.config.refinement.min_element_size:
            return  # 已达最小尺寸

        mid = 0.5 * (elem.bounds[split_dim, 0] +
                     elem.bounds[split_dim, 1])

        # 创建两个子元素
        new_id_start = max(e.element_id for e in self.elements) + 1

        bounds1 = elem.bounds.copy()
        bounds1[split_dim, 1] = mid
        elem1 = StochasticElement(
            new_id_start, bounds1, self.measures,
            self.me_config.local_degree)

        bounds2 = elem.bounds.copy()
        bounds2[split_dim, 0] = mid
        elem2 = StochasticElement(
            new_id_start + 1, bounds2, self.measures,
            self.me_config.local_degree)

        # 继承系数 (投影到子元素)
        if elem.coefficients is not None:
            elem1.set_coefficients(elem.coefficients.copy())
            elem2.set_coefficients(elem.coefficients.copy())

        # 替换原元素
        self.elements.pop(elem_idx)
        self.elements.insert(elem_idx, elem2)
        self.elements.insert(elem_idx, elem1)

    def compute_global_statistics(self) -> Tuple[float, float]:
        """
        计算全局均值和方差。

        E[u] = Σ_e vol_e * E[u|Ξ_e]
        Var[u] = Σ_e vol_e * (Var[u|Ξ_e] + (E[u|Ξ_e] - E[u])^2)

        返回:
            global_mean, global_variance
        """
        global_mean = 0.0
        for elem in self.elements:
            global_mean += elem.volume * elem.evaluate_mean()

        global_var = 0.0
        for elem in self.elements:
            local_mean = elem.evaluate_mean()
            local_var = elem.evaluate_variance()
            global_var += elem.volume * (
                local_var + (local_mean - global_mean) ** 2)

        return global_mean, max(global_var, 0.0)

    def compute_error_indicators(self) -> np.ndarray:
        """
        计算各元素的误差指示器。

        使用谱衰减速率:
          η_e = Σ_{k=p-1}^{P_e} (û_k^{(e)})^2 / Σ_{k=0}^{P_e} (û_k^{(e)})^2

        返回:
            indicators: shape (n_elements,)
        """
        indicators = np.zeros(self.n_elements)
        for idx, elem in enumerate(self.elements):
            if elem.coefficients is None:
                indicators[idx] = 1.0  # 未初始化 → 最大误差
                continue

            coeffs = elem.coefficients
            if coeffs.ndim > 1:
                coeffs = coeffs[0]  # 取第一个场分量

            total_energy = np.sum(coeffs ** 2)
            if total_energy < 1e-30:
                indicators[idx] = 0.0
                continue

            # 高阶能量 (最后 1/3 的系数)
            n = len(coeffs)
            n_tail = max(n // 3, 1)
            tail_energy = np.sum(coeffs[-n_tail:] ** 2)
            indicators[idx] = tail_energy / total_energy

        return indicators

    def summary(self) -> str:
        """返回 ME-gPC 状态的摘要"""
        mean, var = self.compute_global_statistics()
        lines = [
            f"ME-gPC 状态:",
            f"  元素个数: {self.n_elements}",
            f"  总基函数数: {self.total_basis_size}",
            f"  全局均值: {mean:.6e}",
            f"  全局方差: {var:.6e}",
        ]
        for idx, elem in enumerate(self.elements):
            bounds_str = ", ".join(
                f"[{b[0]:.3f}, {b[1]:.3f}]" for b in elem.bounds)
            lines.append(
                f"  Element {idx}: bounds={{{bounds_str}}}, "
                f"vol={elem.volume:.4f}, "
                f"mean={elem.evaluate_mean():.4e}, "
                f"var={elem.evaluate_variance():.4e}")
        return "\n".join(lines)
