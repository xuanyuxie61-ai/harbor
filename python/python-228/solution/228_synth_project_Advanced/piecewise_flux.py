"""
piecewise_flux.py — 分段通量表示与 Hermite 插值
=================================================

融合种子项目:
  - 923_pwc_plot_1d : 分段常数/线性函数表示

本模块实现簇射通量的分段表示:

1. 分段常数 (PWC):
   f(x) = f_i  for x in [x_i, x_{i+1})

2. 分段线性 (PWL):
   f(x) = f_i + (f_{i+1} - f_i) * (x - x_i) / h_i

3. 分段三次 Hermite (PCHIP):
   f(x) = h00(t)*f_i + h10(t)*m_i*h + h01(t)*f_{i+1} + h11(t)*m_{i+1}*h
   其中 t = (x - x_i)/h, h = x_{i+1} - x_i
   h00 = 2t^3 - 3t^2 + 1
   h10 = t^3 - 2t^2 + t
   h01 = -2t^3 + 3t^2
   h11 = t^3 - t^2

4. 单调性保持 (Fritsch-Carlson):
   调整斜率 m_i 保证分段插值的单调性

5. 通量守恒重建:
   integral_{x_i}^{x_{i+1}} f(x) dx = F_i (给定积分值)

关键公式 (能量守恒):
  sum_i integral_{cell_i} f(x) dx = E_total
"""

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable


@dataclass
class PiecewiseFlux:
    """分段通量数据"""
    breakpoints: List[float]   # 断点位置
    values: List[float]        # 函数值
    slopes: Optional[List[float]] = None  # 斜率 (Hermite)
    integral_values: Optional[List[float]] = None  # 单元积分值

    @property
    def n_intervals(self) -> int:
        return len(self.breakpoints) - 1

    @property
    def total_integral(self) -> float:
        """总积分 (梯形法则)"""
        result = 0.0
        for i in range(self.n_intervals):
            h = self.breakpoints[i + 1] - self.breakpoints[i]
            result += h * (self.values[i] + self.values[i + 1]) / 2.0
        return result


class PiecewiseFluxReconstructor:
    """分段通量重构器"""

    def __init__(self):
        pass

    def from_piecewise_constant(
        self, breakpoints: List[float], values: List[float],
    ) -> PiecewiseFlux:
        """
        从分段常数表示创建

        f(x) = values[i]  for x in [breakpoints[i], breakpoints[i+1])
        """
        if len(breakpoints) != len(values) + 1:
            raise ValueError(
                f"断点数 ({len(breakpoints)}) 应比值数 ({len(values)}) 多 1"
            )
        return PiecewiseFlux(
            breakpoints=list(breakpoints),
            values=list(values),
        )

    def from_piecewise_linear(
        self, breakpoints: List[float], values: List[float],
    ) -> PiecewiseFlux:
        """
        从分段线性表示创建 (自动计算斜率)

        斜率: m_i = (f_{i+1} - f_i) / (x_{i+1} - x_i)
        """
        if len(breakpoints) != len(values):
            raise ValueError("断点数应与值数相同")

        slopes = []
        for i in range(len(values) - 1):
            h = breakpoints[i + 1] - breakpoints[i]
            if h < 1e-30:
                slopes.append(0.0)
            else:
                slopes.append((values[i + 1] - values[i]) / h)
        slopes.append(slopes[-1] if slopes else 0.0)

        return PiecewiseFlux(
            breakpoints=list(breakpoints),
            values=list(values),
            slopes=slopes,
        )

    def from_hermite_pchip(
        self, breakpoints: List[float], values: List[float],
    ) -> PiecewiseFlux:
        """
        分段三次 Hermite 插值 (PCHIP — 单调性保持)

        Fritsch-Carlson 方法选择斜率:
        1. 计算差商: d_i = (f_{i+1} - f_i) / h_i
        2. 初始斜率: m_i = (d_{i-1} * h_i + d_i * h_{i-1}) / (h_{i-1} + h_i)
        3. 单调性修正: 若 d_{i-1} * d_i <= 0, 则 m_i = 0
        4. 限幅: |m_i| <= 3 * min(|d_{i-1}|, |d_i|)
        """
        n = len(breakpoints)
        if n != len(values):
            raise ValueError("断点数与值数不匹配")

        if n < 2:
            return PiecewiseFlux(
                breakpoints=list(breakpoints), values=list(values),
                slopes=[0.0],
            )

        # 差商
        h = [breakpoints[i + 1] - breakpoints[i] for i in range(n - 1)]
        d = [(values[i + 1] - values[i]) / max(h[i], 1e-30) for i in range(n - 1)]

        # 初始斜率
        slopes = [0.0] * n
        slopes[0] = d[0]
        slopes[n - 1] = d[-1]

        for i in range(1, n - 1):
            if h[i - 1] + h[i] < 1e-30:
                slopes[i] = 0.0
            else:
                # 加权调和平均
                w1 = 2.0 * h[i] + h[i - 1]
                w2 = h[i] + 2.0 * h[i - 1]
                slopes[i] = (w1 + w2) / (w1 / max(d[i - 1], 1e-30) + w2 / max(d[i], 1e-30))

        # 单调性修正
        for i in range(n):
            if i == 0:
                d_left = d[0]
                d_right = d[0]
            elif i == n - 1:
                d_left = d[-1]
                d_right = d[-1]
            else:
                d_left = d[i - 1]
                d_right = d[i]

            if d_left * d_right <= 0:
                slopes[i] = 0.0
            else:
                # 限幅
                max_slope = 3.0 * min(abs(d_left), abs(d_right))
                if abs(slopes[i]) > max_slope:
                    slopes[i] = max_slope * (1.0 if slopes[i] > 0 else -1.0)

        return PiecewiseFlux(
            breakpoints=list(breakpoints),
            values=list(values),
            slopes=slopes,
        )

    def evaluate(
        self, flux: PiecewiseFlux, x: float, method: str = "linear",
    ) -> float:
        """
        在点 x 处求值

        method:
          "constant" : 分段常数
          "linear"   : 分段线性
          "hermite"  : 三次 Hermite
        """
        bp = flux.breakpoints
        vals = flux.values
        n = len(bp)

        if n == 0:
            return 0.0
        if x <= bp[0]:
            return vals[0]
        if x >= bp[-1]:
            return vals[-1]

        # 二分查找区间
        lo, hi = 0, n - 1
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if bp[mid] <= x:
                lo = mid
            else:
                hi = mid

        i = lo
        h = bp[i + 1] - bp[i]
        if h < 1e-30:
            return vals[i]
        t = (x - bp[i]) / h

        if method == "constant":
            return vals[i]
        elif method == "linear":
            return vals[i] + t * (vals[i + 1] - vals[i])
        elif method == "hermite" and flux.slopes is not None:
            # Hermite 基函数
            h00 = 2.0 * t * t * t - 3.0 * t * t + 1.0
            h10 = t * t * t - 2.0 * t * t + t
            h01 = -2.0 * t * t * t + 3.0 * t * t
            h11 = t * t * t - t * t
            return (
                h00 * vals[i]
                + h10 * h * flux.slopes[i]
                + h01 * vals[i + 1]
                + h11 * h * flux.slopes[i + 1]
            )
        else:
            return vals[i] + t * (vals[i + 1] - vals[i])

    def conserve_reconstruction(
        self,
        breakpoints: List[float],
        cell_integrals: List[float],
    ) -> PiecewiseFlux:
        """
        守恒重建: 从单元积分值重构分段线性函数

        约束: integral_{x_i}^{x_{i+1}} f(x) dx = cell_integrals[i]

        方法: 从积分值反推节点值
          f_avg_i = cell_integrals[i] / h_i
          对于分段线性: f_avg_i = (f_i + f_{i+1}) / 2
          因此: f_{i+1} = 2*f_avg_i - f_i

        需要边界条件 f_0 = 某值 (从上下文确定)。
        """
        n_cells = len(cell_integrals)
        if len(breakpoints) != n_cells + 1:
            raise ValueError("断点数应比单元数多1")

        # 计算单元平均值
        averages = []
        for i in range(n_cells):
            h = breakpoints[i + 1] - breakpoints[i]
            if h < 1e-30:
                averages.append(0.0)
            else:
                averages.append(cell_integrals[i] / h)

        # 从平均值恢复节点值
        values = [0.0] * (n_cells + 1)
        # 左边界: 用第一个单元平均值近似
        values[0] = averages[0] if averages else 0.0
        for i in range(n_cells):
            values[i + 1] = 2.0 * averages[i] - values[i]

        # 保证非负 (物理约束)
        values = [max(v, 0.0) for v in values]

        return self.from_piecewise_linear(breakpoints, values)

    def compute_total_variation(
        self, flux: PiecewiseFlux,
    ) -> float:
        """
        计算总变差 TV(f) = sum_i |f_{i+1} - f_i|

        TV  diminishing 性质是高精度格式的关键要求。
        """
        tv = 0.0
        for i in range(len(flux.values) - 1):
            tv += abs(flux.values[i + 1] - flux.values[i])
        return tv

    def limit_slopes(
        self, flux: PiecewiseFlux, limiter: str = "minmod",
    ) -> PiecewiseFlux:
        """
        斜率限制器 (TVD 格式)

        minmod:  m = minmod(d_left, d_right)
        van_leer: m = 2*d_L*d_R/(d_L+d_R) if same sign, else 0
        superbee: m = maxmod(minmod(d_L, 2*d_R), minmod(2*d_L, d_R))
        """
        if flux.slopes is None or len(flux.values) < 3:
            return flux

        n = len(flux.values)
        bp = flux.breakpoints
        new_slopes = list(flux.slopes)

        for i in range(1, n - 1):
            h_left = bp[i] - bp[i - 1]
            h_right = bp[i + 1] - bp[i]
            d_left = (flux.values[i] - flux.values[i - 1]) / max(h_left, 1e-30)
            d_right = (flux.values[i + 1] - flux.values[i]) / max(h_right, 1e-30)

            if limiter == "minmod":
                if d_left * d_right <= 0:
                    new_slopes[i] = 0.0
                elif abs(d_left) < abs(d_right):
                    new_slopes[i] = d_left
                else:
                    new_slopes[i] = d_right
            elif limiter == "van_leer":
                if d_left * d_right <= 0:
                    new_slopes[i] = 0.0
                else:
                    new_slopes[i] = 2.0 * d_left * d_right / (d_left + d_right)
            elif limiter == "superbee":
                s1 = self._minmod(d_left, 2.0 * d_right)
                s2 = self._minmod(2.0 * d_left, d_right)
                if abs(s1) > abs(s2):
                    new_slopes[i] = s1
                else:
                    new_slopes[i] = s2

        return PiecewiseFlux(
            breakpoints=flux.breakpoints,
            values=flux.values,
            slopes=new_slopes,
        )

    def _minmod(self, a: float, b: float) -> float:
        if a * b <= 0:
            return 0.0
        return a if abs(a) < abs(b) else b
