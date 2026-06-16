"""
chaos_analysis.py — 簇射方程混沌特性分析
==========================================

融合种子项目:
  - 318_dragon_chaos : 迭代函数系统 (IFS) 混沌吸引子

本模块分析级联方程的混沌特性:

1. Lyapunov 指数:
   lambda = lim_{t->inf} (1/t) * ln(|delta_x(t)| / |delta_x(0)|)
   对于级联矩阵 M:
     lambda_i = Re(eigenvalue_i(M))

2. 分岔分析:
   研究参数 sigma_b/sigma_p 变化时系统行为的转变
   临界值: sigma_b/sigma_p = 1 → 粒子数守恒
           sigma_b/sigma_p > 1 → 指数增长
           sigma_b/sigma_p < 1 → 衰减

3. 庞加莱回归:
   检查系统是否回归初始状态附近
   回归时间 T_rec ~ exp(h / lambda_max)  (其中 h 为 Kolmogorov 熵)

4. 分形维数:
   簇射横向分布的分形特性
   D_f = lim_{r->0} ln(N(r)) / ln(1/r)
   对于 Molière 分布: D_f ≈ 1.5-2.0

5. IFS 吸引子 (Dragon 方法映射):
   将级联的随机分支过程视为 IFS:
   x_{n+1} = A_i * x_n + b_i
   吸引子的 Hausdorff 维数: sum_i r_i^D = 1
"""

import math
from typing import List, Tuple, Optional
from material_properties import MaterialSpec


class ChaosAnalyzer:
    """级联方程混沌特性分析器"""

    def __init__(self, material: MaterialSpec):
        self.mat = material

    def compute_lyapunov_exponents(
        self,
        sigma_b: float,
        sigma_p: float,
        sigma_ion: float,
    ) -> Tuple[List[float], float]:
        """
        计算 Lyapunov 指数谱

        对于线性化级联方程:
          d/dt [phi_e, phi_gamma]^T = M * [phi_e, phi_gamma]^T

        M = [[-(sigma_b + sigma_ion), 2*sigma_p],
             [sigma_a,                -sigma_a    ]]

        Lyapunov 指数 = Re(eigenvalues(M))

        特征值:
          lambda = trace(M)/2 ± sqrt(trace(M)^2/4 - det(M))
        """
        trace_M = -(sigma_b + sigma_ion) - sigma_b  # 对角线之和
        det_M = (sigma_b + sigma_ion) * sigma_b - 2.0 * sigma_p * sigma_b

        discriminant = trace_M * trace_M / 4.0 - det_M

        if discriminant >= 0:
            sqrt_disc = math.sqrt(discriminant)
            lambda1 = trace_M / 2.0 + sqrt_disc
            lambda2 = trace_M / 2.0 - sqrt_disc
        else:
            sqrt_disc = math.sqrt(-discriminant)
            lambda1 = trace_M / 2.0
            lambda2 = trace_M / 2.0

        lyap_exponents = sorted([lambda1, lambda2], reverse=True)

        # 最大 Lyapunov 指数
        lambda_max = lyap_exponents[0]

        return lyap_exponents, lambda_max

    def bifurcation_analysis(
        self,
        sigma_p: float,
        sigma_ion: float,
        n_points: int = 50,
        ratio_range: Tuple[float, float] = (0.1, 3.0),
    ) -> List[Tuple[float, str, float]]:
        """
        分岔分析: 研究 sigma_b/sigma_p 变化时的行为

        分类:
          lambda_max < 0: "stable" (衰减)
          lambda_max = 0: "critical" (分岔点)
          lambda_max > 0: "unstable" (指数增长)
          |Im(lambda)| > 0: "oscillatory" (振荡)

        返回: [(ratio, regime, lambda_max), ...]
        """
        results = []
        r_min, r_max = ratio_range

        for i in range(n_points + 1):
            ratio = r_min + (r_max - r_min) * i / n_points
            sigma_b = ratio * sigma_p

            lyap, lambda_max = self.compute_lyapunov_exponents(
                sigma_b, sigma_p, sigma_ion,
            )

            trace_M = -(sigma_b + sigma_ion) - sigma_b
            det_M = (sigma_b + sigma_ion) * sigma_b - 2.0 * sigma_p * sigma_b
            disc = trace_M * trace_M / 4.0 - det_M

            if abs(lambda_max) < 1e-6:
                regime = "critical"
            elif lambda_max > 0:
                regime = "unstable"
            elif disc < 0:
                regime = "oscillatory_decay"
            else:
                regime = "stable"

            results.append((ratio, regime, lambda_max))

        return results

    def compute_fractal_dimension(
        self,
        lateral_profile: List[Tuple[float, float]],
        n_radii: int = 20,
    ) -> float:
        """
        计算横向分布的分形维数 (盒计数法)

        D_f = lim_{r->0} d(ln N(r)) / d(ln(1/r))

        其中 N(r) 为覆盖分布所需的半径为 r 的盒子数。

        对于 shower 横向分布:
          以 r 为尺度, N(r) = total_energy_inside(r) / energy_per_box(r)
        """
        if not lateral_profile:
            return 2.0

        # 按半径排序
        sorted_profile = sorted(lateral_profile, key=lambda x: x[0])
        total_energy = sum(e for _, e in sorted_profile)
        if total_energy < 1e-30:
            return 2.0

        # 累积能量
        cum_energy = []
        running = 0.0
        for r, e in sorted_profile:
            running += e
            cum_energy.append((r, running))

        # 不同尺度的盒子数
        max_r = sorted_profile[-1][0] if sorted_profile else 1.0
        if max_r < 1e-10:
            return 2.0

        dimensions = []
        for i in range(1, n_radii):
            r_scale = max_r * i / n_radii
            # 计算覆盖到 r_scale 的盒子数
            n_boxes = 0
            for r, cum_e in cum_energy:
                if r <= r_scale:
                    # 每个盒子面积 ~ r_scale^2
                    box_energy = total_energy * (r_scale / max_r) ** 2
                    if box_energy > 1e-30:
                        n_boxes += cum_e / box_energy

            if n_boxes > 1 and r_scale > 0:
                # 局部斜率估计
                d_ln_N = math.log(max(n_boxes, 1.0))
                d_ln_r = math.log(max_r / r_scale)
                if d_ln_r > 1e-10:
                    D_local = d_ln_N / d_ln_r
                    dimensions.append(D_local)

        if not dimensions:
            return 2.0

        # 取平均 (对数尺度中段的值最可靠)
        mid_start = len(dimensions) // 4
        mid_end = 3 * len(dimensions) // 4
        if mid_end <= mid_start:
            return sum(dimensions) / len(dimensions)
        return sum(dimensions[mid_start:mid_end]) / (mid_end - mid_start)

    def compute_recurrence_time(
        self,
        lambda_max: float,
        target_radius: float = 0.01,
        phase_space_dim: int = 2,
    ) -> float:
        """
        计算庞加莱回归时间

        T_rec ≈ (1/lambda_max) * (phase_radius / target_radius)^{d-1}

        其中 d 为相空间维数, phase_radius 为典型轨道半径。
        """
        if lambda_max <= 0:
            return float('inf')

        phase_radius = 1.0  # 归一化
        ratio = phase_radius / max(target_radius, 1e-15)

        if phase_space_dim <= 1:
            return math.log(ratio) / lambda_max
        else:
            return (ratio ** (phase_space_dim - 1)) / lambda_max

    def ifs_attractor_dimension(
        self,
        contraction_ratios: List[float],
        tolerance: float = 1e-8,
    ) -> float:
        """
        计算 IFS 吸引子的 Hausdorff 维数

        对于相似比 {r_1, ..., r_k} 的 IFS:
          sum_i r_i^D = 1  →  D = Hausdorff 维数

        求解: f(D) = sum_i r_i^D - 1 = 0
        使用二分法。
        """
        if not contraction_ratios:
            return 0.0

        # 确保 r_i < 1
        valid_ratios = [r for r in contraction_ratios if 0 < r < 1]
        if not valid_ratios:
            return 0.0

        # 检查 sum(r_i) 是否 < 1 (开集条件)
        if sum(valid_ratios) >= len(valid_ratios):
            # 可能需要更高维分析
            return float(len(valid_ratios))

        # 二分法求解 sum(r_i^D) = 1
        D_low, D_high = 0.0, 10.0

        # 确保边界正确
        def f(D):
            return sum(r ** D for r in valid_ratios) - 1.0

        if f(D_low) < 0:
            D_low = -1.0  # 扩展搜索范围
        if f(D_high) > 0:
            D_high = 20.0

        for _ in range(100):
            D_mid = (D_low + D_high) / 2.0
            if f(D_mid) > 0:
                D_low = D_mid
            else:
                D_high = D_mid
            if D_high - D_low < tolerance:
                break

        return (D_low + D_high) / 2.0

    def kolmogorov_entropy(
        self, lyapunov_exponents: List[float],
    ) -> float:
        """
        计算 Kolmogorov-Sinai 熵 (Pesin 公式)

        h_KS = sum_{lambda_i > 0} lambda_i

        物理含义: 信息产生率 (bits/time)
        """
        return sum(lam for lam in lyapunov_exponents if lam > 0)

    def correlation_dimension(
        self, time_series: List[float],
        max_embedding_dim: int = 6,
        n_radii: int = 20,
    ) -> float:
        """
        计算关联维数 (Grassberger-Procaccia 算法)

        C(r) = (2 / (N*(N-1))) * sum_{i<j} H(r - |x_i - x_j|)
        D_2 = lim_{r->0} d(ln C(r)) / d(ln r)

        其中 H 为 Heaviside 阶跃函数。
        """
        N = len(time_series)
        if N < 10:
            return 2.0

        # 时间延迟嵌入
        m = min(max_embedding_dim, N // 4)
        tau = 1

        # 构建嵌入向量
        n_vectors = N - (m - 1) * tau
        if n_vectors < 5:
            return 2.0

        vectors = []
        for i in range(n_vectors):
            v = [time_series[i + k * tau] for k in range(m)]
            vectors.append(v)

        # 计算距离矩阵 (仅上三角)
        distances = []
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                d = math.sqrt(sum((vectors[i][k] - vectors[j][k]) ** 2
                                 for k in range(m)))
                distances.append(d)

        if not distances:
            return 2.0

        distances.sort()
        max_d = distances[-1] if distances else 1.0
        if max_d < 1e-15:
            return 0.0

        # 计算 C(r) 在不同 r 值
        log_r = []
        log_C = []

        for i in range(1, n_radii + 1):
            r = max_d * i / (n_radii + 1)
            n_pairs_within = sum(1 for d in distances if d <= r)
            C_r = 2.0 * n_pairs_within / (len(vectors) * (len(vectors) - 1))

            if C_r > 1e-15 and r > 1e-15:
                log_r.append(math.log(r))
                log_C.append(math.log(C_r))

        if len(log_r) < 3:
            return 2.0

        # 线性回归: ln C = D_2 * ln r + const
        n = len(log_r)
        sum_x = sum(log_r)
        sum_y = sum(log_C)
        sum_xy = sum(x * y for x, y in zip(log_r, log_C))
        sum_x2 = sum(x * x for x in log_r)

        denom = n * sum_x2 - sum_x * sum_x
        if abs(denom) < 1e-30:
            return 2.0

        D2 = (n * sum_xy - sum_x * sum_y) / denom
        return max(0.0, min(float(m), D2))
