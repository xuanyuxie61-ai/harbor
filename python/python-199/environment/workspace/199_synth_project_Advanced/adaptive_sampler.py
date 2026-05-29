"""
adaptive_sampler.py — 自适应采样与分区估计模块
===============================================
融合来源:
  - 541_histogram_pdf_sample (直方图PDF采样与逆CDF)
  - 1014_rbf_interp_2d (RBF径向基函数插值)
  - 397_fem1d_project (一维有限元Galerkin投影)
  - 889_polygon_sample (多边形三角剖分与面积加权采样)

在外排序的采样阶段，需要估计全局数据分布以确定均衡分区边界。
本模块提供：
  1. 直方图密度估计 + 逆CDF采样（快速分位数估计）
  2. RBF全局插值（平滑分布估计）
  3. FEM投影（Galerkin意义下的最优分位数逼近）
  4. 多边形分区（二维及以上键空间的几何剖分）
"""

import math
import random
from typing import List, Tuple, Optional


class HistogramPDFSampler:
    """
    基于直方图的PDF离散化与逆变换采样。

    算法流程：
        1. 将数据范围 [x_min, x_max] 划分为 B 个等宽区间
        2. 统计各区间频数，归一化得到直方图概率 p_i
        3. 构造累积分布函数 CDF：
               C_j = Σ_{i=1}^{j} p_i · Δx_i
        4. 逆CDF采样：生成 U ~ Uniform(0,1)，查找满足 C_{j-1} ≤ U < C_j 的区间 j，
           然后在区间内线性插值

    该方法是外排序中确定分区边界的经典技术，时间复杂度 O(N + B)。
    """

    def __init__(self, data: List[float], num_bins: int = 256):
        if not data:
            raise ValueError("Data must not be empty.")
        self.data = list(data)
        self.n = len(data)
        self.num_bins = max(num_bins, 4)
        self.x_min = min(data)
        self.x_max = max(data)
        if self.x_max <= self.x_min:
            self.x_max = self.x_min + 1.0
        self.bin_width = (self.x_max - self.x_min) / self.num_bins
        self._build_histogram()

    def _build_histogram(self):
        self.counts = [0] * self.num_bins
        for x in self.data:
            idx = int((x - self.x_min) / self.bin_width)
            idx = min(idx, self.num_bins - 1)
            self.counts[idx] += 1
        total = sum(self.counts)
        if total == 0:
            total = 1
        # 概率密度函数值（每单位长度概率）
        self.pdf = [c / (total * self.bin_width) for c in self.counts]
        # 累积概率
        self.cdf = [0.0] * (self.num_bins + 1)
        for i in range(self.num_bins):
            self.cdf[i + 1] = self.cdf[i] + self.counts[i] / total

    def quantile(self, q: float) -> float:
        """
        估计分位数 Q(q) = inf{ x : F(x) ≥ q }，其中 q ∈ [0,1]。

        通过逆CDF方法：在累积概率数组中二分查找，再线性插值。
        """
        if q <= 0.0:
            return self.x_min
        if q >= 1.0:
            return self.x_max
        # 线性搜索（bin数量通常不大）
        for i in range(self.num_bins):
            if self.cdf[i] <= q <= self.cdf[i + 1]:
                # 线性插值
                if self.cdf[i + 1] > self.cdf[i]:
                    frac = (q - self.cdf[i]) / (self.cdf[i + 1] - self.cdf[i])
                else:
                    frac = 0.5
                return self.x_min + (i + frac) * self.bin_width
        return self.x_max

    def partition_boundaries(self, num_partitions: int) -> List[float]:
        """
        生成 num_partitions 个均衡分区的边界。

        目标：每个分区包含大致相等的数据量（1/num_partitions）。
        边界点为 q_i = i / num_partitions 对应的分位数，i = 0, ..., num_partitions。
        """
        bounds = []
        for i in range(num_partitions + 1):
            q = i / num_partitions
            bounds.append(self.quantile(q))
        # 单调性修正
        for i in range(1, len(bounds)):
            if bounds[i] < bounds[i - 1]:
                bounds[i] = bounds[i - 1]
        return bounds

    def sample(self, n_samples: int, seed: int = 0) -> List[float]:
        """
        从估计的PDF中采样n_samples个点（用于验证分布拟合度）。
        """
        random.seed(seed)
        samples = []
        for _ in range(n_samples):
            u = random.random()
            samples.append(self.quantile(u))
        return samples


class RBFInterpolator:
    """
    径向基函数（RBF）全局插值器。

    对散乱数据点 {x_i, f_i}，求解权重 w 使得：
        f(x) = Σ_{j=1}^{N} w_j · φ( ||x - x_j|| / r0 )

    支持的核函数：
        φ1(r) = exp(-r^2/2)              (高斯)
        φ2(r) = 1 / sqrt(1 + r^2)        (逆多二次)
        φ3(r) = r^2 · log(r)             (薄板样条)
        φ4(r) = sqrt(1 + r^2)            (多二次)

    权重通过解线性系统 A · w = f 获得，其中 A_{ij} = φ(||x_i - x_j||/r0)。
    """

    def __init__(self, x_data: List[float], f_data: List[float],
                 r0: float = 1.0, kernel_type: int = 1):
        if len(x_data) != len(f_data):
            raise ValueError("x_data and f_data must have the same length.")
        if len(x_data) < 2:
            raise ValueError("Need at least 2 data points.")
        self.xd = list(x_data)
        self.fd = list(f_data)
        self.nd = len(x_data)
        self.r0 = max(r0, 1e-10)
        self.kernel_type = kernel_type
        self._solve_weights()

    def _phi(self, r: float) -> float:
        if self.kernel_type == 1:
            # Gaussian
            return math.exp(-0.5 * r * r)
        elif self.kernel_type == 2:
            # Inverse multiquadric
            return 1.0 / math.sqrt(1.0 + r * r)
        elif self.kernel_type == 3:
            # Thin plate spline
            if r < 1e-12:
                return 0.0
            return r * r * math.log(r)
        else:
            # Multiquadric
            return math.sqrt(1.0 + r * r)

    def _solve_weights(self):
        """
        用高斯消元法（带部分主元）求解稠密线性系统。
        """
        n = self.nd
        # 构造插值矩阵 A
        A = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                r = abs(self.xd[i] - self.xd[j]) / self.r0
                A[i][j] = self._phi(r)

        # 部分主元高斯消元
        w = list(self.fd)
        pivot = list(range(n))
        for col in range(n):
            # 选主元
            max_row = col
            max_val = abs(A[pivot[col]][col])
            for row in range(col + 1, n):
                if abs(A[pivot[row]][col]) > max_val:
                    max_val = abs(A[pivot[row]][col])
                    max_row = row
            pivot[col], pivot[max_row] = pivot[max_row], pivot[col]

            piv = pivot[col]
            if abs(A[piv][col]) < 1e-15:
                raise ValueError("RBF interpolation matrix is singular or ill-conditioned.")

            # 消元
            for row in range(col + 1, n):
                pr = pivot[row]
                factor = A[pr][col] / A[piv][col]
                for c in range(col, n):
                    A[pr][c] -= factor * A[piv][c]
                w[pr] -= factor * w[piv]

        # 回代
        self.weights = [0.0] * n
        for i in range(n - 1, -1, -1):
            piv = pivot[i]
            s = w[piv]
            for j in range(i + 1, n):
                s -= A[piv][j] * self.weights[j]
            self.weights[i] = s / A[piv][i]

    def evaluate(self, x: float) -> float:
        """
        在点 x 处评估RBF插值。
        """
        result = 0.0
        for j in range(self.nd):
            r = abs(x - self.xd[j]) / self.r0
            result += self.weights[j] * self._phi(r)
        return result


class FEM1DProjector:
    """
    一维有限元Galerkin投影。

    将采样得到的分区估计投影到分段线性有限元空间 V_h，求解：
        找到 u_h ∈ V_h 使得 (u_h, v_h) = (f, v_h),  ∀ v_h ∈ V_h

    即求解质量矩阵系统 M · u = b，其中：
        M_{ij} = ∫ φ_i · φ_j dx
        b_i    = ∫ f · φ_i dx

    采用两点Gauss-Legendre数值积分（精确到三次多项式）：
        ξ = ±1/√3,  w = 1
    """

    def __init__(self, nodes: List[float]):
        if len(nodes) < 2:
            raise ValueError("Need at least 2 nodes.")
        self.nodes = sorted(nodes)
        self.n_nodes = len(nodes)

    def _basis(self, x: float, xl: float, xr: float) -> Tuple[float, float]:
        """
        分段线性基函数在单元 [xl, xr] 上的取值：
            φ_L(x) = (xr - x) / (xr - xl)
            φ_R(x) = (x - xl) / (xr - xl)
        """
        h = xr - xl
        if h < 1e-15:
            return 1.0, 0.0
        phil = (xr - x) / h
        phir = (x - xl) / h
        return phil, phir

    def project(self, f_func) -> List[float]:
        """
        对给定函数 f_func 进行Galerkin投影，返回节点值 u。
        """
        n = self.n_nodes
        # 组装质量矩阵 M（三对角）
        M = [[0.0] * n for _ in range(n)]
        b = [0.0] * n

        # Gauss-Legendre 两点
        xi = [-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0)]
        wg = [1.0, 1.0]

        for e in range(n - 1):
            xl, xr = self.nodes[e], self.nodes[e + 1]
            h = xr - xl
            mid = 0.5 * (xl + xr)

            for q in range(2):
                xq = mid + 0.5 * h * xi[q]
                wq = 0.5 * h * wg[q]
                phil, phir = self._basis(xq, xl, xr)
                fv = f_func(xq)

                # 局部组装到全局
                M[e][e]     += wq * phil * phil
                M[e][e+1]   += wq * phil * phir
                M[e+1][e]   += wq * phir * phil
                M[e+1][e+1] += wq * phir * phir
                b[e]        += wq * fv * phil
                b[e+1]      += wq * fv * phir

        # 解三对角系统（Thomas算法）
        return self._thomas_solve(M, b)

    def _thomas_solve(self, M: List[List[float]], b: List[float]) -> List[float]:
        """
        Thomas算法求解三对角线性系统，O(n)复杂度。
        """
        n = len(b)
        # 提取下对角、主对角、上对角
        lower = [0.0] * n
        diag  = [0.0] * n
        upper = [0.0] * n
        for i in range(n):
            diag[i] = M[i][i]
            if i > 0:
                lower[i] = M[i][i-1]
            if i < n - 1:
                upper[i] = M[i][i+1]

        # 前向消元
        cp = [0.0] * n
        dp = [0.0] * n
        cp[0] = upper[0] / diag[0] if abs(diag[0]) > 1e-15 else 0.0
        dp[0] = b[0] / diag[0] if abs(diag[0]) > 1e-15 else 0.0
        for i in range(1, n):
            denom = diag[i] - lower[i] * cp[i-1]
            if abs(denom) < 1e-15:
                denom = 1e-15
            cp[i] = upper[i] / denom if i < n - 1 else 0.0
            dp[i] = (b[i] - lower[i] * dp[i-1]) / denom

        # 回代
        x = [0.0] * n
        x[-1] = dp[-1]
        for i in range(n - 2, -1, -1):
            x[i] = dp[i] - cp[i] * x[i+1]
        return x


class PolygonPartitionSampler:
    """
    多边形三角剖分与面积加权采样。

    对于二维及以上键空间，将分区边界建模为多边形，进行耳切法
    三角剖分后按面积加权随机选择三角形，在三角形内用重心坐标
    生成均匀随机点。

    二维键值 (k1, k2) 的分区边界可用凸多边形近似，面积加权采样
    保证各子区域采样概率与其数据密度成正比。
    """

    def __init__(self, vertices: List[Tuple[float, float]]):
        if len(vertices) < 3:
            raise ValueError("Polygon must have at least 3 vertices.")
        self.vertices = vertices
        self.triangles = self._ear_clipping_triangulation()

    def _triangle_area(self, a: Tuple[float, float], b: Tuple[float, float],
                       c: Tuple[float, float]) -> float:
        return 0.5 * abs(
            a[0]*(b[1]-c[1]) + b[0]*(c[1]-a[1]) + c[0]*(a[1]-b[1])
        )

    def _is_convex(self, prev: int, curr: int, next_idx: int) -> bool:
        n = len(self.vertices)
        a = self.vertices[prev % n]
        b = self.vertices[curr % n]
        c = self.vertices[next_idx % n]
        cross = (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
        return cross > 0

    def _ear_clipping_triangulation(self) -> List[Tuple[int, int, int]]:
        """
        简化的耳切法三角剖分（假设多边形为简单多边形且顶点逆时针排列）。
        """
        n = len(self.vertices)
        if n == 3:
            return [(0, 1, 2)]

        indices = list(range(n))
        triangles = []
        while len(indices) > 3:
            m = len(indices)
            found = False
            for i in range(m):
                prev = indices[(i - 1) % m]
                curr = indices[i]
                next_idx = indices[(i + 1) % m]
                if self._is_convex(prev, curr, next_idx):
                    triangles.append((prev, curr, next_idx))
                    indices.pop(i)
                    found = True
                    break
            if not found:
                # 回退：直接取前三个
                triangles.append((indices[0], indices[1], indices[2]))
                indices.pop(1)
        triangles.append((indices[0], indices[1], indices[2]))
        return triangles

    def sample(self, n_samples: int, seed: int = 0) -> List[Tuple[float, float]]:
        """
        在多边形内生成 n_samples 个均匀分布点。
        """
        random.seed(seed)
        areas = []
        for t in self.triangles:
            a = self.vertices[t[0]]
            b = self.vertices[t[1]]
            c = self.vertices[t[2]]
            areas.append(self._triangle_area(a, b, c))
        total_area = sum(areas)
        if total_area < 1e-15:
            return [self.vertices[0]] * n_samples

        # 按面积累积分布选择三角形
        cum = [0.0]
        for ar in areas:
            cum.append(cum[-1] + ar / total_area)

        samples = []
        for _ in range(n_samples):
            u = random.random()
            idx = 0
            for i in range(len(cum) - 1):
                if cum[i] <= u <= cum[i + 1]:
                    idx = i
                    break
            t = self.triangles[idx]
            a = self.vertices[t[0]]
            b = self.vertices[t[1]]
            c = self.vertices[t[2]]
            r1 = random.random()
            r2 = random.random()
            if r1 + r2 > 1.0:
                r1 = 1.0 - r1
                r2 = 1.0 - r2
            px = a[0] + r1 * (b[0] - a[0]) + r2 * (c[0] - a[0])
            py = a[1] + r1 * (b[1] - a[1]) + r2 * (c[1] - a[1])
            samples.append((px, py))
        return samples


def adaptive_partition_estimation(
    sample_data: List[float],
    num_partitions: int,
    use_rbf_refinement: bool = True
) -> List[float]:
    """
    自适应分区边界估计主函数。

    流程：
        1. 直方图采样得到粗粒度分位数边界
        2. 若 use_rbf_refinement=True，用RBF插值平滑CDF并重新估计边界
        3. 返回最终分区边界
    """
    # 步骤1：直方图分位数
    hist_sampler = HistogramPDFSampler(sample_data, num_bins=max(num_partitions * 4, 64))
    coarse_bounds = hist_sampler.partition_boundaries(num_partitions)

    if not use_rbf_refinement or len(sample_data) < 20:
        return coarse_bounds

    # 步骤2：RBF平滑CDF
    # 在粗边界上采样CDF值
    x_rbf = coarse_bounds
    y_rbf = [hist_sampler.quantile(q) for q in [i / len(coarse_bounds) for i in range(len(coarse_bounds))]]
    # 简化：直接使用直方图CDF的采样点
    x_rbf = [hist_sampler.x_min + i * (hist_sampler.x_max - hist_sampler.x_min) / 50.0 for i in range(51)]
    y_rbf = [hist_sampler.quantile(i / 50.0) for i in range(51)]

    try:
        rbf = RBFInterpolator(x_rbf, y_rbf, r0=(hist_sampler.x_max - hist_sampler.x_min) / 10.0, kernel_type=1)
        refined = [rbf.evaluate(hist_sampler.x_min + q * (hist_sampler.x_max - hist_sampler.x_min))
                   for q in [i / num_partitions for i in range(num_partitions + 1)]]
        # 单调性修正
        for i in range(1, len(refined)):
            if refined[i] < refined[i - 1]:
                refined[i] = refined[i - 1]
        return refined
    except Exception:
        return coarse_bounds
