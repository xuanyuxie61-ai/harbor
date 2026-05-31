"""
特殊数学工具模块
==================
融合种子项目:
  - 328_ellipse   : 椭圆几何与椭圆积分
  - 1430_zero_laguerre : Laguerre多项式求根法
  - 915_prime_plot     : 素性检测
  - 1377_usa_box_plot  : 统计分位数分析（去除可视化）
  - 1168_stla_to_tri_surface_fast : 快速数据解析与结构化处理
  - 382_fem_to_xml     : 网格数据结构化管理

在金融工程中，本模块提供：
1. 椭圆积分与椭圆几何 —— 用于Heston模型中Feller条件边界分析
2. Laguerre求根法 —— 用于特征函数多项式求根与正交展开
3. 素性检测 —— 用于伪随机数长周期素数模选择
4. 分位数统计 —— 用于蒙特卡洛输出的风险度量(VaR, CVaR)
5. 快速数据解析 —— 用于大规模市场数据的结构化读取
6. 网格数据管理 —— 用于有限差分/有限元网格节点-单元关系管理
"""

import numpy as np
import cmath
from math import sqrt, pi, exp, log, gcd


# ========================================================================
# 328_ellipse : 椭圆几何与椭圆积分
# ========================================================================

def ellipse_perimeter_ramanujan(a, b):
    """
    使用Ramanujan近似公式计算椭圆周长。

    对于半轴为 a, b 的椭圆，精确周长涉及第二类完全椭圆积分 E(e)：
        P = 4a E(e),   e = √(1 - b²/a²)  (离心率)

    Ramanujan一级近似:
        P ≈ π [ 3(a+b) - √((3a+b)(a+3b)) ]

    参数:
    ------
    a, b : float, 椭圆半轴长度

    返回:
    ------
    float, 椭圆周长近似值
    """
    a = abs(float(a))
    b = abs(float(b))
    if a < b:
        a, b = b, a
    if a == 0:
        return 0.0
    # Ramanujan近似
    h = ((a - b) / (a + b)) ** 2
    approx = pi * (a + b) * (1.0 + 3.0 * h / (10.0 + sqrt(4.0 - 3.0 * h)))
    return approx


def ellipse_area_matrix(A, r):
    """
    由正定对称矩阵A定义的椭圆区域面积。

    满足 X^T A X ≤ r² 的点集X构成椭圆，其面积为:
        Area = π r² / √(det(A))

    参数:
    ------
    A : 2x2 正定对称矩阵
    r : float, "半径"参数

    返回:
    ------
    float, 椭圆面积
    """
    A = np.asarray(A, dtype=np.float64)
    if A.shape != (2, 2):
        raise ValueError("A必须为2x2矩阵")
    detA = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    if detA <= 0:
        raise ValueError("A必须正定(det>0)")
    return pi * r * r / sqrt(detA)


def complete_elliptic_integral_second_kind(k):
    """
    使用算术几何平均(AGM)算法计算第二类完全椭圆积分 E(k)。

    定义:
        E(k) = ∫₀^{π/2} √(1 - k² sin²θ) dθ

    AGM迭代:
        a₀ = 1,   b₀ = √(1-k²),   c₀ = k
        a_{n+1} = (a_n + b_n)/2
        b_{n+1} = √(a_n b_n)
        c_{n+1} = (a_n - b_n)/2

    则:
        E(k) = (π/2) · (a_N² - Σ_{j=0}^{N-1} 2^{j+1} c_j²) / a_N

    参数:
    ------
    k : float, 模数, 0 ≤ k ≤ 1

    返回:
    ------
    float, E(k)
    """
    k = float(k)
    if k < 0.0 or k > 1.0:
        raise ValueError("模数k必须在[0,1]区间内")
    if k == 0.0:
        return pi / 2.0
    if k == 1.0:
        return 1.0

    a, b, c = 1.0, sqrt(1.0 - k * k), k
    sum_c2 = 0.0
    two_pow = 1.0
    for _ in range(50):
        if abs(c) < 1e-15:
            break
        two_pow *= 2.0
        sum_c2 += two_pow * c * c
        a_next = (a + b) * 0.5
        b_next = sqrt(a * b)
        c = (a - b) * 0.5
        a, b = a_next, b_next

    return (pi / 2.0) * (a * a - sum_c2) / a


# ========================================================================
# 1430_zero_laguerre : Laguerre求根法
# ========================================================================

def laguerre_rootfind(f, x0, degree, abserr=1e-12, kmax=100):
    """
    Laguerre方法求多项式根。

    迭代公式:
        z = (f')² - (β+1) f f''
        dx = -(β+1) f / (β f' + sign(f')·√z)
        x_{k+1} = x_k + dx

    其中 β = 1/(degree-1)。

    参数:
    ------
    f      : callable, f(x) 返回 (value, dvalue, ddvalue)
    x0     : float, 初始猜测
    degree : int, 多项式次数(≥2)
    abserr : float, 误差容限
    kmax   : int, 最大迭代次数

    返回:
    ------
    x      : float, 估计根
    ierror : int, 0表示成功
    k      : int, 实际迭代次数
    """
    if degree < 2:
        raise ValueError("degree必须≥2")
    x = float(x0)
    beta = 1.0 / (degree - 1.0)
    ierror = 0
    k = 0
    while True:
        fx, dfx, d2fx = f(x)
        if abs(fx) <= abserr:
            break
        k += 1
        if k > kmax:
            ierror = 2
            return x, ierror, k
        z = dfx * dfx - (beta + 1.0) * fx * d2fx
        z = max(z, 0.0)
        bot = beta * dfx + sqrt(z)
        if abs(bot) < 1e-30:
            ierror = 3
            return x, ierror, k
        dx = -(beta + 1.0) * fx / bot
        x += dx
    return x, ierror, k


def heston_characteristic_root(v0, kappa, theta, sigma, rho, u, T):
    """
    使用Laguerre方法求解Heston特征函数的复平面驻点。

    Heston特征函数中的关键量:
        d(u) = √((ρσui - κ)² + σ²(ui + u²))

    本函数寻找使特征函数相位稳定的驻点，用于快速傅里叶反演中的
    最优衰减轮廓选择。
    """
    def phase_func(x):
        # x为实变量，计算特征函数对数的实部及其导数
        d = cmath.sqrt((rho * sigma * 1j * x - kappa)**2 + sigma**2 * (1j * x + x*x))
        d = complex(d)
        val = kappa * theta / (sigma**2) * ((kappa - rho*sigma*1j*x - d)*T
               - 2.0 * cmath.log((1.0 - ((kappa - rho*sigma*1j*x - d)/(kappa - rho*sigma*1j*x + d))*cmath.exp(-d*T))
               / (1.0 - (kappa - rho*sigma*1j*x - d)/(kappa - rho*sigma*1j*x + d))))
        # 数值微分近似
        h = 1e-8
        d_val = (phase_func_real(x+h) - phase_func_real(x-h)) / (2*h)
        dd_val = (phase_func_real(x+h) - 2*phase_func_real(x) + phase_func_real(x-h)) / (h*h)
        return phase_func_real(x), d_val, dd_val

    def phase_func_real(x):
        d = cmath.sqrt((rho * sigma * 1j * x - kappa)**2 + sigma**2 * (1j * x + x*x))
        d = complex(d)
        if abs(d) < 1e-12:
            return 0.0
        g = (kappa - rho*sigma*1j*x - d) / (kappa - rho*sigma*1j*x + d)
        if abs(1.0 - g*cmath.exp(-d*T)) < 1e-12 or abs(1.0 - g) < 1e-12:
            return 0.0
        A = kappa*theta/(sigma**2)*((kappa - rho*sigma*1j*x - d)*T
            - 2.0*cmath.log((1.0 - g*cmath.exp(-d*T))/(1.0 - g)))
        return A.real

    x, ierr, iters = laguerre_rootfind(phase_func, 0.5, 6, abserr=1e-10, kmax=80)
    return x, ierr, iters


# ========================================================================
# 915_prime_plot : 素性检测与素数生成
# ========================================================================

def is_prime(n):
    """Miller-Rabin素性测试。用于选择伪随机数长周期模数。"""
    if n < 2:
        return False
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    for p in small_primes:
        if n % p == 0:
            return n == p
    # 将 n-1 写成 d·2^s
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    # 测试轮数
    for a in [2, 325, 9375, 28178, 450775, 9780504, 1795265022]:
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True


def next_prime(n):
    """查找大于等于n的最小素数。"""
    if n <= 2:
        return 2
    if n % 2 == 0:
        n += 1
    while not is_prime(n):
        n += 2
        if n > 1e12:
            raise RuntimeError("无法在给定范围内找到素数")
    return n


# ========================================================================
# 1377_usa_box_plot : 统计分位数与风险度量（去除可视化）
# ========================================================================

def quantile_statistics(data, probs=None):
    """
    计算数据的分位数统计量，用于VaR/CVaR风险度量。

    参数:
    ------
    data  : array_like, 样本数据
    probs : list, 分位概率，默认 [0.01, 0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99]

    返回:
    ------
    dict, 包含各分位数、VaR(99%)、CVaR(99%)、偏度、峰度等
    """
    data = np.asarray(data, dtype=np.float64)
    if data.size == 0:
        raise ValueError("数据不能为空")
    if probs is None:
        probs = [0.01, 0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99]
    data_sorted = np.sort(data)
    n = len(data_sorted)
    result = {}
    for p in probs:
        idx = int(p * (n - 1))
        result[f'q{int(p*100):02d}'] = data_sorted[idx]
    # VaR at 99%
    result['VaR99'] = data_sorted[int(0.01 * (n - 1))]
    # CVaR (Expected Shortfall) at 99%
    tail = data_sorted[:int(0.01 * n) + 1]
    result['CVaR99'] = np.mean(tail)
    # 统计量
    result['mean'] = np.mean(data)
    result['std'] = np.std(data, ddof=1)
    result['skewness'] = np.mean(((data - result['mean']) / result['std'])**3) if result['std'] > 0 else 0.0
    result['kurtosis'] = np.mean(((data - result['mean']) / result['std'])**4) if result['std'] > 0 else 0.0
    # 异常值检测 (IQR方法)
    q25 = result['q25']
    q75 = result['q75']
    iqr = q75 - q25
    lower = q25 - 1.5 * iqr
    upper = q75 + 1.5 * iqr
    outliers = data[(data < lower) | (data > upper)]
    result['outlier_count'] = len(outliers)
    result['outlier_ratio'] = len(outliers) / n
    return result


def box_plot_summary(data):
    """生成箱线图统计摘要（纯数值，无图形）。"""
    return quantile_statistics(data)


# ========================================================================
# 1168_stla_to_tri_surface_fast : 快速数据解析与结构化处理
# ========================================================================

def fast_structured_parse(lines, keyword_map):
    """
    模仿STLA快速解析思想，对按关键字分区的文本数据进行结构化解析。

    参数:
    ------
    lines       : list of str, 原始文本行
    keyword_map : dict, {keyword: parser_function}

    返回:
    ------
    dict, 按关键字组织的数据结构
    """
    result = {key: [] for key in keyword_map}
    current_key = None
    for line in lines:
        line = line.strip().lower()
        if not line:
            continue
        first_word = line.split()[0] if line.split() else ""
        if first_word in keyword_map:
            current_key = first_word
        elif current_key is not None:
            parsed = keyword_map[current_key](line)
            if parsed is not None:
                result[current_key].append(parsed)
    return result


def parse_market_data_csv(text_lines):
    """
    快速解析市场数据CSV文本（模拟STLA快速文件解析）。
    返回结构化的期权链数据字典。
    """
    def parse_header(line):
        return line.split(',')

    def parse_row(line):
        parts = line.split(',')
        if len(parts) < 4:
            return None
        try:
            return {
                'strike': float(parts[0]),
                'maturity': float(parts[1]),
                'iv': float(parts[2]),
                'price': float(parts[3])
            }
        except (ValueError, IndexError):
            return None

    return fast_structured_parse(text_lines, {
        'header': parse_header,
        'data': parse_row
    })


# ========================================================================
# 382_fem_to_xml : 网格数据结构化管理与格式转换
# ========================================================================

class MeshDataManager:
    """
    管理有限差分/有限元网格的节点-单元关系。
    支持1D/2D网格生成、邻接关系查询、边界标记。
    """

    def __init__(self, dim, nodes, elements=None):
        """
        参数:
        ------
        dim      : int, 空间维度(1或2)
        nodes    : ndarray, 节点坐标 (dim, node_num)
        elements : ndarray, 单元定义 (element_order, element_num), 可选
        """
        self.dim = dim
        self.nodes = np.asarray(nodes, dtype=np.float64)
        self.node_num = self.nodes.shape[1] if self.nodes.ndim > 1 else self.nodes.shape[0]
        if elements is not None:
            self.elements = np.asarray(elements, dtype=np.int64)
            self.element_order = self.elements.shape[0]
            self.element_num = self.elements.shape[1]
        else:
            self.elements = None
            self.element_order = 0
            self.element_num = 0

    @staticmethod
    def generate_1d_uniform(x_min, x_max, n_nodes):
        """生成一维均匀网格。"""
        nodes = np.linspace(x_min, x_max, n_nodes)
        elements = np.zeros((2, n_nodes - 1), dtype=np.int64)
        for e in range(n_nodes - 1):
            elements[0, e] = e
            elements[1, e] = e + 1
        return MeshDataManager(1, nodes.reshape(1, -1), elements)

    @staticmethod
    def generate_2d_tensor(x_nodes, y_nodes):
        """
        生成二维张量积网格。

        参数:
        ------
        x_nodes : array, x方向节点坐标
        y_nodes : array, y方向节点坐标
        """
        nx = len(x_nodes)
        ny = len(y_nodes)
        node_num = nx * ny
        nodes = np.zeros((2, node_num), dtype=np.float64)
        idx = 0
        for j in range(ny):
            for i in range(nx):
                nodes[0, idx] = x_nodes[i]
                nodes[1, idx] = y_nodes[j]
                idx += 1
        # 三角形单元 (每个矩形分成2个三角形)
        element_num = (nx - 1) * (ny - 1) * 2
        elements = np.zeros((3, element_num), dtype=np.int64)
        e = 0
        for j in range(ny - 1):
            for i in range(nx - 1):
                n0 = j * nx + i
                n1 = n0 + 1
                n2 = n0 + nx
                n3 = n2 + 1
                elements[:, e] = [n0, n1, n2]
                e += 1
                elements[:, e] = [n1, n3, n2]
                e += 1
        return MeshDataManager(2, nodes, elements)

    def find_boundary_nodes_1d(self):
        """一维网格边界节点索引。"""
        if self.dim != 1:
            raise ValueError("仅适用于1D网格")
        return [0, self.node_num - 1]

    def find_boundary_nodes_2d_rect(self, nx, ny):
        """二维矩形网格边界节点索引。"""
        if self.dim != 2:
            raise ValueError("仅适用于2D网格")
        boundary = set()
        for j in range(ny):
            for i in range(nx):
                idx = j * nx + i
                if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                    boundary.add(idx)
        return sorted(boundary)

    def export_mesh_dict(self):
        """导出为字典格式。"""
        return {
            'dim': self.dim,
            'node_num': self.node_num,
            'nodes': self.nodes.tolist(),
            'element_order': self.element_order,
            'element_num': self.element_num,
            'elements': self.elements.tolist() if self.elements is not None else []
        }
