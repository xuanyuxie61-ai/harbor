"""
sparse_solver.py
稀疏矩阵与线性系统求解模块，用于数值相对论中的3+1分解ADM方程离散化。

融合种子项目:
- 507_hb_io: Harwell-Boeing稀疏矩阵文件格式 → 稀疏矩阵元数据结构与读取
- 089_biharmonic_fd2d: 二维双调和方程有限差分 → 高阶椭圆算子离散化

核心公式:
1. 二维双调和算子:
   Δ^2 u = ∂^4u/∂x^4 + 2∂^4u/∂x^2∂y^2 + ∂^4u/∂y^4

2. 13点有限差分模板 (在均匀网格 h 上):
   Δ^2 u_{i,j} ≈ (1/h^4) * [
        20 u_{i,j}
      -  8(u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1})
      +  2(u_{i+1,j+1} + u_{i+1,j-1} + u_{i-1,j+1} + u_{i-1,j-1})
      +  1(u_{i+2,j} + u_{i-2,j} + u_{i,j+2} + u_{i,j-2})
   ]

3. 数值相对论中的约束方程离散（类双调和形式）:
   在共形平坦近似下，Hamiltonian约束可写为:
   Δψ + (1/8) ψ^{-7} K_{ij} K^{ij} - (1/8) ψ R = 0
   其中 ψ 为共形因子，在初始数据构造中需要迭代求解。

4. 稀疏矩阵向量乘积（HB格式兼容）:
   y = A * x,  A 以 CSR (Compressed Sparse Row) 格式存储
"""

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve


# ---------------------------------------------------------------------------
# Harwell-Boeing 兼容的稀疏矩阵元数据结构 (源自 507_hb_io)
# ---------------------------------------------------------------------------

class SparseMatrixHB:
    """
    Harwell-Boeing 稀疏矩阵格式的简化Python实现。
    在数值相对论中，3D ADM方程离散化后产生的稀疏矩阵
    通常具有对称结构，可用此类高效存储。
    """
    
    def __init__(self, nrow, ncol, nnzero, mxtype='RSA'):
        """
        参数:
            nrow, ncol: 矩阵维度
            nnzero: 非零元个数
            mxtype: 矩阵类型，如 'RSA' (Real Symmetric Assembled)
        """
        self.nrow = int(nrow)
        self.ncol = int(ncol)
        self.nnzero = int(nnzero)
        self.mxtype = mxtype
        self.title = "Numeric Relativity Sparse Matrix"
        self.key = "NRADM"
        self.colptr = None
        self.rowind = None
        self.values = None
    
    def from_csr(self, csr: csr_matrix):
        """从 scipy CSR 矩阵构建 HB 结构。"""
        if csr.shape != (self.nrow, self.ncol):
            raise ValueError("CSR 矩阵维度与声明不符")
        csr = csr.tocsr()
        self.colptr = csr.indptr.astype(np.int64)
        self.rowind = csr.indices.astype(np.int64)
        self.values = csr.data.astype(np.float64)
        self.nnzero = len(self.values)
        return self
    
    def to_csr(self) -> csr_matrix:
        """转换回 scipy CSR 矩阵。"""
        if self.colptr is None:
            raise ValueError("矩阵数据未初始化")
        return csr_matrix(
            (self.values, self.rowind, self.colptr),
            shape=(self.nrow, self.ncol)
        )
    
    def matvec(self, x: np.ndarray) -> np.ndarray:
        """稀疏矩阵-向量乘积 y = A x。"""
        x = np.asarray(x, dtype=np.float64)
        if x.shape[0] != self.ncol:
            raise ValueError("向量维度不匹配")
        return self.to_csr().dot(x)
    
    def info(self):
        return {
            'title': self.title,
            'key': self.key,
            'nrow': self.nrow,
            'ncol': self.ncol,
            'nnzero': self.nnzero,
            'mxtype': self.mxtype,
            'density': self.nnzero / (self.nrow * self.ncol)
        }


# ---------------------------------------------------------------------------
# 二维双调和方程有限差分离散 (源自 089_biharmonic_fd2d)
# ---------------------------------------------------------------------------

def biharmonic_2d_stencil(h):
    """
    二维双调和算子的 13 点有限差分模板。
    模板系数 (按位置):
        (i,j-2):  1    (i,j-1): -8    (i,j): 20    (i,j+1): -8    (i,j+2): 1
        (i-2,j):  1    (i-1,j): -8
        (i+1,j): -8    (i+2,j):  1
        (i-1,j-1): 2   (i-1,j+1): 2   (i+1,j-1): 2   (i+1,j+1): 2
    
    公式:
        Δ^2 u ≈ (1/h^4) Σ_k c_k u_{i+δi_k, j+δj_k}
    """
    if h <= 0:
        raise ValueError("网格间距 h 必须为正")
    scale = 1.0 / (h ** 4)
    stencil = {
        (0, 0): 20.0 * scale,
        (1, 0): -8.0 * scale,
        (-1, 0): -8.0 * scale,
        (0, 1): -8.0 * scale,
        (0, -1): -8.0 * scale,
        (1, 1): 2.0 * scale,
        (1, -1): 2.0 * scale,
        (-1, 1): 2.0 * scale,
        (-1, -1): 2.0 * scale,
        (2, 0): 1.0 * scale,
        (-2, 0): 1.0 * scale,
        (0, 2): 1.0 * scale,
        (0, -2): 1.0 * scale,
    }
    return stencil


def build_biharmonic_matrix(nx, ny, h, bc='clamped'):
    """
    构建二维双调和方程的稀疏矩阵。
    边界条件:
        clamped: u = 0, ∂u/∂n = 0 在边界上
        (适用于数值相对论中 Brill-Lindquist 初始数据的求解)
    
    返回 scipy CSR 矩阵 A，使得 A u = f。
    """
    if nx < 5 or ny < 5:
        raise ValueError("网格维度至少为 5 以容纳 13 点模板")
    if h <= 0:
        raise ValueError("h 必须为正")
    
    N = nx * ny
    A = lil_matrix((N, N), dtype=np.float64)
    stencil = biharmonic_2d_stencil(h)
    
    def idx(i, j):
        # 将二维索引映射到一维
        if i < 0 or i >= nx or j < 0 or j >= ny:
            return -1
        return i * ny + j
    
    for i in range(nx):
        for j in range(ny):
            row = idx(i, j)
            if row < 0:
                continue
            
            # 边界处理: clamped 边界条件
            is_boundary = (i < 2 or i >= nx - 2 or j < 2 or j >= ny - 2)
            
            if is_boundary:
                # 边界上 u = 0
                A[row, row] = 1.0
            else:
                for (di, dj), coeff in stencil.items():
                    col = idx(i + di, j + dj)
                    if col >= 0:
                        A[row, col] += coeff
    
    return A.tocsr()


def biharmonic_rhs(nx, ny, h, rhs_func, bc='clamped'):
    """
    构建双调和方程的右端项 f。
    在数值相对论中，rhs_func 可能表示物质源项或曲率项。
    """
    N = nx * ny
    f = np.zeros(N, dtype=np.float64)
    
    for i in range(nx):
        for j in range(ny):
            x = i * h
            y = j * h
            row = i * ny + j
            is_boundary = (i < 2 or i >= nx - 2 or j < 2 or j >= ny - 2)
            if is_boundary and bc == 'clamped':
                f[row] = 0.0
            else:
                f[row] = rhs_func(x, y)
    
    return f


# ---------------------------------------------------------------------------
# 数值相对论初始数据求解器 (基于双调和方程思想)
# ---------------------------------------------------------------------------

def solve_initial_data_brill_lindquist(nx=65, ny=65, h=0.1, masses=None, positions=None):
    """
    简化的 Brill-Lindquist 初始数据求解。
    在共形平坦假设下，Hamiltonian约束方程近似为:
        Δ ψ = -2π ρ ψ^5
    此处用一个类双调和的正则化形式求解:
        (Δ^2 + ε Δ) ψ = f(ψ)
    
    其中 ψ 为共形因子，与物理度规的关系:
        g_{ij} = ψ^4 δ_{ij}
    
    公式:
        ψ = 1 + Σ_p m_p / (2 |r - r_p|)
        
    本函数采用有限差分离散并迭代求解。
    """
    if masses is None:
        masses = np.array([1.0, 1.0], dtype=np.float64)
    if positions is None:
        # 两个黑洞在 x 轴上对称放置
        positions = np.array([[-1.5, 0.0], [1.5, 0.0]], dtype=np.float64)
    
    if len(masses) != len(positions):
        raise ValueError("masses 与 positions 长度不匹配")
    
    # 解析解作为参考
    def analytic_psi(x, y):
        val = 1.0
        for m, pos in zip(masses, positions):
            r = np.sqrt((x - pos[0])**2 + (y - pos[1])**2)
            if r < 1e-10:
                r = 1e-10
            val += m / (2.0 * r)
        return val
    
    # 构建双调和正则化矩阵
    A = build_biharmonic_matrix(nx, ny, h, bc='clamped')
    
    # 右端项: 使用解析解构造
    def rhs_func(x, y):
        # 近似源项
        return analytic_psi(x, y) - 1.0
    
    f = biharmonic_rhs(nx, ny, h, rhs_func, bc='clamped')
    
    # 求解线性系统
    try:
        psi_flat = spsolve(A, f)
    except Exception as e:
        # 若直接求解失败，使用最小二乘近似
        psi_flat = np.linalg.lstsq(A.toarray(), f, rcond=None)[0]
    
    psi = psi_flat.reshape((nx, ny))
    
    # 数值验证: 检查解的合理性
    if np.any(np.isnan(psi)) or np.any(np.isinf(psi)):
        raise RuntimeError("初始数据求解产生 NaN 或 Inf")
    
    # 计算 ADM 质量
    adm_mass = np.sum(masses)
    
    return {
        'psi': psi,
        'analytic_psi': analytic_psi,
        'adm_mass': adm_mass,
        'nx': nx,
        'ny': ny,
        'h': h,
        'matrix_info': SparseMatrixHB(nx*ny, nx*ny, A.nnz).from_csr(A).info()
    }


# ---------------------------------------------------------------------------
# 共形平坦度规下的外曲率计算
# ---------------------------------------------------------------------------

def compute_extrinsic_curvature(psi, h):
    """
    计算共形平坦初始数据的外曲率 K_{ij}。
    在最大切片条件 (K=0) 下，trace-free 部分满足:
        K_{ij} = ψ^{-2} (L X)_{ij}
    其中 L 为 conformal Killing 算子。
    
    简化模型: 计算 ψ 的二阶导数作为曲率的代理量。
    """
    psi = np.asarray(psi, dtype=np.float64)
    nx, ny = psi.shape
    if nx < 3 or ny < 3:
        raise ValueError("psi 维度至少为 3x3")
    
    # 中心差分计算二阶导数
    K_xx = np.zeros_like(psi)
    K_yy = np.zeros_like(psi)
    K_xy = np.zeros_like(psi)
    
    K_xx[1:-1, 1:-1] = (psi[2:, 1:-1] - 2*psi[1:-1, 1:-1] + psi[:-2, 1:-1]) / (h**2)
    K_yy[1:-1, 1:-1] = (psi[1:-1, 2:] - 2*psi[1:-1, 1:-1] + psi[1:-1, :-2]) / (h**2)
    K_xy[1:-1, 1:-1] = (psi[2:, 2:] - psi[2:, :-2] - psi[:-2, 2:] + psi[:-2, :-2]) / (4*h**2)
    
    # trace-free 条件 K = K_xx + K_yy = 0
    trace = K_xx + K_yy
    K_xx -= 0.5 * trace
    K_yy -= 0.5 * trace
    
    return {'K_xx': K_xx, 'K_yy': K_yy, 'K_xy': K_xy, 'trace': trace}
