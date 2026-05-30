
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve






class SparseMatrixHB:
    
    def __init__(self, nrow, ncol, nnzero, mxtype='RSA'):
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
        if csr.shape != (self.nrow, self.ncol):
            raise ValueError("CSR 矩阵维度与声明不符")
        csr = csr.tocsr()
        self.colptr = csr.indptr.astype(np.int64)
        self.rowind = csr.indices.astype(np.int64)
        self.values = csr.data.astype(np.float64)
        self.nnzero = len(self.values)
        return self
    
    def to_csr(self) -> csr_matrix:
        if self.colptr is None:
            raise ValueError("矩阵数据未初始化")
        return csr_matrix(
            (self.values, self.rowind, self.colptr),
            shape=(self.nrow, self.ncol)
        )
    
    def matvec(self, x: np.ndarray) -> np.ndarray:
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






def biharmonic_2d_stencil(h):
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
    if nx < 5 or ny < 5:
        raise ValueError("网格维度至少为 5 以容纳 13 点模板")
    if h <= 0:
        raise ValueError("h 必须为正")
    
    N = nx * ny
    A = lil_matrix((N, N), dtype=np.float64)
    stencil = biharmonic_2d_stencil(h)
    
    def idx(i, j):

        if i < 0 or i >= nx or j < 0 or j >= ny:
            return -1
        return i * ny + j
    
    for i in range(nx):
        for j in range(ny):
            row = idx(i, j)
            if row < 0:
                continue
            

            is_boundary = (i < 2 or i >= nx - 2 or j < 2 or j >= ny - 2)
            
            if is_boundary:

                A[row, row] = 1.0
            else:
                for (di, dj), coeff in stencil.items():
                    col = idx(i + di, j + dj)
                    if col >= 0:
                        A[row, col] += coeff
    
    return A.tocsr()


def biharmonic_rhs(nx, ny, h, rhs_func, bc='clamped'):
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






def solve_initial_data_brill_lindquist(nx=65, ny=65, h=0.1, masses=None, positions=None):
    if masses is None:
        masses = np.array([1.0, 1.0], dtype=np.float64)
    if positions is None:

        positions = np.array([[-1.5, 0.0], [1.5, 0.0]], dtype=np.float64)
    
    if len(masses) != len(positions):
        raise ValueError("masses 与 positions 长度不匹配")
    

    def analytic_psi(x, y):
        val = 1.0
        for m, pos in zip(masses, positions):
            r = np.sqrt((x - pos[0])**2 + (y - pos[1])**2)
            if r < 1e-10:
                r = 1e-10
            val += m / (2.0 * r)
        return val
    

    A = build_biharmonic_matrix(nx, ny, h, bc='clamped')
    

    def rhs_func(x, y):

        return analytic_psi(x, y) - 1.0
    
    f = biharmonic_rhs(nx, ny, h, rhs_func, bc='clamped')
    

    try:
        psi_flat = spsolve(A, f)
    except Exception as e:

        psi_flat = np.linalg.lstsq(A.toarray(), f, rcond=None)[0]
    
    psi = psi_flat.reshape((nx, ny))
    

    if np.any(np.isnan(psi)) or np.any(np.isinf(psi)):
        raise RuntimeError("初始数据求解产生 NaN 或 Inf")
    

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






def compute_extrinsic_curvature(psi, h):
    psi = np.asarray(psi, dtype=np.float64)
    nx, ny = psi.shape
    if nx < 3 or ny < 3:
        raise ValueError("psi 维度至少为 3x3")
    

    K_xx = np.zeros_like(psi)
    K_yy = np.zeros_like(psi)
    K_xy = np.zeros_like(psi)
    
    K_xx[1:-1, 1:-1] = (psi[2:, 1:-1] - 2*psi[1:-1, 1:-1] + psi[:-2, 1:-1]) / (h**2)
    K_yy[1:-1, 1:-1] = (psi[1:-1, 2:] - 2*psi[1:-1, 1:-1] + psi[1:-1, :-2]) / (h**2)
    K_xy[1:-1, 1:-1] = (psi[2:, 2:] - psi[2:, :-2] - psi[:-2, 2:] + psi[:-2, :-2]) / (4*h**2)
    

    trace = K_xx + K_yy
    K_xx -= 0.5 * trace
    K_yy -= 0.5 * trace
    
    return {'K_xx': K_xx, 'K_yy': K_yy, 'K_xy': K_xy, 'trace': trace}
