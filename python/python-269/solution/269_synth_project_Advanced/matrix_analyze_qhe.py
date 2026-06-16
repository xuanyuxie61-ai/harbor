"""
matrix_analyze_qhe.py — 量子霍尔哈密顿量的矩阵结构分析
============================================================

本模块对哈密顿量矩阵进行全面的线性代数性质分析:

1. 厄米性检验: ||H - H†||_F
2. 正定性检验: Cholesky 分解
3. 对角占优性: 迭代收敛性保证
4. 对称性/中心对称性: 物理对称性分析
5. 带结构: 带宽和稀疏度
6. 条件数: 数值稳定性指标
7. 谱性质: 本征值分布

这些分析源自 r8mat_analyze 项目中的矩阵属性检测框架,
但专门适配了量子霍尔哈密顿量的特殊结构.

矩阵性质与物理的对应:
    - 厄米性 → 物理可观测量为实数
    - 正定性 → 系统稳定性 (对 H + cI, c 足够大)
    - 稀疏度 → 数值计算效率
    - 条件数 → 本征值问题的数值难度
    - 带结构 → 关联长度和计算复杂度
    - 中心对称 → 粒子-空穴对称性

参考文献:
    [1] Golub, G. H. & Van Loan, C. F. "Matrix Computations" (4th ed.)
    [2] Trefethen, L. N. & Bau, D. "Numerical Linear Algebra" (SIAM)
"""

import numpy as np
from scipy import sparse
from typing import Dict, Any, Tuple


def analyze_hamiltonian(H: sparse.csr_matrix,
                        label: str = "QHE Hamiltonian") -> Dict[str, Any]:
    """全面分析哈密顿量矩阵的线性代数性质

    Args:
        H: 稀疏哈密顿量矩阵
        label: 矩阵标签
    Returns:
        分析结果字典
    """
    results = {
        'label': label,
        'shape': H.shape,
        'nnz': H.nnz,
        'density': H.nnz / (H.shape[0] * H.shape[1]),
    }

    H_dense = H.toarray() if sparse.issparse(H) else np.array(H)
    N = H_dense.shape[0]

    # 1. 厄米性检验: ||H - H†||_F
    hermitian_error = np.linalg.norm(H_dense - H_dense.conj().T, 'fro')
    results['hermitian_error'] = hermitian_error
    results['is_hermitian'] = hermitian_error < 1e-10

    # 2. 对称性 (实数部分)
    H_real = H_dense.real
    symmetric_error = np.linalg.norm(H_real - H_real.T, 'fro')
    results['symmetric_error'] = symmetric_error

    # 3. 反对称性 (虚数部分)
    H_imag = H_dense.imag
    antisymmetric_error = np.linalg.norm(H_imag + H_imag.T, 'fro')
    results['antisymmetric_imag_error'] = antisymmetric_error

    # 4. 对角占优性
    diag_dom = check_diagonal_dominance(H_dense)
    results['diagonally_dominant'] = diag_dom['strict']
    results['diag_dominance_ratio'] = diag_dom['min_ratio']

    # 5. 带宽分析
    bandwidth = compute_bandwidth(H)
    results['upper_bandwidth'] = bandwidth['upper']
    results['lower_bandwidth'] = bandwidth['lower']
    results['total_bandwidth'] = bandwidth['total']

    # 6. 条件数估计 (使用 ARNOP 估计)
    try:
        cond_est = estimate_condition_number(H)
        results['condition_number'] = cond_est
    except Exception:
        results['condition_number'] = float('inf')

    # 7. Frobenius 范数和谱范数
    results['norm_frobenius'] = np.linalg.norm(H_dense, 'fro')
    results['norm_spectral'] = np.linalg.norm(H_dense, 2)

    # 8. 迹 (所有本征值之和)
    results['trace'] = np.trace(H_dense).real

    # 9. Cholesky 分解检验 (对 H + cI)
    shift = abs(results['trace'].real) / N + 1.0
    H_shifted = H_dense + shift * np.eye(N)
    cholesky_ok, cholesky_info = try_cholesky(H_shifted)
    results['cholesky_shifted_ok'] = cholesky_ok
    results['cholesky_shift'] = shift

    # 10. 稀疏度分析
    results['sparsity'] = 1.0 - results['density']
    results['avg_nnz_per_row'] = H.nnz / N

    return results


def check_diagonal_dominance(A: np.ndarray) -> Dict[str, Any]:
    """检查矩阵的对角占优性

    严格对角占优: |a_ii| > Σ_{j≠i} |a_ij|  ∀i
    弱对角占优:   |a_ii| ≥ Σ_{j≠i} |a_ij|  ∀i (至少一个严格)

    对角占优矩阵保证:
    - Jacobi 和 Gauss-Seidel 迭代收敛
    - 非奇异 (Levy-Desplanques 定理)

    Args:
        A: 方阵
    Returns:
        分析结果
    """
    N = A.shape[0]
    ratios = []
    for i in range(N):
        diag = abs(A[i, i])
        off_diag_sum = np.sum(np.abs(A[i, :])) - diag
        if off_diag_sum > 0:
            ratios.append(diag / off_diag_sum)
        else:
            ratios.append(float('inf'))

    min_ratio = min(ratios)
    return {
        'strict': min_ratio > 1.0,
        'weak': min_ratio >= 1.0,
        'min_ratio': min_ratio,
        'ratios': ratios,
    }


def compute_bandwidth(H: sparse.csr_matrix) -> Dict[str, int]:
    """计算矩阵的带宽

    上带宽 p: a_ij = 0 for j - i > p
    下带宽 q: a_ij = 0 for i - j > q
    总带宽: p + q + 1

    带状矩阵的存储和计算优势:
    - 存储: O(N·(p+q)) 而非 O(N²)
    - LU 分解: O(N·(p+q)²) 而非 O(N³)

    Args:
        H: 稀疏矩阵
    Returns:
        带宽信息
    """
    H_coo = sparse.coo_matrix(H)
    if H_coo.nnz == 0:
        return {'upper': 0, 'lower': 0, 'total': 1}

    diffs = H_coo.col - H_coo.row
    upper = max(diffs) if len(diffs) > 0 else 0
    lower = max(-diffs) if len(diffs) > 0 else 0

    return {
        'upper': max(0, upper),
        'lower': max(0, lower),
        'total': upper + lower + 1,
    }


def estimate_condition_number(H: sparse.csr_matrix) -> float:
    """估计矩阵的条件数 κ(H) = ||H||·||H⁻¹||

    使用 Lanczos 方法估计最大和最小本征值的比值.
    对于厄米矩阵: κ₂(H) = |λ_max| / |λ_min|

    条件数的物理含义:
    - κ ~ 1: 良态, 本征值问题容易求解
    - κ ~ 10^p: 损失 p 位有效数字
    - κ → ∞: 病态, 需要正则化

    Args:
        H: 稀疏矩阵
    Returns:
        条件数估计
    """
    from scipy.sparse.linalg import eigsh

    N = H.shape[0]
    try:
        if N <= 10:
            evals = np.linalg.eigvalsh(H.toarray())
        else:
            evals_max = eigsh(H, k=1, which='LA', return_eigenvectors=False)
            evals_min = eigsh(H, k=1, which='SA', return_eigenvectors=False)
            evals = np.concatenate([evals_min, evals_max])

        abs_evals = np.abs(evals)
        abs_evals = abs_evals[abs_evals > 1e-15]
        if len(abs_evals) < 2:
            return float('inf')
        return float(max(abs_evals) / min(abs_evals))
    except Exception:
        return float('inf')


def try_cholesky(A: np.ndarray) -> Tuple[bool, str]:
    """尝试 Cholesky 分解

    Cholesky 分解 A = LL† 存在当且仅当 A 是正定矩阵.
    对于量子霍尔哈密顿量, H 本身不一定正定,
    但 H + cI (c 足够大) 可以正定.

    Args:
        A: 待检验矩阵
    Returns:
        (是否成功, 信息字符串)
    """
    try:
        L = np.linalg.cholesky(A)
        return True, "正定"
    except np.linalg.LinAlgError as e:
        return False, f"非正定: {str(e)}"


def check_centrosymmetry(H: sparse.csr_matrix) -> Dict[str, Any]:
    """检查中心对称性: H = JHJ (J 是反转矩阵)

    中心对称性对应粒子-空穴对称性:
        J_{ij} = δ_{i, N-1-j}
        H 中心对称 ↔ [H, J] = 0

    在量子霍尔系统中, 中心对称性与:
    - 六角格点的子晶格对称性
    - 磁场反演对称性
    有关.

    Args:
        H: 稀疏矩阵
    Returns:
        分析结果
    """
    N = H.shape[0]
    H_dense = H.toarray() if sparse.issparse(H) else np.array(H)
    J = np.fliplr(np.eye(N))
    JHJ = J @ H_dense @ J
    error = np.linalg.norm(H_dense - JHJ, 'fro')
    return {
        'centrosymmetric': error < 1e-10,
        'error': error,
    }


def print_analysis(results: Dict[str, Any]) -> str:
    """格式化输出分析结果"""
    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(f"  矩阵分析: {results['label']}")
    lines.append(f"{'='*60}")
    lines.append(f"  维度: {results['shape']}")
    lines.append(f"  非零元: {results['nnz']}")
    lines.append(f"  密度: {results['density']:.6e}")
    lines.append(f"  稀疏度: {results['sparsity']:.6f}")
    lines.append(f"  平均每行非零元: {results['avg_nnz_per_row']:.1f}")
    lines.append(f"  {'─'*50}")
    lines.append(f"  厄米性: {'是' if results['is_hermitian'] else '否'} "
                 f"(误差: {results['hermitian_error']:.2e})")
    lines.append(f"  对角占优: {'是' if results['diagonally_dominant'] else '否'} "
                 f"(最小比值: {results['diag_dominance_ratio']:.4f})")
    lines.append(f"  上带宽: {results['upper_bandwidth']}")
    lines.append(f"  下带宽: {results['lower_bandwidth']}")
    lines.append(f"  条件数: {results['condition_number']:.4e}")
    lines.append(f"  Frobenius 范数: {results['norm_frobenius']:.6f}")
    lines.append(f"  谱范数: {results['norm_spectral']:.6f}")
    lines.append(f"  迹 (Re): {results['trace']:.6f}")
    lines.append(f"  Cholesky (shifted): {'OK' if results['cholesky_shifted_ok'] else 'FAIL'}")
    lines.append(f"{'='*60}\n")
    return '\n'.join(lines)
