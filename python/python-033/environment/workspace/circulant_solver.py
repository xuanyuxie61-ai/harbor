"""
circulant_solver.py
基于种子项目 976_r8ci 的循环矩阵理论

循环矩阵在核物理中用于描述周期性边界条件下的中子扩散算子。
给定第一行 a = (a_0, a_1, ..., a_{n-1})，循环矩阵 C 满足：
    C_{i,j} = a_{(j-i) mod n}

关键性质：
1. 特征值为 DFT(a): λ_k = Σ_{j=0}^{n-1} a_j ω^{jk}, ω = exp(-2πi/n)
2. det(C) = Π_{k=0}^{n-1} λ_k
3. 线性系统 Cx = b 可通过 FFT 在 O(n log n) 求解

在核天体物理中，循环矩阵用于：
- 周期性核素链的刚性ODE隐式求解
- 中子星壳层中周期性密度扰动的响应矩阵
"""

import numpy as np


def circulant_eigenvalues(a):
    """
    计算循环矩阵的特征值：λ = FFT(a)

    参数:
        a : ndarray, shape (n,), 循环矩阵第一行

    返回:
        lam : ndarray, shape (n,), 复特征值
    """
    a = np.asarray(a, dtype=complex)
    n = a.shape[0]
    if n == 0:
        return np.array([], dtype=complex)
    lam = np.fft.fft(a)
    return lam


def circulant_determinant(a):
    """
    计算循环矩阵行列式：det(C) = Π λ_k

    参数:
        a : ndarray, shape (n,)

    返回:
        det : complex, 行列式
    """
    lam = circulant_eigenvalues(a)
    # 鲁棒处理：避免数值下溢/上溢，使用对数求和
    log_det = np.sum(np.log(lam + 1e-300))
    det = np.exp(log_det)
    return det


def circulant_solve(a, b, job=0):
    """
    求解循环线性系统 C x = b 或其转置 C^T x = b。

    利用 FFT 对角化：
        C = F^H Λ F,  F 为归一化 DFT 矩阵
        x = F^H Λ^{-1} F b = IFFT( FFT(b) / FFT(a) )

    参数:
        a : ndarray, shape (n,), 循环矩阵第一行
        b : ndarray, shape (n,) 或 (n, nrhs)
        job : int, 0 解 Cx=b, 非零 解 C^T x=b

    返回:
        x : ndarray, 解向量
    """
    a = np.asarray(a, dtype=complex)
    b = np.asarray(b, dtype=complex)
    n = a.shape[0]
    if b.shape[0] != n:
        raise ValueError("b 的第一维必须与 a 的长度一致")

    if job != 0:
        # C^T 也是循环矩阵，第一行为 (a_0, a_{n-1}, ..., a_1)
        a = np.concatenate(([a[0]], a[:0:-1]))

    # 检查 a 的 DFT 是否有零分量（矩阵奇异）
    a_fft = np.fft.fft(a)
    # 使用 masked 避免 divide by zero warning
    a_fft_inv = np.zeros_like(a_fft)
    nonzero = np.abs(a_fft) >= 1e-14
    a_fft_inv[nonzero] = 1.0 / a_fft[nonzero]

    if b.ndim == 1:
        b_fft = np.fft.fft(b)
        x_fft = b_fft * a_fft_inv
        x = np.fft.ifft(x_fft)
        return np.real_if_close(x, tol=1e-10)
    else:
        # 多右端项
        x = np.zeros_like(b, dtype=complex)
        for k in range(b.shape[1]):
            b_fft = np.fft.fft(b[:, k])
            x_fft = b_fft * a_fft_inv
            x[:, k] = np.fft.ifft(x_fft)
        return np.real_if_close(x, tol=1e-10)


def circulant_matvec(a, x):
    """
    计算循环矩阵与向量的乘积 y = C x，利用 FFT 加速。

    参数:
        a : ndarray, shape (n,)
        x : ndarray, shape (n,)

    返回:
        y : ndarray, shape (n,)
    """
    a = np.asarray(a, dtype=complex)
    x = np.asarray(x, dtype=complex)
    y = np.fft.ifft(np.fft.fft(a) * np.fft.fft(x))
    return np.real_if_close(y, tol=1e-10)


def build_circulant_dif2(n):
    """
    构造周期二阶差分循环矩阵，对应离散 Laplacian 的周期性边界条件。
    第一行: [2, -1, 0, ..., 0, -1]
    该矩阵对应算子 -d²/dx² 在周期性边界下的离散化。
    """
    a = np.zeros(n, dtype=float)
    a[0] = 2.0
    if n > 1:
        a[1] = -1.0
        a[-1] = -1.0
    return a


def test_circulant_solver():
    """自包含测试"""
    n = 64
    a = build_circulant_dif2(n)
    b = np.random.rand(n)
    x = circulant_solve(a, b, job=0)
    # 验证残差
    y = circulant_matvec(a, x)
    res = np.linalg.norm(y - b)
    print(f"[circulant_solver] Residual ||Cx-b|| = {res:.3e}")
    assert res < 1e-10, "Circulant solver residual too large"
    # 测试行列式
    det_val = circulant_determinant(a)
    print(f"[circulant_solver] det(C) = {det_val.real:.3e}")


if __name__ == "__main__":
    test_circulant_solver()
