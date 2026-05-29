"""
special_matrices.py

特殊结构矩阵运算模块。
融合hankel_inverse的Hankel/Toeplitz矩阵构造与求逆思想，
用于电磁仿真中涉及的具有特殊结构的线性系统求解。

核心数学:
---------
1. Hankel矩阵: H_{ij} = x_{i+j-1} (常数反对角线)
2. Toeplitz矩阵: T_{ij} = t_{i-j} (常数对角线)
3. 这两个矩阵在圆柱坐标系问题、天线阵列分析中频繁出现。
4. Fiedler公式: Hankel矩阵的逆可用Hankel和Toeplitz矩阵的乘积表示。

电磁学应用:
-----------
- 圆柱谐振腔的模式分析涉及Hankel函数
- 线性天线阵列的互阻抗矩阵具有Toeplitz结构
- 时域信号的协方差矩阵分析
"""

import numpy as np


def hankel_matrix(n, x):
    """
    构造Hankel矩阵。
    H_{ij} = x_{i+j-1}, 其中x为长度为2n-1的向量。

    基于hankel_matrix.m的核心算法。

    Parameters
    ----------
    n : int
        矩阵阶数
    x : array_like, length 2n-1
        定义Hankel矩阵的向量

    Returns
    -------
    H : ndarray, shape (n, n)
    """
    x = np.asarray(x).flatten()
    if len(x) < 2 * n - 1:
        raise ValueError(f"向量长度至少为{2*n-1}，当前为{len(x)}")

    H = np.zeros((n, n))
    for j in range(n):
        H[:, j] = x[j:j + n]
    return H


def toeplitz_matrix(n, t):
    """
    构造Toeplitz矩阵。
    T_{ij} = t_{n-1+j-i}, 其中t为长度为2n-1的向量，中心元素为t[n-1]。

    基于toeplitz_matrix.m的核心算法。
    MATLAB公式: A(I,J) = X(N+J-I)
    Python对应: T[i,j] = t[n-1+j-i]

    Parameters
    ----------
    n : int
        矩阵阶数
    t : array_like, length 2n-1
        定义Toeplitz矩阵的向量

    Returns
    -------
    T : ndarray, shape (n, n)
    """
    t = np.asarray(t).flatten()
    if len(t) < 2 * n - 1:
        raise ValueError(f"向量长度至少为{2*n-1}，当前为{len(t)}")

    T = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            T[i, j] = t[n - 1 + j - i]
    return T


def hankel_inverse_fiedler(n, x):
    """
    使用Fiedler公式计算Hankel矩阵的逆。
    基于hankel_inverse.m的核心算法。

    算法步骤:
    1. 构造Hankel矩阵A
    2. 解两个线性系统: A·u = p, A·v = q
       其中 p = [x_n, x_{n+1}, ..., x_{2n-2}, 0]^T
             q = [0, 0, ..., 0, 1]^T
    3. 构造四个辅助矩阵 M1, M2, M3, M4
    4. A^{-1} = M1·M2 - M3·M4

    Parameters
    ----------
    n : int
        矩阵阶数
    x : array_like, length 2n-1
        定义Hankel矩阵的向量

    Returns
    -------
    A_inv : ndarray, shape (n, n)
        Hankel矩阵的逆
    A : ndarray, shape (n, n)
        原始Hankel矩阵
    """
    x = np.asarray(x).flatten()
    if len(x) < 2 * n - 1:
        raise ValueError(f"向量长度至少为{2*n-1}")

    A = hankel_matrix(n, x)

    # 构造右端向量
    p = np.zeros(n)
    p[:n - 1] = x[n:2 * n - 1]

    q = np.zeros(n)
    q[-1] = 1.0

    # 解线性系统（使用numpy的LU分解）
    try:
        u = np.linalg.solve(A, p)
        v = np.linalg.solve(A, q)
    except np.linalg.LinAlgError:
        # 矩阵奇异，使用伪逆
        A_inv = np.linalg.pinv(A)
        return A_inv, A

    # 构造辅助矩阵
    z1 = np.zeros(n)
    w1 = np.concatenate([v[1:], z1])
    M1 = hankel_matrix(n, w1)

    z2 = np.zeros(n - 1)
    w2 = np.concatenate([z2, u])
    M2 = toeplitz_matrix(n, w2)

    z3 = np.zeros(n)
    z3[0] = -1.0
    w3 = np.concatenate([u[1:], z3])
    M3 = hankel_matrix(n, w3)

    z4 = np.zeros(n - 1)
    w4 = np.concatenate([z4, v])
    M4 = toeplitz_matrix(n, w4)

    A_inv = M1 @ M2 - M3 @ M4
    return A_inv, A


def circulant_matrix(n, c):
    """
    构造循环矩阵（Circulant matrix）。
    在圆形天线阵列分析中具有重要应用。

    C_{ij} = c_{(j-i) mod n}

    Parameters
    ----------
    n : int
        矩阵阶数
    c : array_like, length n
        第一行元素

    Returns
    -------
    C : ndarray, shape (n, n)
    """
    c = np.asarray(c).flatten()
    if len(c) < n:
        raise ValueError(f"向量长度至少为{n}")

    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            C[i, j] = c[(j - i) % n]
    return C


def solve_toeplitz_system(n, t, b):
    """
    使用Levinson-Durbin递归算法求解Toeplitz线性系统。
    复杂度O(n²)，远优于一般稠密矩阵的O(n³)。

    Parameters
    ----------
    n : int
        矩阵阶数
    t : array_like, length 2n-1
        Toeplitz矩阵定义向量
    b : ndarray, length n
        右端向量

    Returns
    -------
    x : ndarray
        解向量
    """
    t = np.asarray(t).flatten()
    b = np.asarray(b).flatten()

    if len(t) < 2 * n - 1:
        raise ValueError("t向量长度不足")
    if len(b) < n:
        raise ValueError("b向量长度不足")

    # 使用标准Levinson-Durbin算法
    # T_n · x = b, 其中 T_n 由第一行 r 和第一列 c 定义
    r = t[n - 1:]
    c = t[n - 1::-1]

    x = np.zeros(n)
    f = np.zeros(n)  # 前向预测滤波器
    g = np.zeros(n)  # 后向预测滤波器

    # 初始化
    x[0] = b[0] / r[0]
    f[0] = 1.0 / r[0]
    g[0] = 1.0 / r[0]

    for k in range(1, n):
        # 反射系数
        alpha = np.dot(c[1:k + 1], f[:k])
        beta = 1.0 / (1.0 - alpha ** 2)

        # 更新滤波器
        f_new = np.zeros(k + 1)
        g_new = np.zeros(k + 1)
        f_new[:k] = beta * (f[:k] - alpha * g[k - 1::-1])
        g_new[:k] = beta * (g[k - 1::-1] - alpha * f[:k])
        f_new[k] = -beta * alpha * f[0]
        g_new[k] = beta * (1.0 - alpha) * f[0]

        f[:k + 1] = f_new
        g[:k + 1] = g_new

        # 更新解
        delta = b[k] - np.dot(r[1:k + 1][::-1], x[:k])
        x[:k + 1] += delta * f[:k + 1]

    return x


def antenna_array_impedance_matrix(n_elements, spacing_wavelength, ka=1.0):
    """
    计算线性天线阵列的互阻抗矩阵。
    该矩阵具有Toeplitz结构（均匀间距时）。

    互阻抗近似公式（简化模型）:
    Z_{ij} = Z_11 · J_0(k·|i-j|·d) / (k·|i-j|·d + 1)
    其中J_0为零阶Bessel函数。

    Parameters
    ----------
    n_elements : int
        阵元数量
    spacing_wavelength : float
        阵元间距（以波长为单位）
    ka : float
        天线电尺寸参数

    Returns
    -------
    Z : ndarray, shape (n_elements, n_elements)
        阻抗矩阵
    """
    d = spacing_wavelength * 2.0 * np.pi  # 转换为以k为单位的间距

    Z = np.zeros((n_elements, n_elements))
    Z_self = 50.0 + 20.0j  # 自阻抗近似

    for i in range(n_elements):
        for j in range(n_elements):
            if i == j:
                Z[i, j] = Z_self.real
            else:
                dist = abs(i - j) * d
                # 使用Hankel函数的渐近近似
                if dist > 0.01:
                    Z[i, j] = 30.0 * np.sin(dist) / dist
                else:
                    Z[i, j] = 30.0

    return Z
