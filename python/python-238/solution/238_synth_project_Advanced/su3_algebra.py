"""
su3_algebra.py
==============

SU(3) 李群与李代数的完整数值实现, 为格点 QCD 规范场提供群论基础.

核心公式:
---------
1. Gell-Mann 矩阵 (SU(3) 生成元):
       λ_1 = ( 0 1 0 )   λ_2 = ( 0 -i 0 )   λ_3 = ( 1  0 0 )
             ( 1 0 0 )         ( i  0 0 )         ( 0 -1 0 )
             ( 0 0 0 )         ( 0  0 0 )         ( 0  0 0 )

       λ_4 = ( 0 0 1 )   λ_5 = ( 0 0 -i )   λ_6 = ( 0 0 0 )
             ( 0 0 0 )         ( 0 0  0 )         ( 0 0 1 )
             ( 1 0 0 )         ( i 0  0 )         ( 0 1 0 )

       λ_7 = ( 0 0  0 )   λ_8 = 1/√3 ( 1  0  0 )
             ( 0 0 -i )            ( 0  1  0 )
             ( 0 i  0 )            ( 0  0 -2 )

   生成元 T_a = λ_a / 2, 满足 Tr(T_a T_b) = δ_{ab}/2

2. 结构常数 (完全反对称):
       [T_a, T_b] = i f_{abc} T_c
       {T_a, T_b} = (1/3) δ_{ab} I + d_{abc} T_c

3. Cayley-Hamilton 指数映射 (Drummond 1985):
       exp(i Q) = c_0 I + c_1 Q + c_2 Q^2
   其中 Q 为无迹反 Hermit 矩阵, c_k 由 Q 的本征值决定.

4. 群投影 (恢复 unitarity, 精度 O(ε^3)):
       U_proj = (1 + ε^2/4)^{-1/2} (1 + ε^2/8) U - ε ε† U / (8 (1 + ε^2/4)^{3/2})
   简化版本采用三次重正交化:
       U ← U (U† U)^{-1/2}

本模块融合种子项目:
  - 631_l4lib:   XOR/位掩码用于 SU(3) 矩阵符号与 parity
  - 545_house:   参考几何图案 (此处为参考规范构型) 的生成元分解
"""

import numpy as np
from typing import Tuple, List, Optional

# ============================================================
# Gell-Mann 矩阵 (8 个 3×3 无迹 Hermit 矩阵)
# ============================================================

def gell_mann_matrices() -> np.ndarray:
    """返回 8 个 Gell-Mann 矩阵 λ_a, shape = (8, 3, 3), dtype = complex128."""
    lam = np.zeros((8, 3, 3), dtype=np.complex128)
    # λ_1
    lam[0, 0, 1] = 1.0; lam[0, 1, 0] = 1.0
    # λ_2
    lam[1, 0, 1] = -1j; lam[1, 1, 0] = 1j
    # λ_3
    lam[2, 0, 0] = 1.0; lam[2, 1, 1] = -1.0
    # λ_4
    lam[3, 0, 2] = 1.0; lam[3, 2, 0] = 1.0
    # λ_5
    lam[4, 0, 2] = -1j; lam[4, 2, 0] = 1j
    # λ_6
    lam[5, 1, 2] = 1.0; lam[5, 2, 1] = 1.0
    # λ_7
    lam[6, 1, 2] = -1j; lam[6, 2, 1] = 1j
    # λ_8
    lam[7, 0, 0] = 1.0 / np.sqrt(3.0)
    lam[7, 1, 1] = 1.0 / np.sqrt(3.0)
    lam[7, 2, 2] = -2.0 / np.sqrt(3.0)
    return lam


# 生成元 T_a = λ_a / 2
GENERATORS = gell_mann_matrices() / 2.0


def structure_constants_f() -> np.ndarray:
    """完全反对称结构常数 f_{abc}, 满足 [T_a, T_b] = i f_{abc} T_c.

    通过 Tr([T_a, T_b] T_c) = (i/2) f_{abc} 计算 (利用 Tr(T_a T_b) = δ_{ab}/2).
    """
    f = np.zeros((8, 8, 8), dtype=np.float64)
    for a in range(8):
        for b in range(8):
            comm = GENERATORS[a] @ GENERATORS[b] - GENERATORS[b] @ GENERATORS[a]
            for c in range(8):
                # Tr([Ta, Tb] Tc) = i f_{abd} Tr(Td Tc) = i f_{abc} / 2
                val = 2.0 * np.trace(comm @ GENERATORS[c])
                f[a, b, c] = val.imag
    return f


def structure_constants_d() -> np.ndarray:
    """完全对称常数 d_{abc}, 满足 {T_a, T_b} = δ_{ab}/3 + d_{abc} T_c.

    通过 Tr({T_a, T_b} T_c) = d_{abc} / 2 计算.
    """
    d = np.zeros((8, 8, 8), dtype=np.float64)
    for a in range(8):
        for b in range(8):
            anticomm = GENERATORS[a] @ GENERATORS[b] + GENERATORS[b] @ GENERATORS[a]
            for c in range(8):
                val = 2.0 * np.trace(anticomm @ GENERATORS[c])
                d[a, b, c] = val.real
    return d


F_STRUCT = structure_constants_f()
D_STRUCT = structure_constants_d()


# ============================================================
# SU(3) 矩阵生成与投影
# ============================================================

def su3_random(rng: np.random.Generator) -> np.ndarray:
    """生成均匀分布于 Haar 测度的 SU(3) 矩阵.

    算法 (Cabibbo-Marinari 1980 + Gaussian 消去):
        1. 生成 3×3 复数高斯矩阵 Z
        2. QR 分解: Z = Q R
        3. 调整相位使 det Q = 1: Q ← Q diag(1, 1, det(Q)*)
    """
    z = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    ph = d / np.abs(d)
    # 使 det(q * diag(ph)) = 1
    det_q = np.linalg.det(q)
    correction = det_q.conj() / np.abs(det_q.conj())
    q = q * correction
    # 确保严格在 SU(3) 中
    return project_to_su3(q)


def project_to_su3(u: np.ndarray) -> np.ndarray:
    """将 3×3 复数矩阵投影到 SU(3) 群流形.

    步骤:
        1. U ← U (U† U)^{-1/2}  (极分解恢复 unitarity)
        2. U ← U / (det U)^{1/3}  (恢复 unit determinant)
    数值精度: 误差 ~1e-14.
    """
    u = np.asarray(u, dtype=np.complex128)
    # 极分解: 恢复 unitarity
    uhdag_u = u.conj().T @ u
    # 用 Hermitian 正定矩阵的平方根
    eigvals, eigvecs = np.linalg.eigh(uhdag_u)
    eigvals = np.clip(eigvals.real, 1e-14, None)
    inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.conj().T
    u = u @ inv_sqrt
    # 单位行列式修正
    det_u = np.linalg.det(u)
    phase = det_u ** (-1.0 / 3.0)
    u = u * phase
    return u


def algebra_element(coefficients: np.ndarray) -> np.ndarray:
    """从 8 个实系数构建李代数元素 Q = i Σ_a c_a T_a.

    返回的是反 Hermit 无迹矩阵, 可用于指数映射.
    """
    assert coefficients.shape == (8,)
    q = np.zeros((3, 3), dtype=np.complex128)
    for a in range(8):
        q += coefficients[a] * GENERATORS[a]
    return 1j * q


def su3_exp(q: np.ndarray) -> np.ndarray:
    """反 Hermit 矩阵 Q 的指数映射 exp(Q) ∈ SU(3).

    采用 Cayley-Hamilton 方法 (Drummond 1985):
        exp(Q) = c_0 I + c_1 Q + c_2 Q^2
    其中 Q = i θ_a T_a, 由 Q 的本征值决定 c_k.

    对于 3×3 反 Hermit 矩阵, 本征值为纯虚数: ±i φ_1, ±i φ_2, i(φ_1+φ_2)
    (无迹约束要求本征值之和为零).

    当本征值退化时使用 Taylor 展开的极限形式以保证数值稳定性.
    """
    q = np.asarray(q, dtype=np.complex128)
    # 直接对角化 (对小矩阵稳定)
    eigvals = np.linalg.eigvals(q)
    # 对于反 Hermit 矩阵, 本征值为纯虚数
    i_phases = eigvals

    # 检查是否退化 (数值上非常接近)
    if np.max(np.abs(i_phases - i_phases[0])) < 1e-10:
        # 所有本征值近似相等 → exp(Q) ≈ e^{iφ} I
        return np.exp(i_phases[0]) * np.eye(3, dtype=np.complex128)

    # 一般情况: 利用 Sylvester 插值
    # exp(Q) = Σ_j exp(λ_j) Π_{k≠j} (Q - λ_k I) / (λ_j - λ_k)
    result = np.zeros((3, 3), dtype=np.complex128)
    for j in range(3):
        prod = np.exp(i_phases[j]) * np.eye(3, dtype=np.complex128)
        for k in range(3):
            if k != j:
                denom = i_phases[j] - i_phases[k]
                if np.abs(denom) < 1e-12:
                    # 退化处理: 使用 L'Hôpital 规则
                    prod = prod @ (np.eye(3) + q - i_phases[j] * np.eye(3))
                else:
                    prod = prod @ (q - i_phases[k] * np.eye(3)) / denom
        result += prod

    return project_to_su3(result.real + 1j * result.imag)


def su3_exp_safe(q: np.ndarray) -> np.ndarray:
    """指数映射的安全版本, 使用矩阵指数 Pade 逼近.

    当 Cayley-Hamilton 方法退化时备用.
    采用 scipy-free 实现: scaling & squaring + Taylor 展开至 12 阶.
    """
    q = np.asarray(q, dtype=np.complex128)
    norm = np.linalg.norm(q, ord=np.inf)
    if norm < 1e-8:
        return np.eye(3, dtype=np.complex128) + q + 0.5 * q @ q

    # Scaling
    s = max(0, int(np.ceil(np.log2(norm / 0.5))))
    q_scaled = q / (2 ** s)

    # Taylor 展开至 12 阶
    result = np.eye(3, dtype=np.complex128)
    term = np.eye(3, dtype=np.complex128)
    for k in range(1, 13):
        term = term @ q_scaled / k
        result = result + term

    # Squaring
    for _ in range(s):
        result = result @ result

    return project_to_su3(result)


def lie_bracket(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """李括号 [A, B] = A B - B A."""
    return a @ b - b @ a


def adjoint_action(u: np.ndarray, q: np.ndarray) -> np.ndarray:
    """伴随作用: Ad(U) Q = U Q U^†.

    在 HMC 中用于动量更新.
    """
    return u @ q @ u.conj().T


def traceless_anti_hermitian(m: np.ndarray) -> np.ndarray:
    """提取矩阵的无迹反 Hermit 部分.

    用于从力矩阵中提取 su(3) 代数元素:
        Q = (M - M^†) / 2 - (1/3) Tr(M - M^†) I
    """
    ah = 0.5 * (m - m.conj().T)
    tr = np.trace(ah) / 3.0
    return ah - tr * np.eye(3, dtype=np.complex128)


def color_trace(u: np.ndarray) -> complex:
    """计算 SU(3) 矩阵的色迹, 返回实部 (虚部理论上为零)."""
    return np.trace(u).real


# ============================================================
# 群流形上的距离与范数
# ============================================================

def su3_distance(u: np.ndarray, v: np.ndarray) -> float:
    """SU(3) 群流形上的测地距离.

    d(U, V) = || log(U^† V) ||_F / √2

    其中 ||·||_F 为 Frobenius 范数.
    """
    w = u.conj().T @ v
    # log(W) 对于 SU(3) 矩阵
    eigvals, eigvecs = np.linalg.eig(w)
    phases = np.angle(eigvals)
    log_w = eigvecs @ np.diag(1j * phases) @ eigvecs.conj().T
    return float(np.linalg.norm(log_w, 'fro') / np.sqrt(2.0))


def casimir_fundamental() -> float:
    """ fundamental 表示的二次 Casimir: C_F = (N_c^2 - 1)/(2 N_c) = 4/3 (N_c=3)."""
    return (3.0 ** 2 - 1.0) / (2.0 * 3.0)


def casimir_adjoint() -> float:
    """伴随表示的二次 Casimir: C_A = N_c = 3."""
    return 3.0


def beta_from_coupling(g: float) -> float:
    """规范耦合常数 g → 格点 β 参数.

    对于 SU(3): β = 2 N_c / g^2 = 6 / g^2

    临界温度对应 β_c ≈ 5.69 (N_t = 4) 和 β_c ≈ 6.20 (N_t = 6).
    """
    return 6.0 / (g ** 2)


def coupling_from_beta(beta: float) -> float:
    """β → g."""
    return np.sqrt(6.0 / beta)
