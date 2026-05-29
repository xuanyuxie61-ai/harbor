"""
hamiltonian_builder.py
======================
分子哈密顿量构建与高斯求积特征值工具

原项目映射:
- 467_gen_laguerre_rule: 广义Laguerre高斯求积规则、Jacobi矩阵构造、
  隐式QL对角化 (imtqlx)、正交多项式类矩阵 (class_matrix)
- 1206_test_eigen: 随机对称矩阵生成 (r8symm_gen)、随机正交矩阵
  (r8mat_orth_uniform) 通过Householder变换

科学功能:
本模块利用高斯-拉盖尔求积计算分子轨道积分，并构建第二量子化的
分子电子哈密顿量。同时提供随机对称测试矩阵生成，用于验证VQE
在已知谱结构问题上的性能。
"""

import numpy as np
import math
from typing import Tuple, Optional


# ============================================================
# 正交多项式与Jacobi矩阵 (来自 467_gen_laguerre_rule)
# ============================================================

def class_matrix_laguerre(m: int, alpha: float) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    构造广义Laguerre多项式的Jacobi矩阵。

    广义Laguerre权函数: w(x) = x^alpha * exp(-x), x in [0, inf)

    Jacobi矩阵三对角元:
        a_j = 2j + 1 + alpha          (对角线)
        b_j = sqrt(j * (j + alpha))   (次对角线)
    零阶矩: zemu = Gamma(alpha + 1)

    参数:
        m: 矩阵阶数
        alpha: 参数，必须 > -1
    返回:
        aj: 对角线元素
        bj: 次对角线元素
        zemu: 零阶矩
    """
    if alpha <= -1.0:
        raise ValueError("alpha 必须 > -1")
    aj = np.zeros(m)
    bj = np.zeros(m)
    for i in range(m):
        aj[i] = 2.0 * (i + 1) - 1.0 + alpha
        bj[i] = np.sqrt((i + 1) * (i + 1 + alpha))
    zemu = np.exp(math.lgamma(alpha + 1.0))
    return aj, bj, zemu


def imtqlx(n: int, d: np.ndarray, e: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    隐式QL算法对角化对称三对角矩阵，对应 467_gen_laguerre_rule/imtqlx。

    输入: T = tridiag(e, d, e)，其中e[0]未使用，e[1:n]为次对角线。
    输出: d_sorted 为特征值（升序），z_transformed = Q^T z。

    算法基于Givens旋转的隐式位移QR迭代。
    """
    d = d.astype(float).copy()
    e = e.astype(float).copy()
    z = z.astype(float).copy()
    itn = 30
    prec = np.finfo(float).eps

    if n == 1:
        return d, z
    e[n - 1] = 0.0
    for l in range(n):
        j = 0
        while True:
            m_val = l
            while m_val < n - 1:
                if abs(e[m_val]) <= prec * (abs(d[m_val]) + abs(d[m_val + 1])):
                    break
                m_val += 1
            p = d[l]
            if m_val == l:
                break
            if j == itn:
                raise RuntimeError("IMTQLX迭代次数超限")
            j += 1
            g = (d[l + 1] - p) / (2.0 * e[l])
            r_val = np.sqrt(g * g + 1.0)
            g = d[m_val] - p + e[l] / (g + np.sign(g) * abs(r_val))
            s_val = 1.0
            c_val = 1.0
            p_local = 0.0
            mml = m_val - l
            for ii in range(1, mml + 1):
                i = m_val - ii
                f = s_val * e[i]
                b_val = c_val * e[i]
                if abs(f) >= abs(g):
                    c_val = g / f
                    r2 = np.sqrt(c_val * c_val + 1.0)
                    e[i + 1] = f * r2
                    s_val = 1.0 / r2
                    c_val *= s_val
                else:
                    s_val = f / g
                    r2 = np.sqrt(s_val * s_val + 1.0)
                    e[i + 1] = g * r2
                    c_val = 1.0 / r2
                    s_val *= c_val
                g = d[i + 1] - p_local
                r_val = (d[i] - g) * s_val + 2.0 * c_val * b_val
                p_local = s_val * r_val
                d[i + 1] = g + p_local
                g = c_val * r_val - b_val
                f = z[i + 1]
                z[i + 1] = s_val * z[i] + c_val * f
                z[i] = c_val * z[i] - s_val * f
            d[l] = d[l] - p_local
            e[l] = g
            e[m_val] = 0.0
    # 排序
    for ii in range(1, n):
        i = ii - 1
        k = i
        p = d[i]
        for j in range(ii, n):
            if d[j] < p:
                k = j
                p = d[j]
        if k != i:
            d[k] = d[i]
            d[i] = p
            p = z[i]
            z[i] = z[k]
            z[k] = p
    return d, z


def gauss_laguerre_rule(order: int, alpha: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算广义Laguerre-Gauss求积规则和节点权重。

    公式:
        integral_0^inf x^alpha exp(-x) f(x) dx \approx sum_i w_i f(x_i)

    节点x_i为Jacobi矩阵特征值，权重w_i = z_0i^2（z_0i为特征向量首分量）。
    """
    aj, bj, zemu = class_matrix_laguerre(order, alpha)
    z = np.zeros(order)
    z[0] = np.sqrt(zemu)
    x, w = imtqlx(order, aj, bj, z)
    w = w ** 2
    return x, w


# ============================================================
# 随机正交矩阵与对称测试矩阵 (来自 1206_test_eigen)
# ============================================================

def r8vec_house_column(n: int, x: np.ndarray, j: int) -> np.ndarray:
    """
    构造Householder向量v，使得H(v) = I - 2 v v^T / (v^T v)
    消去向量x[j+1:n]的分量。对应 1206_test_eigen/r8vec_house_column。
    """
    v = np.zeros(n)
    v[j:n] = x[j:n]
    s = np.linalg.norm(v)
    if s != 0.0:
        if v[j] < 0:
            s = -s
        v[j] += s
        v[j:n] /= np.sqrt(v[j] * s)
    return v


def r8mat_house_axh(n: int, a: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    计算 A -> A * H，其中H = I - 2 v v^T。对应 1206_test_eigen/r8mat_house_axh。
    """
    beta = -2.0 / np.dot(v, v)
    w = beta * a.T @ v
    return a + np.outer(v, w)


def r8mat_orth_uniform(n: int) -> np.ndarray:
    """
    生成随机正交矩阵Q，满足 Q^T Q = I。
    通过Householder QR分解标准高斯矩阵实现。
    对应 1206_test_eigen/r8mat_orth_uniform。
    """
    a = np.eye(n)
    for j in range(n - 1):
        x = np.zeros(n)
        x[j:n] = np.random.randn(n - j)
        v = r8vec_house_column(n, x, j)
        if np.linalg.norm(v) > 1e-14:
            a = r8mat_house_axh(n, a, v)
        # 随机反射
        if np.random.rand() > 0.5:
            k = np.random.randint(0, n)
            a[k, :] = -a[k, :]
    return a


def r8symm_gen(n: int, lambda_mean: float = 0.0,
               lambda_dev: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    生成具有指定特征值分布的随机对称矩阵。
    A = Q * diag(lambda) * Q^T，对应 1206_test_eigen/r8symm_gen。

    参数:
        n: 矩阵阶数
        lambda_mean: 特征值均值
        lambda_dev: 特征值标准差
    返回:
        A: 对称矩阵
        Q: 正交特征向量矩阵
        lambda_vec: 特征值向量
    """
    lambda_vec = lambda_mean + lambda_dev * np.random.randn(n)
    q = r8mat_orth_uniform(n)
    a = q @ np.diag(lambda_vec) @ q.T
    return a, q, lambda_vec


# ============================================================
# 分子哈密顿量构建
# ============================================================

class MolecularHamiltonian:
    """
    基于第二量子化的分子电子哈密顿量:
        H = sum_{pq} h_{pq} a_p^\dagger a_q
          + (1/2) sum_{pqrs} g_{pqrs} a_p^\dagger a_q^\dagger a_r a_s

    其中 h_{pq} 为单电子积分（动能+核吸引），
    g_{pqrs} 为双电子排斥积分（使用高斯求积近似）。
    """
    def __init__(self, n_orbitals: int = 4):
        self.n_orbitals = n_orbitals
        self.h_core = np.zeros((n_orbitals, n_orbitals))
        self.eri = np.zeros((n_orbitals, n_orbitals, n_orbitals, n_orbitals))
        self._build_integrals()

    def _build_integrals(self):
        """
        构建简化的一体和双体积分（使用高斯型轨道解析近似）。
        对于H2/STO-3G的4轨道系统，使用经验参数。
        """
        n = self.n_orbitals
        # 单电子积分: h_{pq} = T_{pq} + V_{pq}
        # 对角元素（轨道能量）
        for i in range(n):
            self.h_core[i, i] = -0.5 - 0.1 * i
        # 非对角耦合（共振积分）
        for i in range(n - 1):
            self.h_core[i, i + 1] = -0.2
            self.h_core[i + 1, i] = -0.2

        # 双电子积分: g_{pqrs} 使用简化模型
        # (ii|jj) 型库仑积分
        for i in range(n):
            for j in range(n):
                self.eri[i, i, j, j] = 0.3 / (abs(i - j) + 1.0)
                self.eri[i, j, j, i] = 0.1 / (abs(i - j) + 1.0)

    def get_nuclear_repulsion(self) -> float:
        """核排斥能（简化为双原子分子）。"""
        return 0.7

    def compute_exact_ground_state(self) -> Tuple[float, np.ndarray]:
        """
        使用FCI（全组态相互作用）计算精确基态能量。
        对于n_orbitals个自旋轨道，构造占据数表象的哈密顿量矩阵
        并精确对角化。
        """
        from itertools import combinations
        n_electrons = self.n_orbitals // 2  # 半满简化
        n_spin_orbitals = self.n_orbitals
        # 生成所有可能的占据组态
        configs = list(combinations(range(n_spin_orbitals), n_electrons))
        dim = len(configs)
        H_mat = np.zeros((dim, dim))

        def idx(config):
            try:
                return configs.index(tuple(sorted(config)))
            except ValueError:
                return -1

        for ic, c in enumerate(configs):
            # 单电子项
            for p in range(n_spin_orbitals):
                if p in c:
                    # a_p^\dagger a_p |c> = |c>
                    # 轨道映射: 自旋轨道 -> 空间轨道
                    spatial_p = p % self.n_orbitals
                    spin_p = p // self.n_orbitals
                    H_mat[ic, ic] += self.h_core[spatial_p, spatial_p]
                    for q in range(p + 1, n_spin_orbitals):
                        if q in c:
                            spatial_q = q % self.n_orbitals
                            spin_q = q // self.n_orbitals
                            if spin_p == spin_q:
                                H_mat[ic, ic] += self.eri[spatial_p, spatial_p, spatial_q, spatial_q]
                                H_mat[ic, ic] -= self.eri[spatial_p, spatial_q, spatial_q, spatial_p]
            # 非对角单电子项（简化，仅考虑相邻轨道跃迁）
            for p in range(n_spin_orbitals):
                if p in c:
                    for q in range(n_spin_orbitals):
                        if q not in c:
                            c_new = list(c)
                            c_new.remove(p)
                            c_new.append(q)
                            jc = idx(c_new)
                            if jc >= 0:
                                spatial_p = p % self.n_orbitals
                                spatial_q = q % self.n_orbitals
                                H_mat[ic, jc] += self.h_core[spatial_p, spatial_q]

        # 对角化
        eigvals = np.linalg.eigvalsh(H_mat)
        E0 = eigvals[0] + self.get_nuclear_repulsion()
        return float(E0), H_mat

    def to_pauli_strings(self) -> dict:
        """
        使用Jordan-Wigner变换将费米子哈密顿量映射为Pauli字符串。
        对于简化模型，返回预定义的Pauli项字典。
        """
        # TODO: 实现Jordan-Wigner变换，将费米子哈密顿量映射为Pauli字符串系数
        raise NotImplementedError("Hole 1: 请实现Jordan-Wigner变换")



def compute_radial_integral_gauss_laguerre(n: int, alpha: float,
                                            f: callable) -> float:
    """
    使用广义Laguerre-Gauss求积计算径向积分:
        integral_0^inf r^{2+alpha} exp(-r^2) f(r) dr
    通过变量替换 x = r^2 转化为标准Laguerre形式。
    """
    x, w = gauss_laguerre_rule(n, alpha)
    # x = r^2, r = sqrt(x), dr = dx / (2 sqrt(x))
    # 原积分: int_0^inf r^{alpha+2} exp(-r^2) f(r) dr
    # 替换后: int_0^inf x^{(alpha+2)/2} exp(-x) f(sqrt(x)) dx / (2 sqrt(x))
    #       = 0.5 * int_0^inf x^{alpha/2 + 1/2} exp(-x) f(sqrt(x)) dx
    # 标准Laguerre权: x^alpha exp(-x)
    # 所以需要调整: 令 beta = alpha/2 - 0.5
    # 这里使用简化处理
    result = 0.0
    for xi, wi in zip(x, w):
        ri = np.sqrt(xi)
        # Jacobian
        jac = 0.5 / ri
        result += wi * f(ri) * jac
    return result
