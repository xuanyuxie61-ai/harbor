"""
fem_forward.py — 有限元正演算子模块
====================================
来源项目映射:
  - 387_fem1d_bvp_quadratic → 1D 二次有限元 BVP 求解
    (刚度矩阵装配、L2/H1/L∞ 误差估计)

科学背景:
  随机椭圆 PDE 正演问题:
    -∇·(a(x,ω) ∇u(x,ω)) = f(x)      in Ω = (0,1)
    u(0,ω) = u(1,ω) = 0
  其中扩散系数 a(x,ω) 为随机场, 用 PCE 展开:
    a(x,ω) = ∑_{α} a_α(x) Ψ_α(ω)
  本模块实现 1D 二次有限元离散, 并构造 PCE-Galerkin 投影矩阵.

核心公式:
  单元刚度矩阵 (参考单元 [0,1]):
    K_e = ∫_0^1 a(x_e(ξ)) (dφ/dξ)² (2/h_e) dξ
  二次形函数 (参考单元):
    φ_1(ξ) = (1-ξ)(1-2ξ),  φ_2(ξ) = 4ξ(1-ξ),  φ_3(ξ) = ξ(2ξ-1)
  载荷向量:
    F_i = ∫_0^1 f(x) φ_i(x) dx
  Galerkin 投影 (随机):
    ∑_β E[Ψ_α Ψ_β a] K u_β = E[Ψ_α] F
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


# ----------------------------------------------------------------------
# 1D 二次有限元 (源自 387_fem1d_bvp_quadratic)
# ----------------------------------------------------------------------
def fem1d_nodes(n_elem):
    """生成 n_elem 个二次单元的节点 (每单元 3 节点, 相邻共享端点)."""
    nodes = []
    for e in range(n_elem):
        x0 = e / n_elem
        x1 = (e + 1) / n_elem
        nodes.extend([x0, 0.5 * (x0 + x1), x1])
    # 去重
    nodes = sorted(set(nodes))
    return np.array(nodes)


def fem1d_connectivity(n_elem):
    """单元-节点连接表 (每单元 3 个局部节点索引)."""
    conn = []
    for e in range(n_elem):
        base = 2 * e
        conn.append([base, base + 1, base + 2])
    return np.array(conn)


def reference_quadratic_basis(xi):
    """参考单元上的二次形函数及其导数.

    φ_1(ξ) = (1-ξ)(1-2ξ),   φ_1'(ξ) = -3 + 4ξ
    φ_2(ξ) = 4ξ(1-ξ),        φ_2'(ξ) =  4 - 8ξ
    φ_3(ξ) = ξ(2ξ-1),        φ_3'(ξ) = -1 + 4ξ
    """
    phi = np.array([(1 - xi) * (1 - 2 * xi),
                    4 * xi * (1 - xi),
                    xi * (2 * xi - 1)])
    dphi = np.array([-3 + 4 * xi,
                     4 - 8 * xi,
                     -1 + 4 * xi])
    return phi, dphi


def assemble_stiffness_1d(nodes, conn, diffusivity, n_gauss=3):
    """装配 1D 刚度矩阵.

    K_ij = ∫ a(x) φ_i'(x) φ_j'(x) dx
         = ∑_e ∫_{x_e} a(x) φ_i' φ_j' dx
         = ∑_e (2/h_e) ∫_0^1 a(x_e(ξ)) dφ_i/dξ dφ_j/dξ dξ
    """
    n_nodes = len(nodes)
    K = sparse.lil_matrix((n_nodes, n_nodes))
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(n_gauss)
    xi = 0.5 * (gl_nodes + 1)
    w = 0.5 * gl_weights

    for e in range(len(conn)):
        c = conn[e]
        h = nodes[c[2]] - nodes[c[0]]
        Ke = np.zeros((3, 3))
        for q in range(n_gauss):
            _, dphi = reference_quadratic_basis(xi[q])
            xq = nodes[c[0]] + 0.5 * h * (1 + gl_nodes[q])
            aq = diffusivity(xq)
            for i in range(3):
                for j in range(3):
                    Ke[i, j] += aq * dphi[i] * dphi[j] * w[q] * (2.0 / h)
        for i in range(3):
            for j in range(3):
                K[c[i], c[j]] += Ke[i, j]
    return K.tocsr()


def assemble_load_1d(nodes, conn, rhs_func, n_gauss=3):
    """装配载荷向量 F_i = ∫ f(x) φ_i(x) dx."""
    n_nodes = len(nodes)
    F = np.zeros(n_nodes)
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(n_gauss)
    xi = 0.5 * (gl_nodes + 1)
    w = 0.5 * gl_weights

    for e in range(len(conn)):
        c = conn[e]
        h = nodes[c[2]] - nodes[c[0]]
        Fe = np.zeros(3)
        for q in range(n_gauss):
            phi, _ = reference_quadratic_basis(xi[q])
            xq = nodes[c[0]] + 0.5 * h * (1 + gl_nodes[q])
            fq = rhs_func(xq)
            for i in range(3):
                Fe[i] += fq * phi[i] * w[q] * (h / 2.0)
        for i in range(3):
            F[c[i]] += Fe[i]
    return F


def solve_bvp_1d(n_elem, diffusivity, rhs_func):
    """求解 1D BVP: -(a u')' = f,  u(0)=u(1)=0.

    返回:
        nodes   : 节点坐标
        u       : 节点解
        K       : 刚度矩阵
        F       : 载荷向量
    """
    nodes = fem1d_nodes(n_elem)
    conn = fem1d_connectivity(n_elem)
    K = assemble_stiffness_1d(nodes, conn, diffusivity)
    F = assemble_load_1d(nodes, conn, rhs_func)
    # Dirichlet BC
    free = np.arange(1, len(nodes) - 1)
    u = np.zeros(len(nodes))
    u[free] = spsolve(K[free][:, free], F[free])
    return nodes, u, K, F


# ----------------------------------------------------------------------
# PCE-Galerkin 投影
# ----------------------------------------------------------------------
def stochastic_galerkin_projection(a_pce_coeffs, psi_gram, f_deterministic):
    """构造随机 Galerkin 系统.

    输入:
        a_pce_coeffs : (M,) 扩散系数 PCE 展开系数
        psi_gram     : (M, M, M) 三阶张量 C_{αβγ} = E[Ψ_α Ψ_β Ψ_γ]
        f_deterministic : (n_free,) 确定性载荷
    返回:
        K_sg    : (n_free*M, n_free*M) 随机 Galerkin 刚度
        F_sg    : (n_free*M,)
    """
    M = len(a_pce_coeffs)
    n_free = len(f_deterministic)
    K_sg = sparse.lil_matrix((n_free * M, n_free * M))
    F_sg = np.zeros(n_free * M)

    # 仅非零 C_{αβγ} 贡献
    for alpha in range(M):
        for beta in range(M):
            if abs(a_pce_coeffs[beta]) < 1e-14:
                continue
            coeff = 0.0
            for gamma in range(M):
                coeff += a_pce_coeffs[gamma] * psi_gram[alpha, beta, gamma]
            if abs(coeff) < 1e-14:
                continue
            # 块 (alpha, beta) = coeff * K (简化为恒等 K 近似)
            for i in range(n_free):
                K_sg[alpha * n_free + i, beta * n_free + i] += coeff
                F_sg[alpha * n_free + i] += (1.0 if alpha == 0 else 0.0) * \
                    f_deterministic[i]
    return K_sg.tocsr(), F_sg


# ----------------------------------------------------------------------
# 误差估计 (源自 387 中 L1/L2/H1/max 误差)
# ----------------------------------------------------------------------
def l2_error(nodes, u_num, u_exact):
    """梯形法则 L2 误差."""
    h = np.diff(nodes)
    e2 = 0.5 * (u_num[:-1] - u_exact[:-1]) ** 2 + \
         0.5 * (u_num[1:] - u_exact[1:]) ** 2
    return float(np.sqrt(np.sum(e2 * h)))


def h1_semi_error(nodes, u_num, u_exact):
    """H1 半范数误差 (导数 L2)."""
    h = np.diff(nodes)
    du_num = np.diff(u_num) / h
    du_ex = np.diff(u_exact) / h
    return float(np.sqrt(np.sum((du_num - du_ex) ** 2 * h)))


def max_error(u_num, u_exact):
    return float(np.max(np.abs(u_num - u_exact)))
