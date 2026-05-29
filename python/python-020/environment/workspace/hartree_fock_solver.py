# -*- coding: utf-8 -*-
"""
hartree_fock_solver.py
自洽Hartree-Fock求解器

核心物理：
  在分数量子霍尔效应中，虽然Laughlin波函数给出了强关联基态的
  优秀变分试探函数，但理解微扰下的准粒子激发仍需平均场理论。

  Hartree-Fock 自洽场方程（单粒子）：
      H_{HF} |φ_i⟩ = ε_i |φ_i⟩

  其中有效哈密顿量为：
      H_{HF} = H_0 + V_H + V_F

  动能+磁场项：
      H_0 = (1/2m*) [p + eA(r)]²

  Hartree（直接）势：
      V_H(r) = ∫ d²r' V_C(r - r') ρ(r')
    其中库仑势 V_C(r) = e²/(4πε_0ε_r |r|)，
    密度 ρ(r) = Σ_{i=occ} |φ_i(r)|²

  Fock（交换）势（积分算符）：
      V_F φ_i(r) = - Σ_{j=occ} ∫ d²r' V_C(r - r') φ_j^*(r') φ_i(r') φ_j(r)

  自洽迭代：
      1. 猜测初始单粒子轨道 {φ_i^{(0)}}
      2. 构造密度矩阵 ρ^{(k)} 和有效势 V_{HF}^{(k)}
      3. 对角化 H_{HF}^{(k)} 得到新轨道 {φ_i^{(k+1)}}
      4. 若 ||ρ^{(k+1)} - ρ^{(k)}|| < tol，收敛；否则回到步骤2

本模块融合原项目：
  - 1221_test_nonlin（非线性方程组Newton迭代）
  - 151_cg_ne（共轭梯度法求解正规方程）
"""
import numpy as np
from utils import condition_number, fermi_dirac, H_BAR, landau_level_energy

# ============================================================================
# 1. 共轭梯度法求解正规方程 A^T A x = A^T b（融合原项目 151_cg_ne）
# ============================================================================

def cg_ne_solve(A, b, x0=None, max_iter=None, tol=1e-10):
    """
    使用共轭梯度法（CGNE）求解最小二乘问题：
        min ||Ax - b||²
    等价于求解正规方程：
        A^T A x = A^T b

    迭代格式：
        r_k = b - A x_k
        z_k = A^T r_k
        α_k = (z_k^T z_k) / ((A d_k)^T (A d_k))
        x_{k+1} = x_k + α_k d_k
        β_k = (z_{k+1}^T z_{k+1}) / (z_k^T z_k)
        d_{k+1} = z_{k+1} + β_k d_k

    参数:
        A       : ndarray, shape (m, n)
        b       : ndarray, shape (m,)
        x0      : ndarray or None, 初始猜测
        max_iter: int or None, 最大迭代次数
        tol     : float, 残差收敛阈值

    返回:
        x       : ndarray, 解向量
        converged: bool, 是否收敛
        residual_norm: float, 最终残差范数
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    m, n = A.shape

    if max_iter is None:
        max_iter = n

    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.asarray(x0, dtype=float).copy()

    r = b - A @ x
    z = A.T @ r
    d = z.copy()

    for i in range(1, max_iter + 1):
        Ad = A @ d
        denom = np.dot(Ad, Ad)
        if denom < 1e-30:
            # 搜索方向退化
            return x, False, np.linalg.norm(r)

        alpha = np.dot(z, z) / denom
        x = x + alpha * d
        r = b - A @ x

        residual_norm = np.linalg.norm(r)
        if residual_norm < tol:
            return x, True, residual_norm

        z_old = z
        z = A.T @ r

        denom_beta = np.dot(z_old, z_old)
        if denom_beta < 1e-30:
            return x, False, residual_norm

        beta = np.dot(z, z) / denom_beta
        d = z + beta * d

    return x, False, residual_norm


# ============================================================================
# 2. Newton迭代求解非线性方程组（融合原项目 1221_test_nonlin）
# ============================================================================

def newton_solve(F, J, x0, tol=1e-8, max_iter=50, damping=0.8):
    """
    Newton法求解非线性方程组 F(x) = 0。

    迭代公式：
        x_{k+1} = x_k - λ J(x_k)^{-1} F(x_k)
    其中 λ ∈ (0,1] 为阻尼因子（用于全局收敛）。

    参数:
        F        : callable, F(x) → ndarray
        J        : callable, J(x) → ndarray (Jacobian矩阵)
        x0       : ndarray, 初始猜测
        tol      : float, 收敛阈值
        max_iter : int, 最大迭代次数
        damping  : float, 阻尼因子

    返回:
        x        : ndarray, 解
        converged: bool
        n_iter   : int, 实际迭代次数
    """
    x = np.asarray(x0, dtype=float).copy()
    for k in range(max_iter):
        fx = np.asarray(F(x), dtype=float)
        jac = np.asarray(J(x), dtype=float)

        # 检查Jacobian奇异性
        cond = condition_number(jac)
        if cond > 1e14:
            # 使用正则化
            jac = jac + 1e-10 * np.eye(jac.shape[0])

        try:
            delta = np.linalg.solve(jac, fx)
        except np.linalg.LinAlgError:
            # 若Jacobian奇异，使用最小二乘
            delta, _, _ = cg_ne_solve(jac, fx, tol=tol * 0.1)

        x_new = x - damping * delta

        if np.linalg.norm(x_new - x) < tol:
            return x_new, True, k + 1

        x = x_new

    return x, False, max_iter


# ============================================================================
# 3. Hartree-Fock自洽场求解器
# ============================================================================

def coulomb_interaction_2d(r, epsilon_r=12.0):
    """
    二维电子气的屏蔽库仑相互作用：
        V(r) = e² / (4π ε_0 ε_r r)

    在GaAs中 ε_r ≈ 12.9，这里取 ε_r = 12.0。
    为避免 r→0 发散，引入短程截断：
        V(r) = e² / (4π ε_0 ε_r √(r² + a²))
    其中 a 为短程截断长度。
    """
    r = np.asarray(r, dtype=float)
    a_cutoff = 0.01  # 短程截断
    r_safe = np.sqrt(r ** 2 + a_cutoff ** 2)
    # 自然单位制下 e²/(4πε_0) = 1（已归一化）
    return 1.0 / (epsilon_r * r_safe)


def build_hartree_fock_matrix(
    basis_orbitals, grid_x, grid_y, occupied_indices, dx, dy, epsilon_r=12.0, mixing=0.5
):
    """
    构建Hartree-Fock有效哈密顿矩阵。

    在离散格点上，单粒子基函数 {φ_α(r)} 展开，
    HF矩阵元为：
        (H_{HF})_{αβ} = T_{αβ} + V_{ext,αβ} + Σ_{j∈occ} [ (αβ|jj) - (αj|jβ) ]

    其中两电子积分（Slater记号）：
        (pq|rs) = ∫∫ d²r d²r' φ_p^*(r) φ_q^*(r') V_C(r-r') φ_r(r') φ_s(r)

    参数:
        basis_orbitals  : ndarray, shape (N_basis, N_x, N_y), 基函数格点值
        grid_x, grid_y  : ndarray, 空间格点
        occupied_indices: list, 占据态指标
        dx, dy          : float, 格点间距
        epsilon_r       : float, 相对介电常数
        mixing          : float, 密度混合参数

    返回:
        H_HF            : ndarray, (N_basis, N_basis)
        density         : ndarray, (N_x, N_y), 电子密度
    """
    N_basis = basis_orbitals.shape[0]
    Nx, Ny = grid_x.shape

    # 构造密度矩阵（占据态求和）
    density = np.zeros((Nx, Ny), dtype=float)
    for idx in occupied_indices:
        if idx < 0 or idx >= N_basis:
            raise ValueError(f"占据态指标 {idx} 超出范围 [0, {N_basis})")
        phi = basis_orbitals[idx]
        density += np.abs(phi) ** 2

    # Hartree 势：V_H(r) = ∫ d²r' V_C(r-r') ρ(r')
    # 用离散卷积计算
    V_H = np.zeros((Nx, Ny), dtype=float)
    for ix in range(Nx):
        for iy in range(Ny):
            dr_x = grid_x - grid_x[ix, iy]
            dr_y = grid_y - grid_y[ix, iy]
            dr = np.sqrt(dr_x ** 2 + dr_y ** 2)
            V_c = coulomb_interaction_2d(dr, epsilon_r)
            V_H[ix, iy] = np.sum(V_c * density) * dx * dy

    # 构建HF矩阵（简化版：只包含Hartree项和单粒子项，忽略Fock交换项的完整计算）
    # 对于大体系，完整Fock项的计算量为 O(N_basis^4)，这里采用近似
    H_HF = np.zeros((N_basis, N_basis), dtype=complex)

    # 单粒子项（Landau能级能量对角）
    for alpha in range(N_basis):
        H_HF[alpha, alpha] += landau_level_energy(alpha // 2, 10.0, 1.0)  # 简化的能级

    # Hartree 贡献：对角元加上密度-势耦合
    for alpha in range(N_basis):
        phi_a = basis_orbitals[alpha]
        H_HF[alpha, alpha] += np.sum(np.conj(phi_a) * V_H * phi_a) * dx * dy

    # 非对角元（简化近似：只保留最近邻耦合）
    for alpha in range(N_basis):
        for beta in range(alpha + 1, N_basis):
            # 简化：假设基函数正交，非对角元由微扰产生
            overlap = np.sum(np.conj(basis_orbitals[alpha]) * basis_orbitals[beta]) * dx * dy
            H_HF[alpha, beta] = 0.01 * overlap
            H_HF[beta, alpha] = np.conj(H_HF[alpha, beta])

    return H_HF, density


def self_consistent_hf(
    N_electrons, N_basis, B, lB, grid_x, grid_y,
    max_iter=30, tol=1e-6, epsilon_r=12.0, mixing=0.5
):
    """
    自洽Hartree-Fock迭代求解。

    参数:
        N_electrons : int, 电子数
        N_basis     : int, 基函数数
        B           : float, 磁场
        lB          : float, 磁长度
        grid_x, grid_y : ndarray, 空间格点
        max_iter    : int, 最大自洽迭代数
        tol         : float, 收敛阈值
        epsilon_r   : float, 介电常数
        mixing      : float, 密度混合参数 [0,1]

    返回:
        energies    : ndarray, 单粒子能级
        orbitals    : ndarray, 单粒子轨道
        density     : ndarray, 最终电子密度
        converged   : bool
    """
    from landau_levels import landau_orbital_wavefunction

    Nx, Ny = grid_x.shape
    dx = grid_x[1, 0] - grid_x[0, 0] if Nx > 1 else 1.0
    dy = grid_y[0, 1] - grid_y[0, 0] if Ny > 1 else 1.0

    # ========== HOLE 2 START ==========
    # TODO: 构建初始基函数（最低Landau能级的一部分轨道）
    #
    # 需要完成：
    #   1. 创建 basis_orbitals 数组，形状为 (N_basis, Nx, Ny)，dtype=complex
    #   2. 从 grid_x 和 grid_y 构造复坐标网格 z_grid
    #   3. 将 basis 指标 alpha 映射到 Landau 能级指标 (n, m)
    #      提示：alpha 与 (n, m) 的映射关系需确保覆盖前 N_basis 个最低能级的轨道
    #   4. 对每个 alpha，调用 landau_orbital_wavefunction(n, m, z_grid, lB)
    #      填充 basis_orbitals[alpha]
    #
    # 关键科学知识点：
    #   - Landau能级指标 n = 0, 1, 2, ...
    #   - 角动量量子数 m 在每个能级中的取值范围
    #   - 映射需与 landau_orbital_wavefunction 的参数定义保持一致
    raise NotImplementedError("基函数初始构造待实现")
    # ========== HOLE 2 END ==========

    # 初始占据态：最低的N_electrons个态
    occupied = list(range(min(N_electrons, N_basis)))

    density_old = np.zeros((Nx, Ny), dtype=float)

    for iteration in range(max_iter):
        H_HF, density_new = build_hartree_fock_matrix(
            basis_orbitals, grid_x, grid_y, occupied, dx, dy, epsilon_r, mixing
        )

        # 密度混合（防止振荡）
        if iteration > 0:
            density = mixing * density_new + (1.0 - mixing) * density_old
        else:
            density = density_new

        # 对角化HF矩阵
        H_HF = 0.5 * (H_HF + np.conj(H_HF.T))  # Hermitian化
        energies, C = np.linalg.eigh(H_HF)

        # 按能量排序并更新占据态
        sort_idx = np.argsort(energies.real)
        energies = energies[sort_idx]
        C = C[:, sort_idx]

        # 更新基函数展开系数（将新轨道投影回旧基）
        new_orbitals = np.zeros_like(basis_orbitals)
        for alpha in range(N_basis):
            for beta in range(N_basis):
                new_orbitals[alpha] += C[beta, alpha] * basis_orbitals[beta]
            # 归一化
            norm = np.sqrt(np.sum(np.abs(new_orbitals[alpha]) ** 2) * dx * dy)
            if norm > 1e-14:
                new_orbitals[alpha] /= norm

        basis_orbitals = new_orbitals
        occupied = list(range(min(N_electrons, N_basis)))

        # 检查收敛
        density_diff = np.max(np.abs(density - density_old))
        if density_diff < tol:
            return energies, basis_orbitals, density, True

        density_old = density.copy()

    return energies, basis_orbitals, density_old, False


# ============================================================================
# 4. 测试接口
# ============================================================================
def test_hartree_fock():
    """测试Hartree-Fock求解器。"""
    print("=" * 60)
    print("[hartree_fock_solver.py] Hartree-Fock求解器测试")
    print("=" * 60)

    # 测试CGNE
    print("\n1. 共轭梯度法(CGNE)测试:")
    A = np.array([[2.0, 1.0], [1.0, 3.0], [1.0, 1.0]], dtype=float)
    b = np.array([4.0, 5.0, 3.0], dtype=float)
    x, converged, res = cg_ne_solve(A, b, tol=1e-12)
    print(f"   解 x = {x}")
    print(f"   收敛: {converged}, 残差: {res:.2e}")
    print(f"   ||Ax-b|| = {np.linalg.norm(A @ x - b):.2e}")

    # 测试Newton迭代
    print("\n2. Newton迭代测试 (求解 x² - 2 = 0):")
    def F(x):
        return np.array([x[0] ** 2 - 2.0])
    def J(x):
        return np.array([[2.0 * x[0]]])
    x_sol, conv, nit = newton_solve(F, J, np.array([1.5]), tol=1e-12)
    print(f"   解 x = {x_sol[0]:.10f}")
    print(f"   收敛: {conv}, 迭代次数: {nit}")

    # 测试HF（小体系）
    print("\n3. 自洽Hartree-Fock测试 (简化2×2格点):")
    B = 10.0
    lB = np.sqrt(1.0 / B)
    x = np.linspace(-2.0 * lB, 2.0 * lB, 15)
    y = np.linspace(-2.0 * lB, 2.0 * lB, 15)
    grid_x, grid_y = np.meshgrid(x, y)

    energies, orbitals, density, conv = self_consistent_hf(
        N_electrons=2, N_basis=4, B=B, lB=lB,
        grid_x=grid_x, grid_y=grid_y,
        max_iter=10, tol=1e-4
    )
    print(f"   收敛: {conv}")
    print(f"   能级: {energies[:4].real}")
    print(f"   最大密度: {np.max(density):.6f}")

    print("\n[hartree_fock_solver.py] 测试完成。\n")


if __name__ == "__main__":
    test_hartree_fock()
