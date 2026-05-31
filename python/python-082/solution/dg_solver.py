# -*- coding: utf-8 -*-
"""
dg_solver.py
============
一维间断伽辽金（Discontinuous Galerkin, DG）谱元求解器。

源自种子项目 274_dg1d_maxwell 的 nodal DG 框架，
从电磁学 Maxwell 方程改造为结构力学一维应力波方程。

科学背景：
---------
一维弹性波在含损伤复合材料杆中的传播控制方程：
  ρ(x) * ∂²u/∂t² = ∂σ/∂x + f(x,t)
  σ(x,t) = E(x,d(x,t)) * ∂u/∂x
其中：
  u(x,t) — 轴向位移 [m]
  σ(x,t) — 轴向应力 [Pa]
  ρ(x)   — 等效密度 [kg/m³]
  E(x,d) — 含损伤等效弹性模量 [Pa]
  d(x,t) — 标量损伤变量 [0,1]
  f(x,t) — 体积力 [N/m³]

引入速度 v = ∂u/∂t 和应变 ε = ∂u/∂x，化为一阶双曲系统：
  ∂ε/∂t = ∂v/∂x
  ρ * ∂v/∂t = ∂σ/∂x + f
  σ = E(ε, d) * ε

对于线弹性情况（E 为常数），该系统的特征速度为 ±c = ±sqrt(E/ρ)。

DG 弱形式（单元 K_e = [x_e^L, x_e^R]）：
  在参考单元 [-1,1] 上映射 x = x_e(ξ)，Jacobian J_e = h_e/2。
  对试验函数 φ_i(ξ)，分部积分得到：
    ∫_{-1}^{1} ∂ε_h/∂t * φ_i * J_e dξ = [v_h^* * φ_i]_{-1}^{1} - ∫_{-1}^{1} v_h * φ_i' dξ
    ∫_{-1}^{1} ρ * ∂v_h/∂t * φ_i * J_e dξ = [σ_h^* * φ_i]_{-1}^{1} - ∫_{-1}^{1} σ_h * φ_i' dξ + ∫ f φ_i J_e dξ

数值通量（upwind/Roe 型，基于特征分解）：
  定义波阻抗 Z = sqrt(ρ * E)。
  在界面 x_{e+1/2} 处，左右状态 (σ^-, v^-) 和 (σ^+, v^+)。
  特征变量：
    w^+ = σ + Z * v   (右行波)
    w^- = σ - Z * v   (左行波)
  Upwind 通量取各自上游的特征变量重构：
    σ^* = (Z^+ * σ^- + Z^- * σ^+) / (Z^+ + Z^-) + (Z^+ * Z^-) / (Z^+ + Z^-) * (v^- - v^+)
    v^* = (Z^+ * v^+ + Z^- * v^-) / (Z^+ + Z^-) + (σ^- - σ^+) / (Z^+ + Z^-)

时间推进：采用低存储四阶 Runge-Kutta（LSERK45），
  du/dt = L(u) 的推进格式：
    u^{(0)} = u^n
    u^{(i)} = u^{(0)} + a_i * Δt * L(u^{(i-1)}), i=1,...,5
    u^{n+1} = u^{(5)}
  其中 a = [0, 1/4, 1/3, 1/2, 1]（经典 LSERK 系数）。
"""

import numpy as np
from typing import Callable, Optional, Tuple
from vandermonde_basis import (
    jacobi_gauss_lobatto_points,
    jacobi_gauss_lobatto_weights,
    vandermonde_matrix_1d,
    differentiation_matrix_1d,
)


class DGSolver1D:
    """
    一维间断伽辽金谱元求解器，用于复合材料杆中的应力波传播。
    """

    def __init__(self, num_elements: int, poly_order: int,
                 x_min: float, x_max: float,
                 rho_func: Callable, E_func: Callable,
                 refine_strength: float = 0.0):
        """
        Parameters
        ----------
        num_elements : int
            单元数。
        poly_order : int
            每个单元内的多项式阶数 N（节点数 N+1）。
        x_min, x_max : float
            计算域。
        rho_func : callable
            密度函数 ρ(x)，输入标量或数组，返回标量或数组。
        E_func : callable
            弹性模量函数 E(x)，输入标量或数组，返回标量或数组。
        refine_strength : float
            网格中部加密强度（见 mesh_geometry.py 的映射）。
        """
        if num_elements < 1:
            raise ValueError("num_elements must be >= 1.")
        if poly_order < 1:
            raise ValueError("poly_order must be >= 1.")

        self.num_elements = num_elements
        self.poly_order = poly_order
        self.x_min = x_min
        self.x_max = x_max
        self.L = x_max - x_min
        self.rho_func = rho_func
        self.E_func = E_func

        # 参考单元 GLL 节点与权重
        self.ref_nodes = jacobi_gauss_lobatto_points(poly_order)
        self.ref_weights = jacobi_gauss_lobatto_weights(self.ref_nodes)

        # 微分矩阵
        self.D_ref = differentiation_matrix_1d(poly_order, self.ref_nodes)

        # 生成物理网格与映射
        self._build_mesh(refine_strength)

        # 每个单元上的物理节点坐标、Jacobian、材料参数
        self._build_element_data()

        # LSERK45 系数
        self.lserk_a = np.array([0.0, -567301805773.0 / 1357537059087.0,
                                 -2404267990393.0 / 2016746695238.0,
                                 -3550918686646.0 / 2091501179385.0,
                                 -1275806237668.0 / 842570457699.0])
        self.lserk_b = np.array([1432997174477.0 / 9575080441755.0,
                                 5161836677717.0 / 13612068292357.0,
                                 1720146321549.0 / 2090206949498.0,
                                 3134564353537.0 / 4481467310338.0,
                                 2277821191437.0 / 14882151754819.0])
        self.lserk_c = np.array([0.0, 1432997174477.0 / 9575080441755.0,
                                 2526269341429.0 / 6820363962896.0,
                                 2006345519317.0 / 3224310063776.0,
                                 2802321613138.0 / 2924317926251.0])

        # 解变量初始化
        self.num_dof = num_elements * (poly_order + 1)
        self.strain = np.zeros(self.num_dof)   # ε
        self.velocity = np.zeros(self.num_dof)  # v
        self.stress = np.zeros(self.num_dof)   # σ = E * ε
        self.time = 0.0

    def _build_mesh(self, refine_strength: float):
        """生成非均匀网格节点坐标（单调安全映射）。"""
        xi_uniform = np.linspace(0.0, 1.0, self.num_elements + 1)
        a = refine_strength
        f_xi = xi_uniform + a * 0.5 * (1.0 - np.cos(np.pi * xi_uniform))
        f_1 = 1.0 + a * 0.5 * (1.0 - np.cos(np.pi))
        x_nodes = self.x_min + self.L * f_xi / f_1
        x_nodes[0] = self.x_min
        x_nodes[-1] = self.x_max
        self.elem_vertices = np.stack([x_nodes[:-1], x_nodes[1:]], axis=1)

    def _build_element_data(self):
        """计算各单元的物理节点、Jacobian、材料参数。"""
        Np = self.poly_order + 1
        self.elem_nodes = np.zeros((self.num_elements, Np))
        self.elem_jac = np.zeros(self.num_elements)
        self.elem_invjac = np.zeros(self.num_elements)
        self.elem_rho = np.zeros((self.num_elements, Np))
        self.elem_E = np.zeros((self.num_elements, Np))
        self.elem_Z = np.zeros((self.num_elements, Np))  # 波阻抗
        self.elem_mass_inv = np.zeros((self.num_elements, Np))

        for e in range(self.num_elements):
            xL, xR = self.elem_vertices[e]
            h_e = xR - xL
            J = h_e / 2.0
            self.elem_jac[e] = J
            self.elem_invjac[e] = 1.0 / J

            # 物理节点坐标
            x_phys = 0.5 * (xL + xR) + J * self.ref_nodes
            self.elem_nodes[e, :] = x_phys

            # 材料参数
            rho_e = self.rho_func(x_phys)
            E_e = self.E_func(x_phys)
            self.elem_rho[e, :] = rho_e
            self.elem_E[e, :] = E_e
            self.elem_Z[e, :] = np.sqrt(rho_e * E_e)

            # 质量矩阵逆（对角 lumped mass，GLL 积分天然对角）
            self.elem_mass_inv[e, :] = 1.0 / (J * self.ref_weights * rho_e)

    def get_dof_index(self, elem: int, node: int) -> int:
        """获取全局自由度索引。"""
        return elem * (self.poly_order + 1) + node

    def get_element_solution(self, elem: int):
        """提取单元局部的应变和速度。"""
        idx = slice(elem * (self.poly_order + 1), (elem + 1) * (self.poly_order + 1))
        return self.strain[idx], self.velocity[idx]

    def set_element_solution(self, elem: int, strain_loc: np.ndarray, velocity_loc: np.ndarray):
        """写入单元局部的应变和速度。"""
        idx = slice(elem * (self.poly_order + 1), (elem + 1) * (self.poly_order + 1))
        self.strain[idx] = strain_loc
        self.velocity[idx] = velocity_loc

    def compute_stress(self):
        """根据当前应变和损伤状态更新应力。"""
        self.stress = self.elem_E.flatten() * self.strain

    def _compute_interface_fluxes(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算所有单元界面的数值通量。
        返回全局数组 flux_v[iface], flux_sigma[iface]，iface 为单元之间的内部界面索引。
        同时处理左右边界。
        """
        Np = self.poly_order + 1
        # 界面数量 = num_elements + 1（含左右边界）
        num_interfaces = self.num_elements + 1
        flux_v = np.zeros(num_interfaces)
        flux_sigma = np.zeros(num_interfaces)

        for iface in range(num_interfaces):
            if iface == 0:
                # 左边界：假设无应力自由边界（σ^* = 0）或固定边界
                # 这里采用非反射边界（characteristic boundary）
                # 外部状态设为零入射波
                e = 0
                idx = e * Np
                sigma_in = self.stress[idx]
                v_in = self.velocity[idx]
                Z_in = self.elem_Z[e, 0]
                # 非反射：仅出射波离开，无入射波
                # w_out = sigma_in - Z_in * v_in（左行波离开）
                # 令入射 w_in = 0，则 sigma^* = w_out/2, v^* = -w_out/(2Z)
                w_out = sigma_in - Z_in * v_in
                flux_sigma[iface] = 0.5 * w_out
                flux_v[iface] = -0.5 * w_out / Z_in
            elif iface == num_interfaces - 1:
                # 右边界：非反射
                e = self.num_elements - 1
                idx = (e + 1) * Np - 1
                sigma_in = self.stress[idx]
                v_in = self.velocity[idx]
                Z_in = self.elem_Z[e, -1]
                # 右行波离开
                w_out = sigma_in + Z_in * v_in
                flux_sigma[iface] = 0.5 * w_out
                flux_v[iface] = 0.5 * w_out / Z_in
            else:
                # 内部界面
                eL = iface - 1
                eR = iface
                idxL = (eL + 1) * Np - 1
                idxR = eR * Np
                sigmaL = self.stress[idxL]
                sigmaR = self.stress[idxR]
                vL = self.velocity[idxL]
                vR = self.velocity[idxR]
                ZL = self.elem_Z[eL, -1]
                ZR = self.elem_Z[eR, 0]
                Z_sum = ZL + ZR
                if Z_sum < 1e-30:
                    Z_sum = 1e-30
                # Roe/upwind 通量（见模块文档）
                flux_sigma[iface] = (ZR * sigmaL + ZL * sigmaR) / Z_sum + (ZL * ZR) * (vL - vR) / Z_sum
                flux_v[iface] = (ZR * vR + ZL * vL) / Z_sum + (sigmaL - sigmaR) / Z_sum
        return flux_v, flux_sigma

    def _rhs_strain_velocity(self, strain: np.ndarray, velocity: np.ndarray,
                             f_func: Optional[Callable] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算应变和速度的右端项（空间离散算子 L）。
        返回 d(strain)/dt, d(velocity)/dt。
        """
        Np = self.poly_order + 1
        # 临时更新应力
        stress = self.elem_E.flatten() * strain

        dstrain_dt = np.zeros_like(strain)
        dvelocity_dt = np.zeros_like(velocity)

        flux_v, flux_sigma = self._compute_interface_fluxes_from_state(strain, velocity, stress)

        for e in range(self.num_elements):
            idx = slice(e * Np, (e + 1) * Np)
            strain_e = strain[idx]
            velocity_e = velocity[idx]
            stress_e = stress[idx]
            invJ = self.elem_invjac[e]
            mass_inv = self.elem_mass_inv[e, :]

            # 应变方程 RHS: ∂ε/∂t = ∂v/∂x
            # DG 弱形式：M * dε/dt = [v^* φ] - ∫ v φ' dx
            surface_term_v = np.zeros(Np)
            surface_term_v[0] = -flux_v[e]      # 左界面（法向 -1）
            surface_term_v[-1] = flux_v[e + 1]  # 右界面（法向 +1）
            volume_term_v = self.D_ref @ velocity_e * invJ
            dstrain_dt[idx] = mass_inv * (surface_term_v - self.ref_weights * volume_term_v)

            # 速度方程 RHS: ρ ∂v/∂t = ∂σ/∂x + f
            surface_term_s = np.zeros(Np)
            surface_term_s[0] = -flux_sigma[e]
            surface_term_s[-1] = flux_sigma[e + 1]
            volume_term_s = self.D_ref @ stress_e * invJ
            rhs_v = surface_term_s - self.ref_weights * volume_term_s

            if f_func is not None:
                x_e = self.elem_nodes[e, :]
                f_e = f_func(x_e, self.time)
                rhs_v += self.ref_weights * f_e * self.elem_jac[e]

            # 质量矩阵已含 ρ，mass_inv = 1/(J w ρ)
            dvelocity_dt[idx] = mass_inv * rhs_v

        return dstrain_dt, dvelocity_dt

    def _compute_interface_fluxes_from_state(self, strain: np.ndarray, velocity: np.ndarray,
                                              stress: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """从给定状态计算界面通量。"""
        Np = self.poly_order + 1
        num_interfaces = self.num_elements + 1
        flux_v = np.zeros(num_interfaces)
        flux_sigma = np.zeros(num_interfaces)

        for iface in range(num_interfaces):
            if iface == 0:
                e = 0
                idx = e * Np
                sigma_in = stress[idx]
                v_in = velocity[idx]
                Z_in = self.elem_Z[e, 0]
                w_out = sigma_in - Z_in * v_in
                flux_sigma[iface] = 0.5 * w_out
                flux_v[iface] = -0.5 * w_out / (Z_in + 1e-30)
            elif iface == num_interfaces - 1:
                e = self.num_elements - 1
                idx = (e + 1) * Np - 1
                sigma_in = stress[idx]
                v_in = velocity[idx]
                Z_in = self.elem_Z[e, -1]
                w_out = sigma_in + Z_in * v_in
                flux_sigma[iface] = 0.5 * w_out
                flux_v[iface] = 0.5 * w_out / (Z_in + 1e-30)
            else:
                eL = iface - 1
                eR = iface
                idxL = (eL + 1) * Np - 1
                idxR = eR * Np
                sigmaL = stress[idxL]
                sigmaR = stress[idxR]
                vL = velocity[idxL]
                vR = velocity[idxR]
                ZL = self.elem_Z[eL, -1]
                ZR = self.elem_Z[eR, 0]
                Z_sum = ZL + ZR
                if Z_sum < 1e-30:
                    Z_sum = 1e-30
                flux_sigma[iface] = (ZR * sigmaL + ZL * sigmaR) / Z_sum + (ZL * ZR) * (vL - vR) / Z_sum
                flux_v[iface] = (ZR * vR + ZL * vL) / Z_sum + (sigmaL - sigmaR) / Z_sum
        return flux_v, flux_sigma

    def step(self, dt: float, f_func: Optional[Callable] = None):
        """
        用 LSERK45 推进一个时间步。
        """
        Np = self.poly_order + 1
        strain0 = self.strain.copy()
        velocity0 = self.velocity.copy()

        strain_curr = strain0.copy()
        velocity_curr = velocity0.copy()

        for stage in range(5):
            self.strain = strain_curr
            self.velocity = velocity_curr
            self.compute_stress()
            rhs_strain, rhs_vel = self._rhs_strain_velocity(strain_curr, velocity_curr, f_func)

            if stage == 0:
                strain0 = strain_curr.copy()
                velocity0 = velocity_curr.copy()

            # LSERK 更新
            strain_curr = strain0 + self.lserk_b[stage] * dt * rhs_strain
            velocity_curr = velocity0 + self.lserk_b[stage] * dt * rhs_vel

        self.strain = strain_curr
        self.velocity = velocity_curr
        self.time += dt
        self.compute_stress()

    def run(self, t_final: float, dt: Optional[float] = None,
            f_func: Optional[Callable] = None,
            callback: Optional[Callable] = None) -> dict:
        """
        运行到终止时间 t_final。

        Returns
        -------
        history : dict
            包含时间序列、应变、速度、应力的历史数据。
        """
        if dt is None:
            # CFL 条件：dt <= CFL * min(h_e) / (N^2 * max(c))
            min_h = np.min(self.elem_vertices[:, 1] - self.elem_vertices[:, 0])
            max_c = np.max(self.elem_Z / self.elem_rho)
            cfl = 0.5
            dt = cfl * min_h / (self.poly_order ** 2 * max_c + 1e-30)

        num_steps = int(np.ceil(t_final / dt))
        dt = t_final / num_steps

        # 采样历史
        hist = {
            "time": [],
            "strain_max": [],
            "velocity_max": [],
            "stress_max": [],
            "energy_kinetic": [],
            "energy_strain": [],
        }

        for step in range(num_steps):
            self.step(dt, f_func)
            if callback is not None:
                callback(self.time, self)

            # 每 10 步记录一次
            if step % max(1, num_steps // 100) == 0:
                hist["time"].append(self.time)
                hist["strain_max"].append(np.max(np.abs(self.strain)))
                hist["velocity_max"].append(np.max(np.abs(self.velocity)))
                hist["stress_max"].append(np.max(np.abs(self.stress)))

                # 动能和应变能
                ke = 0.0
                se = 0.0
                for e in range(self.num_elements):
                    idx = slice(e * (self.poly_order + 1), (e + 1) * (self.poly_order + 1))
                    v_e = self.velocity[idx]
                    eps_e = self.strain[idx]
                    rho_e = self.elem_rho[e, :]
                    E_e = self.elem_E[e, :]
                    J = self.elem_jac[e]
                    ke += np.sum(J * self.ref_weights * rho_e * v_e ** 2)
                    se += np.sum(J * self.ref_weights * E_e * eps_e ** 2)
                hist["energy_kinetic"].append(0.5 * ke)
                hist["energy_strain"].append(0.5 * se)

        # 转换为 numpy 数组
        for key in hist:
            hist[key] = np.array(hist[key])
        return hist


if __name__ == "__main__":
    # 自测试：均匀杆中的应力波传播
    L = 1.0
    rho0 = 1600.0
    E0 = 100e9
    solver = DGSolver1D(
        num_elements=20, poly_order=3,
        x_min=0.0, x_max=L,
        rho_func=lambda x: np.full_like(np.asarray(x), rho0),
        E_func=lambda x: np.full_like(np.asarray(x), E0),
        refine_strength=0.5
    )

    # 初始条件：高斯脉冲位移 → 导出应变
    x_all = solver.elem_nodes.flatten()
    sigma0 = 1e6 * np.exp(-((x_all - 0.3 * L) ** 2) / (2 * (0.05 * L) ** 2))
    solver.stress = sigma0
    solver.strain = sigma0 / E0
    solver.velocity = np.zeros_like(solver.velocity)

    c = np.sqrt(E0 / rho0)
    t_final = 2 * L / c
    dt = 0.2 * (L / 20) / (3 ** 2 * c)
    hist = solver.run(t_final=t_final, dt=dt)
    print("DG solver self-test completed.")
    print("Final time:", solver.time)
    print("Max strain history:", hist["strain_max"][:5])
