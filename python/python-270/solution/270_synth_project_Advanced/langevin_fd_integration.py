"""
langevin_fd_integration.py
==========================

Langevin 动力学 + 高阶有限差分离散化。

物理背景
--------
自旋玻璃的连续自旋版本可以用 Langevin 动力学描述:

    dS_i / dt = -Gamma * delta H / delta S_i + sqrt(2*Gamma*T) * eta_i(t)

其中:
- S_i 是连续自旋变量 (或投影到 [-1, 1])
- Gamma 是动力学系数
- H 是哈密顿量
- eta_i(t) 是高斯白噪声, <eta_i(t)> = 0, <eta_i(t) eta_j(t')> = delta_{ij} delta(t-t')

对于 Ising 自旋 S_i in {-1, +1}, Langevin 动力学退化为
Glauber 动力学 / Metropolis Monte Carlo。

高阶有限差分格式
----------------
对于哈密顿量中的局部场:
    h_i = sum_{j in nbr(i)} J_{ij} * S_j

这是离散 Laplacian 的推广:
    h_i = (Delta_d S)_i (对于均匀 J = 1 的情况)

在空间连续极限下:
    dS/dt = D * Delta S + noise

二阶中心差分 (O(h^2)):
    (Delta S)_i = S_{i+1} + S_{i-1} - 2*S_i (1D)
    在 3D: (Delta S)_{ijk} = S_{i+1,j,k} + S_{i-1,j,k} + S_{i,j+1,k}
                              + S_{i,j-1,k} + S_{i,j,k+1} + S_{i,j,k-1} - 6*S_{ijk}

四阶中心差分 (O(h^4)):
    (Delta4 S)_{ijk} = sum_{mu=x,y,z} [
        -(S_{i+2e_mu} + S_{i-2e_mu}) + 16*(S_{i+e_mu} + S_{i-e_mu}) - 30*S_i
    ] / 12

稳定性分析
----------
前向 Euler + 二阶差分:
    S^{n+1} = S^n + dt * D * Delta2 * S^n
    稳定性: dt <= h^2 / (2*d*D) = 1 / (6*D)  (对 d=3, h=1)

前向 Euler + 四阶差分:
    稳定性: dt <= 3*h^4 / (4*d*D*(8*h^2+...)) ≈ 1/(8*D)

本模块核心算法来源于 seed project:
- 002_advection_pde: 对流 PDE 的守恒量与初始条件
  (映射为 Langevin 方程的守恒量与初值)
- 842_ozone2_ode: 臭氧 ODE 的刚性系统
  (映射为 Langevin 动力学的刚性时间积分)
- 646_laplace_radial_exact: 径向 Laplace 精确解
  (映射为均匀耦合下的精确有限差分解)
"""

import numpy as np
from typing import Dict, Tuple, Optional
from spin_lattice_geometry import CubicLattice3D
from spin_glass_couplings import effective_local_field


# =====================================================================
#  有限差分算子
# =====================================================================

def laplacian_2nd_order(spins: np.ndarray,
                        couplings: Dict[Tuple[int, int], float],
                        lattice: CubicLattice3D) -> np.ndarray:
    """
    二阶有限差分 Laplacian (作用于自旋配置):
        (L2 S)_i = sum_{j in nbr(i)} J_{ij} * S_j - z * J_avg * S_i

    对于均匀 J=1 情况, 这等价于标准离散 Laplacian:
        (Delta_2 S)_i = sum_{j in nbr(i)} S_j - 2*d * S_i

    参数
    ----
    spins : ndarray (L, L, L)
    couplings : dict
    lattice : CubicLattice3D

    返回
    ----
    L2_spins : ndarray (L, L, L)
        Laplacian 作用后的场
    """
    L = lattice.L
    L2 = np.zeros_like(spins, dtype=np.float64)

    for ix in range(L):
        for iy in range(L):
            for iz in range(L):
                n = lattice.coord_to_index(ix, iy, iz)
                h_eff = effective_local_field(spins, couplings, n, lattice)
                # 减去 on-site 项以构成 Laplacian
                z = len([m for m in lattice.neighbor_table[n] if m >= 0])
                J_avg = 0.0
                for m in lattice.neighbor_table[n]:
                    if m < 0:
                        continue
                    key = (min(n, m), max(n, m))
                    J_avg += abs(couplings.get(key, 0.0))
                J_avg = J_avg / z if z > 0 else 0.0
                L2[ix, iy, iz] = h_eff - z * J_avg * spins[ix, iy, iz]

    return L2


def laplacian_4th_order(spins: np.ndarray,
                        couplings: Dict[Tuple[int, int], float],
                        lattice: CubicLattice3D) -> np.ndarray:
    """
    四阶有限差分 Laplacian:
        (L4 S)_i = sum_{mu} [-(S_{i+2e_mu}+S_{i-2e_mu}) + 16*(S_{i+e_mu}+S_{i-e_mu}) - 30*S_i]/12

    在 PBC 下, 使用模索引获取 S_{i±2e_mu}。
    """
    L = lattice.L
    L4 = np.zeros_like(spins, dtype=np.float64)

    for ix in range(L):
        for iy in range(L):
            for iz in range(L):
                val = 0.0
                # x 方向
                ixp = (ix + 1) % L
                ixm = (ix - 1) % L
                ixp2 = (ix + 2) % L
                ixm2 = (ix - 2) % L
                val += (-spins[ixp2, iy, iz] + 16 * spins[ixp, iy, iz]
                        - spins[ixm2, iy, iz] + 16 * spins[ixm, iy, iz]
                        - 30 * spins[ix, iy, iz]) / 12.0

                # y 方向
                iyp = (iy + 1) % L
                iym = (iy - 1) % L
                iyp2 = (iy + 2) % L
                iym2 = (iy - 2) % L
                val += (-spins[ix, iyp2, iz] + 16 * spins[ix, iyp, iz]
                        - spins[ix, iym2, iz] + 16 * spins[ix, iym, iz]
                        - 30 * spins[ix, iy, iz]) / 12.0

                # z 方向
                izp = (iz + 1) % L
                izm = (iz - 1) % L
                izp2 = (iz + 2) % L
                izm2 = (iz - 2) % L
                val += (-spins[ix, iy, izp2] + 16 * spins[ix, iy, izp]
                        - spins[ix, iy, izm2] + 16 * spins[ix, iy, izm]
                        - 30 * spins[ix, iy, iz]) / 12.0

                L4[ix, iy, iz] = val

    return L4


# =====================================================================
#  Langevin 时间积分器
# =====================================================================

class LangevinIntegrator:
    """
    Langevin 方程的时间积分器。

    dS/dt = F(S) + sqrt(2*T) * eta(t)

    其中 F(S)_i = -delta H / delta S_i = h_i^{eff} (有效局部场)

    参数
    ----
    lattice : CubicLattice3D
    temperature : float
        温度 T
    dt : float
        时间步长
    fd_order : int, default 2
        有限差分阶数 (2 或 4)
    method : str, default "euler_maruyama"
        积分方法: "euler_maruyama", "rk2_stochastic", "heun"
    """

    def __init__(self,
                 lattice: CubicLattice3D,
                 temperature: float,
                 dt: float = 0.001,
                 fd_order: int = 2,
                 method: str = "euler_maruyama",
                 rng: Optional[np.random.Generator] = None):
        if temperature < 0:
            raise ValueError(f"温度不能为负: T={temperature}")
        if dt <= 0:
            raise ValueError(f"时间步长必须 > 0: dt={dt}")
        if fd_order not in (2, 4):
            raise ValueError(f"有限差分阶数必须为 2 或 4: order={fd_order}")

        self.lattice = lattice
        self.temperature = temperature
        self.dt = dt
        self.fd_order = fd_order
        self.method = method
        self.rng = rng if rng is not None else np.random.default_rng()

        # 稳定性检查
        self._check_stability()

    def _check_stability(self):
        """检查时间步长是否满足 von Neumann 稳定性条件"""
        # 最坏情况 D = 1 (单位耦合)
        D_max = 1.0
        if self.fd_order == 2:
            dt_crit = 1.0 / (3.0 * D_max)  # 3D
        else:
            dt_crit = 1.0 / (8.0 * D_max)

        if self.dt > dt_crit:
            self.stability_warning = (
                f"dt={self.dt:.4f} > dt_crit={dt_crit:.4f}, "
                f"可能不稳定 (fd_order={self.fd_order})"
            )
        else:
            self.stability_warning = None

    def compute_force(self, spins: np.ndarray,
                      couplings: Dict[Tuple[int, int], float]) -> np.ndarray:
        """
        计算确定性力 F(S) = h^{eff} (局部有效场)
        """
        L = self.lattice.L
        force = np.zeros_like(spins, dtype=np.float64)
        for ix in range(L):
            for iy in range(L):
                for iz in range(L):
                    n = self.lattice.coord_to_index(ix, iy, iz)
                    force[ix, iy, iz] = effective_local_field(
                        spins, couplings, n, self.lattice
                    )
        return force

    def step(self, spins: np.ndarray,
             couplings: Dict[Tuple[int, int], float]) -> np.ndarray:
        """
        执行一个 Langevin 时间步。

        Euler-Maruyama 方法:
            S^{n+1} = S^n + dt * F(S^n) + sqrt(2*T*dt) * Z^n
        其中 Z^n ~ N(0, 1) 是标准高斯随机变量

        Heun 方法 (随机 Heun):
            S_tilde = S^n + dt * F(S^n) + sqrt(2*T*dt) * Z^n
            S^{n+1} = S^n + (dt/2) * [F(S^n) + F(S_tilde)] + sqrt(2*T*dt) * Z^n
        """
        L = self.lattice.L
        shape = spins.shape

        if self.method == "euler_maruyama":
            force = self.compute_force(spins, couplings)
            noise = self.rng.standard_normal(shape)
            noise_amplitude = np.sqrt(2.0 * self.temperature * self.dt)
            spins_new = spins + self.dt * force + noise_amplitude * noise

        elif self.method == "heun":
            force_n = self.compute_force(spins, couplings)
            noise = self.rng.standard_normal(shape)
            noise_amplitude = np.sqrt(2.0 * self.temperature * self.dt)

            # 预测步
            spins_tilde = spins + self.dt * force_n + noise_amplitude * noise

            # 投影到 [-1, 1] (保持自旋有界)
            spins_tilde = np.clip(spins_tilde, -1.0, 1.0)

            # 校正步
            force_tilde = self.compute_force(spins_tilde, couplings)
            spins_new = spins + 0.5 * self.dt * (force_n + force_tilde) + noise_amplitude * noise

        elif self.method == "rk2_stochastic":
            # 随机 RK2 (Runge-Kutta 2 阶)
            force_n = self.compute_force(spins, couplings)
            noise = self.rng.standard_normal(shape)
            noise_amplitude = np.sqrt(2.0 * self.temperature * self.dt)

            # 中间步
            k1 = self.dt * force_n + noise_amplitude * noise
            spins_mid = np.clip(spins + 0.5 * k1, -1.0, 1.0)

            force_mid = self.compute_force(spins_mid, couplings)
            k2 = self.dt * force_mid + noise_amplitude * noise

            spins_new = spins + k2

        else:
            raise ValueError(f"不支持的积分方法: {self.method}")

        # 保持自旋有界 (连续自旋版本)
        spins_new = np.clip(spins_new, -1.0, 1.0)

        return spins_new

    # -----------------------------------------------------------------
    #  能量守恒检查
    # -----------------------------------------------------------------

    def check_energy_drift(self, spins: np.ndarray,
                           couplings: Dict[Tuple[int, int], float],
                           n_steps: int = 100) -> Dict[str, float]:
        """
        在保守系统 (T=0) 中检查能量漂移。

        返回
        ----
        result : dict
            E_initial, E_final, drift, relative_drift
        """
        from spin_lattice_geometry import total_energy

        E_initial = total_energy(spins, couplings, self.lattice)
        spins_current = spins.copy()
        original_T = self.temperature
        self.temperature = 0.0  # 保守系统

        for _ in range(n_steps):
            spins_current = self.step(spins_current, couplings)

        self.temperature = original_T
        E_final = total_energy(spins_current, couplings, self.lattice)

        drift = E_final - E_initial
        rel_drift = drift / (abs(E_initial) + 1e-15)

        return {
            "E_initial": E_initial,
            "E_final": E_final,
            "drift": drift,
            "relative_drift": rel_drift,
            "n_steps": n_steps,
        }
