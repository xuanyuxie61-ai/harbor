"""
cascade_equation_solver.py — Rossi-Greisen 级联方程求解器
==========================================================

融合种子项目:
  - 437_flame_ode : ODE 时间推进算法 (Runge-Kutta)
  - 923_pwc_plot_1d : 分段函数表示

本模块求解描述电磁簇射发展的 Rossi-Greisen 级联方程 (近似 B):

  d phi_e(t, k) / dt = -(sigma_ion + sigma_b) * phi_e(t, k)
                        + 2 * sigma_p * integral_k^inf phi_gamma(t, k') *
                          (1/k') * dk'

  d phi_gamma(t, k) / dt = -sigma_p * phi_gamma(t, k)
                            + sigma_b * integral_k^inf phi_e(t, k') *
                              (1/k') * dk'

  其中:
    phi_e(t, k)    : 深度 t 处能量为 k 的电子/正电子通量
    phi_gamma(t, k): 深度 t 处能量为 k 的光子通量
    sigma_b(k)     : bremsstrahlung 截面 ∝ 1/X0
    sigma_p(k)     : 对产生截面 ∝ 1/X0
    sigma_ion(k)   : 电离损失 (∝ Ec / (k * X0))

简化 (矩方法):
  定义 N 阶矩: M_n(t) = integral k^n * phi(t, k) dk
  则级联方程变为矩方程:
    d M_n^e / dt = -(sigma_b + sigma_ion_n) * M_n^e + 2*sigma_p * M_n^gamma / (n+1)
    d M_n^gamma / dt = -sigma_p * M_n^gamma + sigma_b * M_n^e / (n+1)

数值方法:
  空间离散: 高阶有限差分 (finite_diff_schemes.py)
  时间推进: RK4 / 自适应 RK45
  积分核: Gauss-Legendre 求积 (quadrature.py)
"""

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from material_properties import MaterialSpec
from finite_diff_schemes import FiniteDiffOperator


@dataclass
class CascadeState:
    """级联状态 (某深度处的通量分布)"""
    electron_flux: List[float]       # 电子通量 phi_e(k_i)
    photon_flux: List[float]         # 光子通量 phi_gamma(k_i)
    energy_bins: List[float]         # 能量网格 k_i [MeV]
    depth_X0: float                  # 当前深度 [辐射长度]

    @property
    def total_electron_energy(self) -> float:
        """电子总能量 integral k * phi_e(k) dk"""
        return self._compute_moment(1, self.electron_flux)

    @property
    def total_photon_energy(self) -> float:
        """光子总能量"""
        return self._compute_moment(1, self.photon_flux)

    @property
    def total_energy(self) -> float:
        """总能量 (电子 + 光子)"""
        return self.total_electron_energy + self.total_photon_energy

    @property
    def particle_count(self) -> float:
        """总粒子数 integral phi(k) dk (零阶矩)"""
        return self._compute_moment(0, self.electron_flux) + \
               self._compute_moment(0, self.photon_flux)

    def _compute_moment(self, n: int, flux: List[float]) -> float:
        """计算通量的 n 阶矩: integral k^n * phi(k) dk (梯形法则)"""
        result = 0.0
        for i in range(len(self.energy_bins) - 1):
            k_mid = (self.energy_bins[i] + self.energy_bins[i + 1]) / 2.0
            f_mid = (flux[i] + flux[i + 1]) / 2.0
            dk = self.energy_bins[i + 1] - self.energy_bins[i]
            result += (k_mid ** n) * f_mid * dk
        return result


@dataclass
class CascadeSolution:
    """级联方程完整解"""
    states: List[CascadeState]      # 各深度的状态
    depths: List[float]             # 深度列表 [X0]
    energy_deposition: List[float]  # 每层的能量沉积 dE/dt [MeV/X0]
    shower_max_depth: float         # 簇射极大值深度 [X0]
    shower_max_energy: float        # 极大值处的沉积能量
    containment_depth: float        # 95% 能量包含深度 [X0]
    energy_conservation_error: float  # 能量守恒误差


class CascadeEquationSolver:
    """
    Rossi-Greisen 级联方程求解器

    求解策略:
    1. 能量空间离散: 对数均匀网格 k_i = k_min * r^i
    2. 深度方向推进: RK4 或自适应步长
    3. 积分核计算: 梯形法则 + 解析修正
    """

    def __init__(
        self,
        material: MaterialSpec,
        n_energy_bins: int = 32,
        energy_range: Tuple[float, float] = (1.0, 1e5),
        fd_order: int = 4,
    ):
        """
        参数:
          material     : 量能器材料
          n_energy_bins: 能量网格点数
          energy_range : (k_min, k_max) [MeV]
          fd_order     : 有限差分精度阶数
        """
        self.mat = material
        self.n_bins = n_energy_bins
        self.k_min, self.k_max = energy_range
        self.fd = FiniteDiffOperator(n_energy_bins)

        # 构建对数均匀能量网格
        log_k_min = math.log(max(self.k_min, 1e-6))
        log_k_max = math.log(self.k_max)
        self.energy_bins = [
            math.exp(log_k_min + (log_k_max - log_k_min) * i / (n_energy_bins - 1))
            for i in range(n_energy_bins)
        ]

    def solve(
        self,
        initial_energy: float,
        max_depth_X0: float = 30.0,
        depth_step: float = 0.1,
        particle_type: str = "electron",
    ) -> CascadeSolution:
        """
        求解级联方程

        初始条件:
          电子入射: phi_e(0, k) = E0 * delta(k - E0)
                    phi_gamma(0, k) = 0
          光子入射: phi_e(0, k) = 0
                    phi_gamma(0, k) = E0 * delta(k - E0)

        delta 函数用高斯近似:
          delta(k - E0) ≈ (1/(sigma*sqrt(2*pi))) * exp(-(k-E0)^2/(2*sigma^2))
          sigma = (k_max - k_min) / n_bins / 2
        """
        # 初始通量分布
        sigma_k = (self.k_max - self.k_min) / self.n_bins / 2.0
        sigma_k = max(sigma_k, initial_energy * 0.01)

        electron_flux = [0.0] * self.n_bins
        photon_flux = [0.0] * self.n_bins

        for i, k in enumerate(self.energy_bins):
            gauss = math.exp(-0.5 * ((k - initial_energy) / sigma_k) ** 2)
            gauss /= sigma_k * math.sqrt(2.0 * math.pi)
            if particle_type == "electron":
                electron_flux[i] = initial_energy * gauss
            else:
                photon_flux[i] = initial_energy * gauss

        initial_state = CascadeState(
            electron_flux=electron_flux,
            photon_flux=photon_flux,
            energy_bins=list(self.energy_bins),
            depth_X0=0.0,
        )

        # 深度推进 (RK4)
        states = [initial_state]
        energy_deposition = []
        current_depth = 0.0
        current_state = initial_state

        adaptive_step = depth_step
        min_depth_step = depth_step / 100.0
        max_depth_step = depth_step * 5.0

        while current_depth < max_depth_X0:
            # 自适应步长控制
            dt = min(adaptive_step, max_depth_X0 - current_depth)
            dt = max(dt, min_depth_step)

            # RK4 步进
            k1 = self._rhs(current_state)
            k2 = self._rhs_step(current_state, k1, dt / 2.0)
            k3 = self._rhs_step(current_state, k2, dt / 2.0)
            k4 = self._rhs_step(current_state, k3, dt)

            # 组合
            new_electron = [
                max(0.0, current_state.electron_flux[i]
                    + dt / 6.0 * (k1[0][i] + 2 * k2[0][i] + 2 * k3[0][i] + k4[0][i]))
                for i in range(self.n_bins)
            ]
            new_photon = [
                max(0.0, current_state.photon_flux[i]
                    + dt / 6.0 * (k1[1][i] + 2 * k2[1][i] + 2 * k3[1][i] + k4[1][i]))
                for i in range(self.n_bins)
            ]

            # 能量沉积: dE/dt = sigma_ion * integral k * phi_e(k) dk
            dep_energy = self._compute_energy_deposition(
                current_state, new_electron, new_photon, dt,
            )
            energy_deposition.append(dep_energy)

            current_depth += dt
            current_state = CascadeState(
                electron_flux=new_electron,
                photon_flux=new_photon,
                energy_bins=list(self.energy_bins),
                depth_X0=current_depth,
            )
            states.append(current_state)

            # 自适应步长调整 (基于通量变化率)
            flux_change = sum(
                abs(new_electron[i] - current_state.electron_flux[i])
                + abs(new_photon[i] - current_state.photon_flux[i])
                for i in range(self.n_bins)
            )
            total_flux = sum(new_electron) + sum(new_photon) + 1e-30
            rel_change = flux_change / total_flux

            if rel_change > 0.5:
                adaptive_step = max(adaptive_step * 0.5, min_depth_step)
            elif rel_change < 0.05:
                adaptive_step = min(adaptive_step * 1.5, max_depth_step)
            else:
                adaptive_step = depth_step

            # 终止条件: 总能量低于阈值
            if current_state.total_energy < initial_energy * 1e-6:
                break

        # 后处理: 计算 shower max 和 containment
        depths = [s.depth_X0 for s in states]
        shower_max_depth, shower_max_energy = self._find_shower_max(
            depths, energy_deposition,
        )
        containment_depth = self._find_containment_depth(
            depths, energy_deposition, initial_energy,
        )

        # 能量守恒检查
        total_deposited = sum(energy_deposition) * depth_step
        remaining = states[-1].total_energy
        conservation_error = abs(total_deposited + remaining - initial_energy) / initial_energy

        return CascadeSolution(
            states=states,
            depths=depths,
            energy_deposition=energy_deposition,
            shower_max_depth=shower_max_depth,
            shower_max_energy=shower_max_energy,
            containment_depth=containment_depth,
            energy_conservation_error=conservation_error,
        )

    def _rhs(self, state: CascadeState) -> Tuple[List[float], List[float]]:
        """
        计算级联方程右端项

        d phi_e / dt = -(sigma_b + sigma_ion) * phi_e
                        + 2 * sigma_p * integral_k^inf phi_gamma(k')/k' dk'

        d phi_gamma / dt = -sigma_p * phi_gamma
                            + sigma_b * integral_k^inf phi_e(k')/k' dk'
        """
        n = self.n_bins
        de_dt = [0.0] * n
        dg_dt = [0.0] * n

        for i in range(n):
            k = self.energy_bins[i]
            sigma_b = self._bremsstrahlung_cross_section(k)
            sigma_p = self._pair_production_cross_section(k)
            sigma_ion = self._ionization_loss_rate(k)

            # 积分核: integral_k^inf phi(k')/k' dk' (向后累积)
            int_gamma = self._downstream_integral(state.photon_flux, i)
            int_electron = self._downstream_integral(state.electron_flux, i)

            de_dt[i] = -(sigma_b + sigma_ion) * state.electron_flux[i] + \
                       2.0 * sigma_p * int_gamma
            dg_dt[i] = -sigma_p * state.photon_flux[i] + sigma_b * int_electron

        return de_dt, dg_dt

    def _rhs_step(
        self, state: CascadeState,
        k_vals: Tuple[List[float], List[float]], dt: float,
    ) -> Tuple[List[float], List[float]]:
        """在半步处计算右端项"""
        perturbed_electron = [
            max(0.0, state.electron_flux[i] + dt * k_vals[0][i])
            for i in range(self.n_bins)
        ]
        perturbed_photon = [
            max(0.0, state.photon_flux[i] + dt * k_vals[1][i])
            for i in range(self.n_bins)
        ]
        perturbed_state = CascadeState(
            electron_flux=perturbed_electron,
            photon_flux=perturbed_photon,
            energy_bins=state.energy_bins,
            depth_X0=state.depth_X0 + dt,
        )
        return self._rhs(perturbed_state)

    def _bremsstrahlung_cross_section(self, k_MeV: float) -> float:
        """
        Bremsstrahlung 截面 (近似) [1/X0]

        sigma_b ≈ 1/X0 * (1 - Ec/(3*k))  for k >> Ec
        sigma_b ≈ 0                        for k << Ec

        完整公式 (Bethe-Heitler):
          sigma_b = (alpha * r_e^2 * N_A * Z^2 / A) *
                    [(4/3)*ln(183*Z^{-1/3}) + 1/18]
        简化为 1/X0 (高能近似)。
        """
        if k_MeV <= self.mat.Ec_MeV:
            ratio = k_MeV / max(self.mat.Ec_MeV, 1e-10)
            return (1.0 / self.mat.X0_cm) * ratio * ratio * 0.5
        return (1.0 / self.mat.X0_cm) * (1.0 - self.mat.Ec_MeV / (3.0 * k_MeV))

    def _pair_production_cross_section(self, k_MeV: float) -> float:
        """
        对产生截面 [1/X0]

        sigma_p ≈ (7/9) * (1/X0) * (1 - Ec/(3*k))  for k >> 2*m_e*c^2
        sigma_p ≈ 0                                    for k < 2*m_e*c^2

        阈值: k > 2 * m_e * c^2 = 1.022 MeV
        高能极限: sigma_p → 7/(9*X0)
        """
        threshold = 2.0 * 0.511  # 2 * m_e * c^2 [MeV]
        if k_MeV <= threshold:
            return 0.0
        if k_MeV <= self.mat.Ec_MeV:
            return (7.0 / 9.0) * (1.0 / self.mat.X0_cm) * (k_MeV / self.mat.Ec_MeV) ** 2
        return (7.0 / 9.0) * (1.0 / self.mat.X0_cm) * (1.0 - self.mat.Ec_MeV / (3.0 * k_MeV))

    def _ionization_loss_rate(self, k_MeV: float) -> float:
        """
        电离损失率 dE/dx [MeV / (cm * X0)]

        在级联方程中表现为连续能量损失:
          (dE/dt)_ion ≈ Ec / (k * X0) * k = Ec / X0

        对于低能电子 (k < Ec): 损失率增大
        """
        return self.mat.ionization_loss_rate(k_MeV)

    def _downstream_integral(self, flux: List[float], start_idx: int) -> float:
        """
        计算下游积分: integral_k^inf phi(k')/k' dk'

        使用梯形法则在对数能量网格上:
          integral ≈ sum_{j>start} phi(k_j) / k_j * delta_k_j
        """
        result = 0.0
        for j in range(start_idx, len(flux)):
            k = self.energy_bins[j]
            if k < 1e-30:
                continue
            integrand = flux[j] / k
            if j < len(flux) - 1:
                dk = self.energy_bins[j + 1] - self.energy_bins[j]
            else:
                dk = self.energy_bins[j] - self.energy_bins[j - 1]
            result += integrand * dk
        return result

    def _compute_energy_deposition(
        self,
        old_state: CascadeState,
        new_electron: List[float],
        new_photon: List[float],
        dt: float,
    ) -> float:
        """
        计算深度步长内的能量沉积

        dE/dt = integral (dE/dx)_ion * phi_e(k) dk
              ≈ sigma_ion_eff * N_e(t)

        其中 sigma_ion_eff 为有效电离损失截面。
        """
        dep = 0.0
        for i in range(self.n_bins):
            k = self.energy_bins[i]
            sigma_ion = self.mat.Ec_MeV / self.mat.X0_cm
            e_mid = (old_state.electron_flux[i] + new_electron[i]) / 2.0
            dep += sigma_ion * e_mid
            if i < self.n_bins - 1:
                dk = self.energy_bins[i + 1] - self.energy_bins[i]
                dep *= dk
                dep /= max(dk, 1e-30)
        return dep

    def _find_shower_max(
        self, depths: List[float], depositions: List[float],
    ) -> Tuple[float, float]:
        """找到簇射极大值位置和能量"""
        if not depositions:
            return 0.0, 0.0
        max_dep = max(depositions)
        max_idx = depositions.index(max_dep)
        depth = depths[min(max_idx + 1, len(depths) - 1)]
        return depth, max_dep

    def _find_containment_depth(
        self, depths: List[float], depositions: List[float],
        total_energy: float, fraction: float = 0.95,
    ) -> float:
        """找到包含 fraction 能量的深度"""
        if not depositions or total_energy < 1e-30:
            return 0.0
        target = fraction * total_energy
        cumsum = 0.0
        step = depths[1] - depths[0] if len(depths) > 1 else 0.1
        for i, dep in enumerate(depositions):
            cumsum += dep * step
            if cumsum >= target:
                return depths[min(i + 1, len(depths) - 1)]
        return depths[-1] if depths else 0.0
