"""
平流层臭氧化学动力学模型 (Stratospheric Ozone Chemical Kinetics Model)

本模块实现了平流层臭氧化学的完整反应网络，包括：
- Chapman 机制（基础光化学循环）
- NOx 催化破坏循环
- ClOx 催化破坏循环  
- HOx 催化破坏循环
- 三体反应与压强依赖反应

核心科学方程:
1. Chapman 光化学稳态:
   O2 + hν(λ<242nm) → O + O      (光解, 速率 J_O2)
   O + O2 + M → O3 + M            (三体反应, 速率 k1)
   O3 + hν → O2 + O(1D)           (光解 Hartley带, 速率 J_O3)
   O + O3 → 2O2                   (直接反应, 速率 k2)

2. 催化循环 (以 NOx 为例):
   NO + O3 → NO2 + O2             (k3)
   NO2 + O → NO + O2              (k4)
   净反应: O + O3 → 2O2

3. 连续性方程（对每个物种 i）:
   ∂n_i/∂t + ∇·(v n_i) = P_i - L_i + D∇²n_i
   其中 P_i 为化学生产率, L_i 为化学损失率, D 为涡旋扩散系数

4. 三体反应速率 (Troe 近似):
   k = k0[M] / (1 + k0[M]/k∞) × F^(1/(1+(log10(k0[M]/k∞)/N)²))
   其中 [M] 为第三体浓度, F = 0.6, N = 1.0

5. Arrhenius 温度依赖:
   k(T) = A × exp(-Ea / (R T))

融入原项目: 547_human_data (排放轮廓参数化), 545_house (几何结构定义)
"""

import numpy as np
from typing import Dict, Tuple, Optional

# 物理常数
R_GAS = 8.314  # J/(mol·K), 通用气体常数
N_A = 6.022e23  # Avogadro常数
H_PLANCK = 6.626e-34  # J·s
C_LIGHT = 2.998e8  # m/s
K_BOLTZMANN = 1.381e-23  # J/K


class StratosphericChemistry:
    """
    平流层臭氧化学动力学主类
    """

    def __init__(self, num_altitude_levels: int = 80,
                 t_min: float = 180.0, t_max: float = 270.0):
        """
        初始化化学模型

        Parameters
        ----------
        num_altitude_levels : int
            垂直高度层数
        t_min, t_max : float
            温度范围 (K)
        """
        if num_altitude_levels < 10:
            raise ValueError("高度层数必须 >= 10")
        if t_min <= 0 or t_max <= 0:
            raise ValueError("温度必须为正")

        self.nz = num_altitude_levels
        self.z = np.linspace(10000.0, 50000.0, num_altitude_levels)  # 10-50 km
        self.dz = self.z[1] - self.z[0]

        # 温度剖面 (标准大气近似)
        self.T = self._temperature_profile(self.z)
        self.T = np.clip(self.T, t_min, t_max)

        # 空气摩尔质量 (需要在压强计算之前定义)
        self.M_air = 0.0289644  # kg/mol, 干空气摩尔质量

        # 压强剖面 (barometric formula)
        self.P = self._pressure_profile(self.z)
        self.P = np.clip(self.P, 1.0, 1013.25)

        # 空气密度 (理想气体定律)
        self.rho_air = self.P * 100.0 / (R_GAS * self.T)  # mol/m³ -> *100 for hPa->Pa
        self.rho_air = np.clip(self.rho_air, 1e-6, 100.0)

        # 化学物种浓度 ( molec/cm³ )
        self.species = {
            'O': np.ones(num_altitude_levels) * 1e6,
            'O3': np.ones(num_altitude_levels) * 1e12,
            'O2': np.ones(num_altitude_levels) * 5e18,
            'N2': np.ones(num_altitude_levels) * 2e19,
            'NO': np.ones(num_altitude_levels) * 1e9,
            'NO2': np.ones(num_altitude_levels) * 5e8,
            'Cl': np.ones(num_altitude_levels) * 1e3,
            'ClO': np.ones(num_altitude_levels) * 5e2,
            'OH': np.ones(num_altitude_levels) * 1e6,
            'HO2': np.ones(num_altitude_levels) * 5e6,
            'H': np.ones(num_altitude_levels) * 1e2,
            'Br': np.ones(num_altitude_levels) * 1e2,
            'BrO': np.ones(num_altitude_levels) * 5e1,
        }

        # 反应速率参数 (A: cm³/molec/s, Ea: J/mol)
        self.reactions = self._init_reactions()

        # 光解速率 J (s⁻¹) 的近似垂直剖面
        self.J_rates = self._init_photolysis_rates()

        # 垂直涡旋扩散系数 Kzz (m²/s)
        self.Kzz = self._init_eddy_diffusion()

        # 边界排放轮廓 (融入 human_data 思想: 参数化轮廓)
        self.emission_profile = self._init_emission_profile()

    def _temperature_profile(self, z: np.ndarray) -> np.ndarray:
        """
        标准平流层温度剖面
        T(z) = T_tropo - L*(z - z_tropo)  for z < 20km
        T(z) = T_strat                   for 20km < z < 50km
        其中 L = 2.0 K/km (平流层逆温)
        """
        T = np.zeros_like(z)
        z_km = z / 1000.0
        for i, zk in enumerate(z_km):
            if zk <= 11.0:
                T[i] = 216.65
            elif zk <= 20.0:
                T[i] = 216.65 + 1.0 * (zk - 11.0)
            elif zk <= 32.0:
                T[i] = 216.65 + 1.0 * 9.0 + 2.8 * (zk - 20.0)
            elif zk <= 47.0:
                T[i] = 216.65 + 9.0 + 33.6 - 2.8 * (zk - 32.0)
            else:
                T[i] = 270.65 - 2.0 * (zk - 47.0)
        return T

    def _pressure_profile(self, z: np.ndarray) -> np.ndarray:
        """
        压强剖面 (barometric formula)
        P(z) = P0 * exp(-z/H)
        H = RT/(Mg) 为标高 (~7km)
        """
        g = 9.81  # m/s²
        H_scale = R_GAS * np.mean(self._temperature_profile(z)) / (self.M_air * g)
        P = 1013.25 * np.exp(-z / (H_scale * 1000.0))
        return P

    def _init_reactions(self) -> Dict:
        """
        初始化反应速率参数
        速率常数使用 Arrhenius 形式: k = A * exp(-Ea/(RT))
        单位: A 为 cm³ molec⁻¹ s⁻¹ (双分子) 或 cm⁶ molec⁻² s⁻¹ (三体)
        """
        reactions = {
            # Chapman 机制
            'R1': {'A': 6.0e-34, 'Ea': 0.0, 'type': 'termolecular',
                   'reactants': ['O', 'O2', 'M'], 'products': ['O3']},
            'R2': {'A': 8.0e-12, 'Ea': 2060.0 * R_GAS, 'type': 'bimolecular',
                   'reactants': ['O', 'O3'], 'products': ['O2', 'O2']},
            # NOx 催化
            'R3': {'A': 1.8e-12, 'Ea': 1370.0 * R_GAS, 'type': 'bimolecular',
                   'reactants': ['NO', 'O3'], 'products': ['NO2', 'O2']},
            'R4': {'A': 9.3e-12, 'Ea': 0.0, 'type': 'bimolecular',
                   'reactants': ['NO2', 'O'], 'products': ['NO', 'O2']},
            'R5': {'A': 1.0e-20, 'Ea': 0.0, 'type': 'termolecular',
                   'reactants': ['NO', 'O', 'M'], 'products': ['NO2']},
            # ClOx 催化
            'R6': {'A': 2.9e-11, 'Ea': 260.0 * R_GAS, 'type': 'bimolecular',
                   'reactants': ['Cl', 'O3'], 'products': ['ClO', 'O2']},
            'R7': {'A': 2.8e-11, 'Ea': 0.0, 'type': 'bimolecular',
                   'reactants': ['ClO', 'O'], 'products': ['Cl', 'O2']},
            # HOx 催化
            'R8': {'A': 1.7e-12, 'Ea': 940.0 * R_GAS, 'type': 'bimolecular',
                   'reactants': ['OH', 'O3'], 'products': ['HO2', 'O2']},
            'R9': {'A': 3.0e-11, 'Ea': 200.0 * R_GAS, 'type': 'bimolecular',
                   'reactants': ['HO2', 'O'], 'products': ['OH', 'O2']},
            'R10': {'A': 7.2e-11, 'Ea': 0.0, 'type': 'bimolecular',
                    'reactants': ['OH', 'O'], 'products': ['H', 'O2']},
            # BrOx 催化
            'R11': {'A': 1.7e-12, 'Ea': 600.0 * R_GAS, 'type': 'bimolecular',
                    'reactants': ['Br', 'O3'], 'products': ['BrO', 'O2']},
            'R12': {'A': 1.5e-11, 'Ea': 0.0, 'type': 'bimolecular',
                    'reactants': ['BrO', 'O'], 'products': ['Br', 'O2']},
        }
        return reactions

    def _init_photolysis_rates(self) -> Dict[str, np.ndarray]:
        """
        初始化光解速率 J (s⁻¹) 的垂直剖面
        J(z) = J0 * exp(-τ(z) * sec(θ))
        其中 τ 为光学厚度, θ 为太阳天顶角
        """
        J = {}
        theta_sza = np.deg2rad(45.0)  # 太阳天顶角 45°
        sec_theta = 1.0 / np.cos(theta_sza)

        # O2 光解 (Schumann-Runge bands, λ < 242nm)
        tau_o2 = np.cumsum(self.species['O2'] * 1e6 * self.dz * 1e-5)  # 简化光学厚度
        tau_o2 = np.clip(tau_o2, 0.0, 50.0)
        J['J_O2'] = 1e-10 * np.exp(-tau_o2 * sec_theta)

        # O3 光解 (Hartley band, λ ~ 200-310nm)
        tau_o3 = np.cumsum(self.species['O3'] * 1e-20 * self.dz * 1e-2)
        tau_o3 = np.clip(tau_o3, 0.0, 30.0)
        J['J_O3'] = 1e-2 * np.exp(-tau_o3 * sec_theta)

        # NO2 光解
        J['J_NO2'] = 1e-2 * np.exp(-tau_o3 * sec_theta * 0.5)

        # ClO 光解 (简化)
        J['J_ClO'] = 1e-5 * np.ones(self.nz)

        return J

    def _init_eddy_diffusion(self) -> np.ndarray:
        """
        垂直涡旋扩散系数 Kzz (m²/s)
        平流层典型值: 0.1 - 10 m²/s
        Kzz(z) = K0 + K1 * exp(-(z - z_max)² / σ²)
        """
        z_km = self.z / 1000.0
        Kzz = 0.5 + 2.0 * np.exp(-((z_km - 25.0) / 10.0) ** 2)
        Kzz = np.clip(Kzz, 0.01, 50.0)
        return Kzz

    def _init_emission_profile(self) -> Dict[str, np.ndarray]:
        """
        初始化人为排放源项的垂直分布轮廓
        融入 547_human_data 思想: 参数化轮廓函数
        使用高斯型轮廓描述 CFCs / N2O 等源气体的向上输送
        """
        profiles = {}
        z_km = self.z / 1000.0

        # N2O 排放 (地表源, 向上扩散)
        profiles['N2O'] = np.exp(-((z_km - 0.0) / 5.0) ** 2)
        profiles['N2O'] = profiles['N2O'] / (np.sum(profiles['N2O']) + 1e-30)

        # CFC-11 排放轮廓
        profiles['CFC11'] = np.exp(-((z_km - 0.0) / 3.0) ** 2)
        profiles['CFC11'] = profiles['CFC11'] / (np.sum(profiles['CFC11']) + 1e-30)

        # NOx 飞机排放 (集中在 9-12 km, 但平流层底部有贡献)
        profiles['NOx_aircraft'] = np.exp(-((z_km - 11.0) / 2.0) ** 2)
        profiles['NOx_aircraft'] = profiles['NOx_aircraft'] / (np.sum(profiles['NOx_aircraft']) + 1e-30)

        return profiles

    def arrhenius_rate(self, A: float, Ea: float, T: float) -> float:
        """
        Arrhenius 反应速率常数计算
        k = A * exp(-Ea / (R * T))

        Parameters
        ----------
        A : float
            指前因子 (cm³ molec⁻¹ s⁻¹)
        Ea : float
            活化能 (J/mol)
        T : float
            温度 (K)

        Returns
        -------
        k : float
            速率常数
        """
        if T <= 0:
            raise ValueError("温度必须大于零")
        k = A * np.exp(-Ea / (R_GAS * T))
        return max(k, 1e-40)

    def termolecular_rate(self, k0: float, kinf: float, M: float,
                          T: float, F: float = 0.6) -> float:
        """
        Troe 近似三体反应速率
        k = k0[M] / (1 + k0[M]/k∞) × F^(1/(1+(log10(k0[M]/k∞)/N)²))

        Parameters
        ----------
        k0 : float
            低压极限速率常数 (cm⁶ molec⁻² s⁻¹)
        kinf : float
            高压极限速率常数 (cm³ molec⁻¹ s⁻¹)
        M : float
            第三体浓度 (molec/cm³)
        T : float
            温度 (K)
        F : float
            Troe 展宽因子

        Returns
        -------
        k : float
            有效双分子速率常数 (cm³ molec⁻¹ s⁻¹)
        """
        if M <= 0 or T <= 0:
            return 0.0

        k0_T = k0 * (T / 300.0) ** (-2.0)
        kinf_T = kinf * (T / 300.0) ** (-1.0)

        k0M = k0_T * M
        ratio = k0M / (kinf_T + 1e-40)

        N = 0.75 - 1.27 * np.log10(F)
        c = -0.4 - 0.67 * np.log10(ratio + 1e-40)

        log_ratio = np.log10(ratio + 1e-40)
        d = 1.0 + (log_ratio / N) ** 2

        k = k0M / (1.0 + ratio) * F ** (1.0 / d)
        return max(k, 1e-40)

    def compute_reaction_rates(self) -> Dict[str, np.ndarray]:
        """
        计算所有化学反应的速率 (molec cm⁻³ s⁻¹)

        Returns
        -------
        rates : dict
            各反应速率数组
        """
        rates = {}
        nz = self.nz

        for key, rxn in self.reactions.items():
            rate = np.zeros(nz)
            T_local = self.T

            if rxn['type'] == 'bimolecular':
                A = rxn['A']
                Ea = rxn['Ea']
                r1 = rxn['reactants'][0]
                r2 = rxn['reactants'][1]
                for iz in range(nz):
                    k = self.arrhenius_rate(A, Ea, T_local[iz])
                    conc1 = max(self.species[r1][iz], 0.0)
                    conc2 = max(self.species[r2][iz], 0.0)
                    rate[iz] = k * conc1 * conc2

            elif rxn['type'] == 'termolecular':
                A = rxn['A']
                Ea = rxn['Ea']
                r1 = rxn['reactants'][0]
                r2 = rxn['reactants'][1]
                # 简化为双分子+第三体
                for iz in range(nz):
                    k = self.arrhenius_rate(A, Ea, T_local[iz])
                    conc1 = max(self.species[r1][iz], 0.0)
                    conc2 = max(self.species[r2][iz], 0.0)
                    M = self.rho_air[iz] * 1e-6  # mol/m³ -> molec/cm³ (简化)
                    rate[iz] = k * conc1 * conc2 * M

            rates[key] = np.clip(rate, 0.0, 1e30)

        return rates

    def compute_production_loss(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        计算各物种的化学生产率和损失率

        P_i = Σ_j ν_ij^+ R_j
        L_i = n_i * Σ_j l_ij R_j / [reactant]

        Returns
        -------
        production, loss : dict of ndarray
            各物种的生产率和损失率 (molec cm⁻³ s⁻¹)
        """
        rates = self.compute_reaction_rates()
        nz = self.nz

        production = {s: np.zeros(nz) for s in self.species}
        loss = {s: np.zeros(nz) for s in self.species}

        # 光解反应贡献
        for iz in range(nz):
            # O2 + hν -> 2O
            production['O'][iz] += 2.0 * self.J_rates['J_O2'][iz] * max(self.species['O2'][iz], 0.0)
            loss['O2'][iz] += self.J_rates['J_O2'][iz] * max(self.species['O2'][iz], 0.0)

            # O3 + hν -> O2 + O
            production['O'][iz] += self.J_rates['J_O3'][iz] * max(self.species['O3'][iz], 0.0)
            production['O2'][iz] += self.J_rates['J_O3'][iz] * max(self.species['O3'][iz], 0.0)
            loss['O3'][iz] += self.J_rates['J_O3'][iz] * max(self.species['O3'][iz], 0.0)

            # NO2 + hν -> NO + O
            production['NO'][iz] += self.J_rates['J_NO2'][iz] * max(self.species['NO2'][iz], 0.0)
            production['O'][iz] += self.J_rates['J_NO2'][iz] * max(self.species['NO2'][iz], 0.0)
            loss['NO2'][iz] += self.J_rates['J_NO2'][iz] * max(self.species['NO2'][iz], 0.0)

        # 双分子/三体反应贡献
        for key, rate_arr in rates.items():
            rxn = self.reactions[key]
            reacs = rxn['reactants']
            prods = rxn['products']

            for iz in range(nz):
                r = max(rate_arr[iz], 0.0)
                # 损失: 反应物
                for reac in reacs:
                    if reac in loss and reac != 'M':
                        loss[reac][iz] += r
                # 生产: 产物
                for prod in prods:
                    if prod in production:
                        production[prod][iz] += r

        # 边界处理
        for s in production:
            production[s] = np.clip(production[s], 0.0, 1e30)
            loss[s] = np.clip(loss[s], 0.0, 1e30)

        return production, loss

    def ozone_tendency(self) -> np.ndarray:
        """
        计算臭氧的净化学趋势 dn(O3)/dt = P(O3) - L(O3)

        Returns
        -------
        tendency : ndarray
            臭氧趋势 (molec cm⁻³ s⁻¹)
        """
        production, loss = self.compute_production_loss()
        tendency = production['O3'] - loss['O3']
        return np.clip(tendency, -1e20, 1e20)

    def ozone_column_density(self) -> float:
        """
        计算臭氧柱总量 (Dobson Unit)
        1 DU = 2.69e16 molec/cm²
        TOC = ∫ n_O3(z) dz
        """
        # 数值积分 (梯形法则)
        n_o3 = self.species['O3']  # molec/cm³
        # dz 单位转换: m -> cm
        dz_cm = self.dz * 100.0
        toc = np.trapezoid(n_o3, dx=dz_cm)
        # 转换为 DU
        du = toc / 2.69e16
        return du

    def update_species(self, dt: float, production: Dict[str, np.ndarray],
                       loss: Dict[str, np.ndarray]) -> None:
        """
        半隐式更新物种浓度 (保证正定性)
        n(t+dt) = (n(t) + dt * P) / (1 + dt * k_loss)
        其中 k_loss = L / n
        对短寿命物种使用准稳态近似
        """
        if dt <= 0:
            raise ValueError("时间步长必须为正")

        for s in self.species:
            if s in ['O', 'OH', 'Cl', 'Br']:
                # 短寿命物种: 准稳态近似 n = P / k_loss
                k_loss = np.clip(loss[s] / (self.species[s] + 1e-30), 1e-30, 1e30)
                self.species[s] = np.clip(production[s] / k_loss, 1e-3, 1e15)
            else:
                # 半隐式更新 ( unconditionally positive )
                k_loss = np.clip(loss[s] / (self.species[s] + 1e-30), 0.0, 1e30)
                n_new = (self.species[s] + dt * production[s]) / (1.0 + dt * k_loss)
                self.species[s] = np.clip(n_new, 1e-3, 1e20)

    def get_state_vector(self) -> np.ndarray:
        """
        将物种浓度展开为状态向量 (用于数值求解)
        """
        state = []
        for s in sorted(self.species.keys()):
            state.extend(self.species[s])
        return np.array(state)

    def set_state_vector(self, state: np.ndarray) -> None:
        """
        从状态向量恢复物种浓度
        """
        n_species = len(self.species)
        expected_len = n_species * self.nz
        if len(state) != expected_len:
            raise ValueError(f"状态向量长度不匹配: {len(state)} != {expected_len}")

        idx = 0
        for s in sorted(self.species.keys()):
            self.species[s] = np.clip(state[idx:idx + self.nz], 1e-30, 1e25)
            idx += self.nz

    def compute_jacobian_diagonal(self, species_name: Optional[str] = None) -> np.ndarray:
        """
        计算化学雅可比矩阵的对角块近似
        用于隐式时间积分
        J_ii = ∂f_i/∂n_i ≈ -L_i/n_i

        Parameters
        ----------
        species_name : str, optional
            若指定，只返回该物种的雅可比对角线 (长度 nz)
            否则返回所有物种拼接的向量 (长度 nz * n_species)
        """
        production, loss = self.compute_production_loss()
        if species_name is not None:
            if species_name not in self.species:
                raise KeyError(f"未知物种: {species_name}")
            diag = -np.clip(loss[species_name] / (self.species[species_name] + 1e-30), 0.0, 1e20)
            return diag
        jac_diag = []
        for s in sorted(self.species.keys()):
            diag = -np.clip(loss[s] / (self.species[s] + 1e-30), 0.0, 1e20)
            jac_diag.extend(diag)
        return np.array(jac_diag)
