"""
反应速率计算与温度插值模块

本模块处理平流层化学反应速率的温度依赖插值，包括：
- Arrhenius 方程及其修正形式
- 三体反应的 Troe 近似
- 基于 Vandermonde 矩阵的温度剖面多项式拟合
- 反应速率常数的三维查找表 (高度 × 温度 × 压强)

科学公式:
1. 修正 Arrhenius 方程:
   k(T) = A × (T/300)^n × exp(-Ea/(R T))

2. 三体反应 Troe 形式:
   k([M], T) = k0(T)[M] / (1 + k0(T)[M]/k∞(T)) × F^G
   其中 G = 1 / (1 + [log10(k0[M]/k∞) / N]^2)
   N = 0.75 - 1.27 log10(F)

3. 光解速率温度修正:
   J(T) = J0 × (1 + α(T - T0))

4. Vandermonde 多项式拟合:
   T(z) ≈ Σ_{j=0}^{m} c_j × z^j
   系数 c 通过最小二乘求解 V c = T_data

融入原项目: 1382_vandermonde_approx_1d (一维多项式拟合)
"""

import numpy as np
from typing import Tuple, Optional

R_GAS = 8.314  # J/(mol·K)


def vandermonde_matrix(n: int, m: int, x: np.ndarray) -> np.ndarray:
    """
    构建 Vandermonde 近似矩阵
    V[i,j] = x[i]^j

    Parameters
    ----------
    n : int
        数据点数量
    m : int
        多项式阶数
    x : ndarray
        数据点位置 (n,)

    Returns
    -------
    V : ndarray
        Vandermonde 矩阵 (n, m+1)
    """
    if n <= 0 or m < 0:
        raise ValueError("n > 0 且 m >= 0")
    if len(x) != n:
        raise ValueError(f"x 长度 {len(x)} 不等于 n={n}")

    V = np.zeros((n, m + 1))
    for j in range(m + 1):
        V[:, j] = x ** j
    return V


def vandermonde_approx_coef(n: int, m: int, x: np.ndarray,
                            y: np.ndarray) -> np.ndarray:
    """
    使用 Vandermonde 矩阵进行最小二乘多项式拟合
    求解 min ||V c - y||₂

    Parameters
    ----------
    n : int
        数据点数
    m : int
        多项式阶数
    x, y : ndarray
        数据 (n,)

    Returns
    -------
    c : ndarray
        多项式系数 (m+1,)
    """
    V = vandermonde_matrix(n, m, x)
    # 使用正规方程或伪逆求解
    c, residuals, rank, s = np.linalg.lstsq(V, y, rcond=1e-12)
    return c


def polyval_horner(c: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    使用 Horner 法则计算多项式值
    p(x) = c0 + c1*x + c2*x^2 + ... + cm*x^m
         = c0 + x*(c1 + x*(c2 + ...))

    Parameters
    ----------
    c : ndarray
        系数 [c0, c1, ..., cm]
    x : ndarray
        求值点

    Returns
    -------
    y : ndarray
        多项式值
    """
    if len(c) == 0:
        return np.zeros_like(x)

    y = np.full_like(x, c[-1])
    for i in range(len(c) - 2, -1, -1):
        y = y * x + c[i]
    return y


class ReactionRateInterpolator:
    """
    反应速率常数的温度-压强插值器
    基于 Vandermonde 多项式拟合温度剖面，并查表获取速率常数
    """

    def __init__(self, temp_range: Tuple[float, float] = (180.0, 270.0),
                 pres_range: Tuple[float, float] = (1.0, 1000.0),
                 n_t: int = 50, n_p: int = 30):
        """
        Parameters
        ----------
        temp_range : tuple
            温度范围 (K)
        pres_range : tuple
            压强范围 (hPa)
        n_t, n_p : int
            温度/压强网格点数
        """
        if temp_range[0] >= temp_range[1] or pres_range[0] >= pres_range[1]:
            raise ValueError("范围下界必须小于上界")
        if n_t < 5 or n_p < 5:
            raise ValueError("网格点数必须 >= 5")

        self.T_grid = np.linspace(temp_range[0], temp_range[1], n_t)
        self.P_grid = np.linspace(pres_range[0], pres_range[1], n_p)
        self.n_t = n_t
        self.n_p = n_p

        # 预计算反应速率查找表
        self.rate_tables = {}
        self._build_rate_tables()

        # 温度插值多项式系数 (用于高度剖面)
        self.temp_poly_coef = None

    def _build_rate_tables(self) -> None:
        """
        构建反应速率常数查找表 k(T, P)
        """
        reactions = {
            'k_O_O2_M': {'A': 6.0e-34, 'Ea': 0.0, 'n': -2.4, 'type': 'termolecular'},
            'k_O_O3': {'A': 8.0e-12, 'Ea': 2060.0 * R_GAS, 'n': 0.0, 'type': 'bimolecular'},
            'k_NO_O3': {'A': 1.8e-12, 'Ea': 1370.0 * R_GAS, 'n': 0.0, 'type': 'bimolecular'},
            'k_NO2_O': {'A': 9.3e-12, 'Ea': 0.0, 'n': 0.0, 'type': 'bimolecular'},
            'k_Cl_O3': {'A': 2.9e-11, 'Ea': 260.0 * R_GAS, 'n': 0.0, 'type': 'bimolecular'},
            'k_ClO_O': {'A': 2.8e-11, 'Ea': 0.0, 'n': 0.0, 'type': 'bimolecular'},
            'k_OH_O3': {'A': 1.7e-12, 'Ea': 940.0 * R_GAS, 'n': 0.0, 'type': 'bimolecular'},
            'k_HO2_O': {'A': 3.0e-11, 'Ea': 200.0 * R_GAS, 'n': 0.0, 'type': 'bimolecular'},
        }

        for name, params in reactions.items():
            table = np.zeros((self.n_t, self.n_p))
            for i, T in enumerate(self.T_grid):
                for j, P in enumerate(self.P_grid):
                    k = self._compute_k(T, P, params)
                    table[i, j] = k
            self.rate_tables[name] = table

    def _compute_k(self, T: float, P: float, params: dict) -> float:
        """
        计算单点反应速率常数
        """
        if T <= 0:
            return 0.0

        A = params['A']
        Ea = params['Ea']
        n = params.get('n', 0.0)

        k = A * (T / 300.0) ** n * np.exp(-Ea / (R_GAS * T + 1e-30))

        if params['type'] == 'termolecular':
            # 简化的三体修正: [M] ∝ P/T
            M = P * 100.0 / (R_GAS * T) * 6.022e23 * 1e-6  # molec/cm³
            k = k * max(M, 0.0)

        return max(k, 1e-40)

    def fit_temperature_profile(self, z: np.ndarray, T: np.ndarray,
                                degree: int = 6) -> np.ndarray:
        """
        使用 Vandermonde 多项式拟合温度垂直剖面
        T(z) ≈ Σ c_j z^j

        Parameters
        ----------
        z : ndarray
            高度 (m)
        T : ndarray
            温度 (K)
        degree : int
            多项式阶数

        Returns
        -------
        c : ndarray
            多项式系数
        """
        n = len(z)
        if len(T) != n:
            raise ValueError("z 和 T 长度必须相同")
        if degree < 1 or degree >= n:
            degree = min(degree, n - 1)

        # 归一化 z 以提高数值稳定性
        z_norm = (z - np.mean(z)) / (np.std(z) + 1e-30)

        c = vandermonde_approx_coef(n, degree, z_norm, T)
        self.temp_poly_coef = c
        self.temp_z_mean = np.mean(z)
        self.temp_z_std = np.std(z) + 1e-30
        return c

    def evaluate_temperature(self, z: np.ndarray) -> np.ndarray:
        """
        使用拟合多项式评估温度
        """
        if self.temp_poly_coef is None:
            raise RuntimeError("先调用 fit_temperature_profile")

        z_norm = (z - self.temp_z_mean) / self.temp_z_std
        return polyval_horner(self.temp_poly_coef, z_norm)

    def lookup_rate(self, rate_name: str, T: float, P: float) -> float:
        """
        双线性插值查找反应速率常数
        """
        if rate_name not in self.rate_tables:
            raise KeyError(f"未知反应: {rate_name}")

        T = np.clip(T, self.T_grid[0], self.T_grid[-1])
        P = np.clip(P, self.P_grid[0], self.P_grid[-1])

        # 找到包围网格点
        i = np.searchsorted(self.T_grid, T) - 1
        j = np.searchsorted(self.P_grid, P) - 1
        i = np.clip(i, 0, self.n_t - 2)
        j = np.clip(j, 0, self.n_p - 2)

        # 双线性插值
        t1 = self.T_grid[i]
        t2 = self.T_grid[i + 1]
        p1 = self.P_grid[j]
        p2 = self.P_grid[j + 1]

        dt = t2 - t1
        dp = p2 - p1
        if dt == 0:
            dt = 1e-30
        if dp == 0:
            dp = 1e-30

        wt = (T - t1) / dt
        wp = (P - p1) / dp

        table = self.rate_tables[rate_name]
        k = (1 - wt) * (1 - wp) * table[i, j] + \
            wt * (1 - wp) * table[i + 1, j] + \
            (1 - wt) * wp * table[i, j + 1] + \
            wt * wp * table[i + 1, j + 1]

        return max(k, 1e-40)

    def lookup_rate_array(self, rate_name: str, T_arr: np.ndarray,
                          P_arr: np.ndarray) -> np.ndarray:
        """
        向量化查找反应速率常数
        """
        return np.array([self.lookup_rate(rate_name, T, P)
                         for T, P in zip(T_arr, P_arr)])


class PhotolysisRateCalculator:
    """
    光解速率计算器
    基于太阳辐射通量和吸收截面计算光解速率
    """

    def __init__(self, n_wavelength: int = 100):
        """
        Parameters
        ----------
        n_wavelength : int
            波长网格点数
        """
        self.n_wavelength = n_wavelength
        # 波长网格: 120 - 350 nm (对平流层臭氧重要)
        self.wavelength = np.linspace(120.0, 350.0, n_wavelength)  # nm
        self.dw = self.wavelength[1] - self.wavelength[0]

    def absorption_cross_section(self, species: str, wavelength: np.ndarray,
                                  T: float = 220.0) -> np.ndarray:
        """
        计算物种的吸收截面 σ(λ, T) (cm²)

        Parameters
        ----------
        species : str
            物种名称 ('O2', 'O3', 'NO2')
        wavelength : ndarray
            波长 (nm)
        T : float
            温度 (K)

        Returns
        -------
        sigma : ndarray
            吸收截面 (cm²)
        """
        lam = wavelength
        sigma = np.zeros_like(lam)

        if species == 'O2':
            # Schumann-Runge bands 简化模型
            mask = lam < 242.0
            sigma[mask] = 1e-20 * np.exp(-(lam[mask] - 150.0) / 30.0)
        elif species == 'O3':
            # Hartley + Huggins bands 简化
            mask1 = lam < 240.0
            sigma[mask1] = 1.0e-19 * np.exp(-(lam[mask1] - 200.0) ** 2 / 800.0)
            mask2 = (lam >= 240.0) & (lam < 350.0)
            sigma[mask2] = 1.0e-20 * np.exp(-(lam[mask2] - 310.0) ** 2 / 2000.0)
            # 温度依赖修正
            sigma = sigma * (1.0 + 0.001 * (T - 220.0))
        elif species == 'NO2':
            mask = (lam >= 300.0) & (lam <= 500.0)
            sigma[mask] = 1e-19 * np.exp(-(lam[mask] - 400.0) ** 2 / 5000.0)

        return np.clip(sigma, 0.0, 1e-15)

    def quantum_yield(self, species: str, wavelength: np.ndarray) -> np.ndarray:
        """
        光解量子产额 Φ(λ)
        """
        lam = wavelength
        phi = np.zeros_like(lam)

        if species == 'O2':
            phi[lam < 242.0] = 1.0
        elif species == 'O3':
            # Hartley band: O3 + hν -> O2 + O(1D) for λ < 310
            mask = lam < 310.0
            phi[mask] = 0.9
            mask2 = (lam >= 310.0) & (lam < 350.0)
            phi[mask2] = 0.1
        elif species == 'NO2':
            mask = (lam >= 300.0) & (lam <= 420.0)
            phi[mask] = 1.0

        return np.clip(phi, 0.0, 1.0)

    def solar_irradiance(self, wavelength: np.ndarray,
                         altitude: float = 25000.0) -> np.ndarray:
        """
        太阳辐射通量 F(λ, z) (photons cm⁻² s⁻¹ nm⁻¹)
        F(z) = F_top * exp(-τ(z))
        """
        # 简化太阳辐射 (TOF)
        F_top = 1e13 * np.exp(-(wavelength - 200.0) ** 2 / 5000.0)
        F_top = np.clip(F_top, 1e8, 1e16)

        # 光学厚度衰减 (简化)
        # τ ∝ sec(θ) * σ * n * dz
        tau = 0.1 * (altitude / 10000.0) ** (-1.5)
        tau = np.clip(tau, 0.0, 50.0)

        F = F_top * np.exp(-tau)
        return np.clip(F, 0.0, 1e16)

    def compute_photolysis_rate(self, species: str, altitude: float,
                                 T: float, solar_zenith_angle: float = 45.0) -> float:
        """
        计算光解速率 J (s⁻¹)
        J = ∫ σ(λ,T) × Φ(λ) × F(λ,z) dλ

        Parameters
        ----------
        species : str
            物种名称
        altitude : float
            高度 (m)
        T : float
            温度 (K)
        solar_zenith_angle : float
            太阳天顶角 (度)

        Returns
        -------
        J : float
            光解速率 (s⁻¹)
        """
        if solar_zenith_angle >= 90.0:
            return 0.0

        sec_chi = 1.0 / np.cos(np.deg2rad(solar_zenith_angle))

        lam = self.wavelength
        sigma = self.absorption_cross_section(species, lam, T)
        phi = self.quantum_yield(species, lam)
        F = self.solar_irradiance(lam, altitude)

        # 数值积分 (梯形法则)
        integrand = sigma * phi * F * sec_chi
        J = np.trapezoid(integrand, dx=self.dw)

        return max(J, 1e-30)
