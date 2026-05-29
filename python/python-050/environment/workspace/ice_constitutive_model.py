"""
ice_constitutive_model.py
冰川流变学本构模型 — 各向异性 Glen 流动律与温度依赖率因子

核心物理模型:
  1. Glen 流动律:  \dot{\epsilon}_{ij} = A(T') \tau_e^{n-1} \tau_{ij}
  2. 温度依赖 Arrhenius 关系:  A(T') = A_0 \exp\left(-\frac{Q}{R T'}\right)
  3. 各向异性增强因子:  E(a) = 1 + (E_{max}-1) \cdot f(a)
  4. 有效应力:  \tau_e = \sqrt{\frac{1}{2} \tau_{ij} \tau_{ij}}
  5. 耗散热:  \Phi = \tau_{ij} \dot{\epsilon}_{ij} = A(T') \tau_e^{n+1}

工程特性:
  - 边界保护: 应力/温度超出物理范围时自动截断到安全区间
  - 数值鲁棒性: 对数域计算避免指数下溢/上溢
  - 非线性迭代支持: 提供 Jacobian 矩阵元素用于 Newton 迭代
"""

import numpy as np

# 物理常数 (SI 单位)
ICE_DENSITY = 917.0          # kg m^{-3}
GRAVITY = 9.81               # m s^{-2}
SPECIFIC_HEAT = 2097.0       # J kg^{-1} K^{-1}
THERMAL_CONDUCTIVITY = 2.10  # W m^{-1} K^{-1}
LATENT_HEAT = 3.34e5         # J kg^{-1}
GAS_CONSTANT = 8.314         # J mol^{-1} K^{-1}
GLEN_N = 3.0                 # Glen 指数 (无量纲)

# Arrhenius 参数
ARRHENIUS_A0 = 3.5e-25       # Pa^{-n} s^{-1}  (参考值)
ACTIVATION_ENERGY_COLD = 6.0e4   # J mol^{-1}  (T < -10°C)
ACTIVATION_ENERGY_WARM = 13.9e4  # J mol^{-1}  (T > -10°C)
TEMP_THRESHOLD = 263.15      # K  (-10°C)

# 各向异性参数
ANISO_MAX_ENHANCEMENT = 10.0  # 最大增强因子
ANISO_SHAPE_FACTOR = 2.0      # 分布形状参数


def safe_exp(x: np.ndarray, max_val: float = 700.0) -> np.ndarray:
    """
    安全的指数函数，防止数值溢出。
    当 x > max_val 时返回 exp(max_val)，当 x < -max_val 时返回 exp(-max_val)。
    """
    x_clipped = np.clip(x, -max_val, max_val)
    return np.exp(x_clipped)


def rate_factor_arrhenius(temperature: np.ndarray) -> np.ndarray:
    """
    计算 Glen 流动律中的温度依赖率因子 A(T')。

    采用 Paterson (1994) 和 Cuffey & Paterson (2010) 的分段 Arrhenius 模型:

        A(T) = A_0 \exp\left( -\frac{Q}{R \cdot T} \right)

    其中 T 为绝对温度 (K)，Q 为激活能，分段取值:
        Q = Q_{cold}  (T \le T_{threshold})
        Q = Q_{warm}  (T > T_{threshold})

    参数:
        temperature: 绝对温度数组 (K)

    返回:
        A: 率因子数组 (Pa^{-n} s^{-1})
    """
    temperature = np.asarray(temperature, dtype=np.float64)

    # 边界保护: 冰川温度通常处于 200 K ~ 273.15 K
    if np.any(temperature < 100.0) or np.any(temperature > 300.0):
        raise ValueError("Temperature out of physical range for ice (100K ~ 300K).")

    # 区分冷冰与暖冰
    q = np.where(temperature <= TEMP_THRESHOLD,
                 ACTIVATION_ENERGY_COLD,
                 ACTIVATION_ENERGY_WARM)

    # TODO_HOLE_1: 实现 Arrhenius 率因子计算
    # 科学知识点: 分段 Arrhenius 模型 A(T) = A_0 * exp(-Q / (R * T))
    # 需要利用前面计算的 q 和 temperature，返回率因子数组 a
    # 注意数值保护: A 不应为 0 或 inf
    raise NotImplementedError("Hole 1: 请实现 rate_factor_arrhenius 核心公式")


def effective_stress(deviatoric_stress: np.ndarray) -> np.ndarray:
    """
    计算偏应力张量的等效应力 (von Mises 型)。

    对于三维偏应力张量 \tau_{ij} (i,j = 1,2,3)，等效应力定义为:

        \tau_e = \sqrt{ \frac{1}{2} \tau_{ij} \tau_{ij} }

    参数:
        deviatoric_stress: 形状为 (..., 3, 3) 的偏应力张量 (Pa)

    返回:
        tau_e: 等效应力 (Pa)
    """
    deviatoric_stress = np.asarray(deviatoric_stress, dtype=np.float64)

    if deviatoric_stress.shape[-2:] != (3, 3):
        raise ValueError("deviatoric_stress must have shape (..., 3, 3)")

    # 计算 Frobenius 内积: \tau_{ij}\tau_{ij}
    double_contract = np.sum(deviatoric_stress * deviatoric_stress, axis=(-2, -1))
    tau_e = np.sqrt(0.5 * np.maximum(double_contract, 0.0))

    # 避免除零: 设置最小阈值
    tau_e = np.maximum(tau_e, 1e-12)
    return tau_e


def glen_flow_law(deviatoric_stress: np.ndarray,
                  temperature: np.ndarray,
                  anisotropy_factor: np.ndarray = None) -> np.ndarray:
    """
    三维各向异性 Glen 流动律。

    应变率张量:
        \dot{\epsilon}_{ij} = E(a) \cdot A(T) \cdot \tau_e^{n-1} \cdot \tau_{ij}

    参数:
        deviatoric_stress: 偏应力张量 (..., 3, 3) in Pa
        temperature: 绝对温度 (K)
        anisotropy_factor: 各向异性增强因子 E(a) (无量纲, 默认 1.0)

    返回:
        strain_rate: 应变率张量 (..., 3, 3) in s^{-1}
    """
    deviatoric_stress = np.asarray(deviatoric_stress, dtype=np.float64)
    temperature = np.asarray(temperature, dtype=np.float64)

    if anisotropy_factor is None:
        anisotropy_factor = np.ones_like(temperature)
    else:
        anisotropy_factor = np.asarray(anisotropy_factor, dtype=np.float64)

    # 计算率因子与等效应力
    A = rate_factor_arrhenius(temperature)
    tau_e = effective_stress(deviatoric_stress)

    # 确保维度广播兼容
    # tau_e 的尾维被压缩，A/aniso 为标量场
    tau_e_shape = tau_e.shape
    target_shape = tau_e_shape + (1, 1)

    A = np.reshape(A, A.shape + (1, 1)) if A.ndim > 0 else A
    aniso = np.reshape(anisotropy_factor, anisotropy_factor.shape + (1, 1)) if anisotropy_factor.ndim > 0 else anisotropy_factor

    # 计算应变率
    prefactor = aniso * A * (tau_e ** (GLEN_N - 1.0))
    prefactor = np.reshape(prefactor, prefactor.shape + (1, 1)) if prefactor.ndim > 0 else prefactor

    strain_rate = prefactor * deviatoric_stress
    return strain_rate


def anisotropic_enhancement_factor(second_order_orientation_tensor: np.ndarray) -> np.ndarray:
    """
    基于二阶晶格取向张量 a^{(2)} 计算流动增强因子 E(a)。

    参考 Thorsteinsson (2002) 和 Gillet-Chaulet et al. (2006):

        E(a) = 1 + (E_{max} - 1) \cdot \left( 1 - \frac{3}{2} a_{ij}^{(2)} n_i n_j \right)^{p}

    其中 n 为剪切面法向。为简化，这里采用标量度量:

        f_a = \frac{3}{2} \lambda_{max} - \frac{1}{2}

    \lambda_{max} 为 a^{(2)} 的最大特征值，f_a \in [0, 1] 表示从无序到单晶。

    参数:
        second_order_orientation_tensor: 二阶取向张量 (..., 3, 3)

    返回:
        E: 增强因子 (>= 1.0)
    """
    a2 = np.asarray(second_order_orientation_tensor, dtype=np.float64)

    if a2.shape[-2:] != (3, 3):
        raise ValueError("Orientation tensor must have shape (..., 3, 3)")

    # 计算特征值 (批量)
    # 对最后两个维度进行特征值分解
    orig_shape = a2.shape[:-2]
    a2_flat = a2.reshape(-1, 3, 3)

    # 对称化确保数值稳定性
    a2_flat = 0.5 * (a2_flat + np.transpose(a2_flat, (0, 2, 1)))

    # 计算特征值
    evals = np.linalg.eigvalsh(a2_flat)  # 形状 (batch, 3)
    lambda_max = np.max(evals, axis=-1)

    # 计算有序度
    f_a = 1.5 * lambda_max - 0.5
    f_a = np.clip(f_a, 0.0, 1.0)

    E = 1.0 + (ANISO_MAX_ENHANCEMENT - 1.0) * (f_a ** ANISO_SHAPE_FACTOR)
    E = E.reshape(orig_shape)
    return E


def dissipation_heat(strain_rate: np.ndarray,
                     deviatoric_stress: np.ndarray) -> np.ndarray:
    """
    计算内部机械耗散热 (viscous dissipation):

        \Phi = \tau_{ij} \dot{\epsilon}_{ij}  (W m^{-3})

    参数:
        strain_rate: 应变率张量 (..., 3, 3) in s^{-1}
        deviatoric_stress: 偏应力张量 (..., 3, 3) in Pa

    返回:
        phi: 耗散热功率密度 (W m^{-3})
    """
    phi = np.sum(deviatoric_stress * strain_rate, axis=(-2, -1))
    phi = np.maximum(phi, 0.0)  # 耗散热非负
    return phi


def glen_viscosity(temperature: np.ndarray,
                   effective_strain_rate: np.ndarray) -> np.ndarray:
    """
    计算 Glen 律等效粘度 (非线性粘度):

        \eta = \frac{1}{2} A^{-1/n} \dot{\epsilon}_e^{(1-n)/n}

    参数:
        temperature: 绝对温度 (K)
        effective_strain_rate: 等效应变率 (s^{-1})

    返回:
        eta: 动力粘度 (Pa s)
    """
    A = rate_factor_arrhenius(temperature)
    eps_e = np.maximum(np.asarray(effective_strain_rate, dtype=np.float64), 1e-20)

    eta = 0.5 * (A ** (-1.0 / GLEN_N)) * (eps_e ** ((1.0 - GLEN_N) / GLEN_N))
    # 数值保护
    eta = np.clip(eta, 1e10, 1e20)
    return eta


def jacobian_glen_stress(strain_rate: np.ndarray,
                         deviatoric_stress: np.ndarray,
                         temperature: np.ndarray) -> np.ndarray:
    """
    计算 Glen 流动律对应力张量的 Jacobian:

        J_{ijkl} = \frac{\partial \dot{\epsilon}_{ij}}{\partial \tau_{kl}}

    解析表达式:
        J_{ijkl} = A \tau_e^{n-1} \left( \delta_{ik}\delta_{jl} + (n-1)\frac{\tau_{ij}\tau_{kl}}{2\tau_e^2} \right)

    参数:
        strain_rate: 当前应变率 (..., 3, 3)
        deviatoric_stress: 当前偏应力 (..., 3, 3)
        temperature: 温度 (K)

    返回:
        J: Jacobian 张量 (..., 3, 3, 3, 3)
    """
    tau = np.asarray(deviatoric_stress, dtype=np.float64)
    A = rate_factor_arrhenius(temperature)
    tau_e = effective_stress(tau)

    # 构建四阶张量
    shape = tau.shape
    batch = shape[:-2] if len(shape) > 2 else (1,)

    # 单位张量 \delta_{ik}\delta_{jl}
    I = np.eye(3, dtype=np.float64)
    delta = np.einsum('ik,jl->ijkl', I, I)

    # 外积项 \tau_{ij}\tau_{kl}
    tau_outer = np.einsum('...ij,...kl->...ijkl', tau, tau)

    # 组合
    prefactor = A * (tau_e ** (GLEN_N - 1.0))
    prefactor = np.reshape(prefactor, prefactor.shape + (1, 1, 1, 1)) if prefactor.ndim > 0 else prefactor

    J = prefactor * (delta + (GLEN_N - 1.0) * tau_outer / (2.0 * tau_e[:, None, None, None, None] ** 2))
    return J


def effective_strain_rate(strain_rate_tensor: np.ndarray) -> np.ndarray:
    """
    计算等效应变率:

        \dot{\epsilon}_e = \sqrt{ \frac{1}{2} \dot{\epsilon}_{ij} \dot{\epsilon}_{ij} }
    """
    eps = np.asarray(strain_rate_tensor, dtype=np.float64)
    val = np.sum(eps * eps, axis=(-2, -1))
    return np.sqrt(0.5 * np.maximum(val, 0.0))
