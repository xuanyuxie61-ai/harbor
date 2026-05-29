"""
thermostat.py
热浴与温度控制模块

融合原项目:
- 312_dosage_ode: 参数管理与时间演化协议设计
- 360_fd1d_heat_explicit: 温度场演化思想用于淬火协议

功能:
1. Nose-Hoover 热浴参数管理
2. 温度淬火协议（线性降温、阶梯降温、VFT 协议）
3. 温度-时间曲线用于玻璃化转变模拟
"""

import numpy as np
from typing import Tuple, Optional, Callable


class TemperatureProtocol:
    """
    温度协议管理器。
    
    融合原项目 312_dosage_ode 的参数管理思想，
    设计用于聚合物玻璃化转变的温度-时间曲线。
    
    物理背景:
        玻璃化转变温度 T_g 的实验测定通常通过 DSC 的降温扫描实现。
        冷却速率 q = dT/dt 对测得的 T_g 有显著影响:
            T_g(q) ≈ T_g^0 - A / ln(q)
        其中 A 为材料常数。
    
    支持的协议:
        1. 线性降温: T(t) = T0 - q * t
        2. 阶梯降温: 分段恒温-降温
        3. 对数降温: T(t) = T0 / (1 + α ln(1 + β t))
    """
    
    def __init__(
        self,
        t0: float = 0.0,
        T_initial: float = 2.0,
        T_final: float = 0.1,
        t_stop: float = 1000.0,
        protocol: str = "linear",
    ):
        """
        初始化温度协议。
        
        参数:
            t0: 初始时间
            T_initial: 初始温度
            T_final: 最终温度
            t_stop: 结束时间
            protocol: "linear" | "step" | "logarithmic"
        """
        if t_stop <= t0:
            raise ValueError("t_stop 必须 > t0")
        if T_final <= 0:
            raise ValueError("T_final 必须 > 0")
        
        self.t0 = t0
        self.T_initial = T_initial
        self.T_final = T_final
        self.t_stop = t_stop
        self.protocol = protocol
        
        # 预计算协议参数
        self._setup_protocol()
    
    def _setup_protocol(self):
        """预计算各协议的内部参数。"""
        if self.protocol == "linear":
            # T(t) = T0 - q * (t - t0)
            self.cooling_rate = (self.T_initial - self.T_final) / (self.t_stop - self.t0)
        
        elif self.protocol == "step":
            # 阶梯降温: 每段恒温后骤降
            self.n_steps = max(3, int((self.t_stop - self.t0) / 100.0))
            self.step_times = np.linspace(self.t0, self.t_stop, self.n_steps + 1)
            self.step_temps = np.linspace(self.T_initial, self.T_final, self.n_steps + 1)
        
        elif self.protocol == "logarithmic":
            # T(t) = T_final + (T_initial - T_final) * exp(-α * t)
            # 选择 α 使得 t_stop 时接近 T_final
            self.alpha = -np.log(0.01) / (self.t_stop - self.t0)
        
        else:
            raise ValueError(f"不支持的协议: {self.protocol}")
    
    def temperature(self, t: float) -> float:
        """
        获取时刻 t 的目标温度。
        
        参数:
            t: 时间
        
        返回:
            目标温度
        """
        if t <= self.t0:
            return self.T_initial
        if t >= self.t_stop:
            return self.T_final
        
        if self.protocol == "linear":
            T = self.T_initial - self.cooling_rate * (t - self.t0)
            return max(T, self.T_final)
        
        elif self.protocol == "step":
            # 找到当前所在的阶梯
            idx = np.searchsorted(self.step_times, t, side='right') - 1
            idx = max(0, min(idx, len(self.step_temps) - 1))
            return self.step_temps[idx]
        
        elif self.protocol == "logarithmic":
            T = self.T_final + (self.T_initial - self.T_final) * np.exp(
                -self.alpha * (t - self.t0)
            )
            return max(T, self.T_final)
        
        return self.T_final
    
    def cooling_rate_at(self, t: float) -> float:
        """
        计算时刻 t 的瞬时冷却速率 dT/dt。
        
        参数:
            t: 时间
        
        返回:
            冷却速率（负值表示降温）
        """
        if t <= self.t0 or t >= self.t_stop:
            return 0.0
        
        if self.protocol == "linear":
            return -self.cooling_rate
        
        elif self.protocol == "step":
            return 0.0  # 阶梯内恒温
        
        elif self.protocol == "logarithmic":
            dT = -(self.T_initial - self.T_final) * self.alpha * np.exp(
                -self.alpha * (t - self.t0)
            )
            return dT
        
        return 0.0
    
    def get_schedule(self, n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取完整的温度-时间曲线。
        
        参数:
            n_points: 采样点数
        
        返回:
            (times, temperatures)
        """
        times = np.linspace(self.t0, self.t_stop, n_points)
        temps = np.array([self.temperature(t) for t in times])
        return times, temps


class AndersenThermostat:
    """
    Andersen 热浴：随机碰撞恒温。
    
    物理模型:
        以概率 ν dt 随机选择粒子，将其速度从 Maxwell-Boltzmann 分布重采样。
        ν 为碰撞频率。
    
    优点:
        - 严格保持正则系综
        - 简单高效
    
    缺点:
        - 破坏动量守恒
        - 干扰动力学关联
    """
    
    def __init__(self, collision_frequency: float = 0.1, random_seed: int = 42):
        """
        参数:
            collision_frequency: 碰撞频率 ν
            random_seed: 随机种子
        """
        if collision_frequency < 0:
            raise ValueError("collision_frequency 必须 >= 0")
        self.nu = collision_frequency
        self.rng = np.random.RandomState(random_seed)
    
    def apply(
        self,
        velocities: np.ndarray,
        masses: np.ndarray,
        temperature: float,
        dt: float,
    ) -> np.ndarray:
        """
        应用 Andersen 热浴。
        
        参数:
            velocities: (N, 3) 速度数组
            masses: (N,) 质量数组
            temperature: 目标温度
            dt: 时间步长
        
        返回:
            更新后的速度数组
        """
        if temperature <= 0:
            return velocities
        
        new_velocities = velocities.copy()
        N = velocities.shape[0]
        
        # 每个粒子的碰撞概率
        p_collision = self.nu * dt
        if p_collision > 1.0:
            p_collision = 1.0
        
        sigma = np.sqrt(temperature / masses)
        
        for i in range(N):
            if self.rng.rand() < p_collision:
                # 从 Maxwell-Boltzmann 分布重采样
                new_velocities[i] = self.rng.normal(0.0, sigma[i], 3)
        
        return new_velocities


class BerendsenThermostat:
    """
    Berendsen 弱耦合热浴。
    
    算法:
        v_new = v * λ
        λ = sqrt(1 + (dt/τ) * (T_target/T - 1))
    
    其中 τ 为耦合时间常数。
    
    优点:
        - 快速达到目标温度
        - 对动力学扰动较小
    
    缺点:
        - 不产生严格的正则系综
    """
    
    def __init__(self, tau: float = 0.5):
        """
        参数:
            tau: 耦合时间常数 τ
        """
        if tau <= 0:
            raise ValueError("tau 必须 > 0")
        self.tau = tau
    
    def apply(
        self,
        velocities: np.ndarray,
        masses: np.ndarray,
        target_temperature: float,
        dt: float,
    ) -> np.ndarray:
        """
        应用 Berendsen 热浴。
        
        参数:
            velocities: (N, 3) 速度数组
            masses: (N,) 质量数组
            target_temperature: 目标温度
            dt: 时间步长
        
        返回:
            更新后的速度数组
        """
        if target_temperature <= 0:
            return velocities
        
        # TODO: 请实现 Berendsen 热浴的温度计算与速度缩放。
        # 注意: 此处的自由度计算和温度公式必须与 polymer_chain.py 中的 instantaneous_temperature() 保持一致。
        # 注意: 缩放因子的计算需要正确使用 dt / tau 和目标温度。
        
        # HOLE 2 START
        raise NotImplementedError("Hole 2: 请实现 Berendsen 热浴的 apply 方法核心逻辑，确保与 polymer_chain.py 的温度定义一致")
        # HOLE 2 END
