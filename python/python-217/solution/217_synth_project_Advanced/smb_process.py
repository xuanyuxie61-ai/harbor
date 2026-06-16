"""
smb_process.py
--------------
模拟移动床色谱过程模型 —— 映射自种子项目 1128_cadet_RDM-Example-Simulated-Moving-Bed
核心思想：实现 SMB 色谱过程的多柱串联模型，
作为鲁棒优化的应用目标与 PDE 约束的物理背景。

科学背景：
    SMB 色谱由多个吸附柱串联组成，通过周期性切换进出口位置
    实现连续分离。四区 SMB 的物料平衡：
    Zone I  (解吸)：  高浓度产品
    Zone II (精馏)：  提纯 extract
    Zone III (吸附)：  进料区
    Zone IV (缓冲)：  提纯 raffinate

    每柱的 PDE 模型 (对流-扩散-吸附)：
        dC/dt + F dQ/dt + v dC/dx = D d^2C/dx^2
        dQ/dt = k_m (Q* - Q)
    其中 Q* = a C / (1 + b C) (Langmuir 等温线).

    设计变量：流速 (Q_I, Q_II, Q_III, Q_IV)、切换时间 t_sw.
    目标：最大化生产率，约束：产品纯度 >= 阈值.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional, Dict


# =============================================================================
# Langmuir 等温线
# =============================================================================
def langmuir_isotherm(C: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Langmuir 等温线：Q* = a C / (1 + b C).
    """
    C = np.atleast_1d(C).astype(float)
    return a * C / (1.0 + b * np.abs(C))


def langmuir_derivative(C: np.ndarray, a: float, b: float) -> np.ndarray:
    """dQ*/dC = a / (1 + b C)^2."""
    C = np.atleast_1d(C).astype(float)
    return a / (1.0 + b * np.abs(C)) ** 2


# =============================================================================
# 单柱模型 (对流-扩散-吸附)
# =============================================================================
class ChromatographyColumn:
    """
    单柱对流-扩散-吸附模型：
        dC/dt + v dC/dx = D d^2C/dx^2 - F dQ/dt
        dQ/dt = k_m (Q*(C) - Q)
    离散：中心差分 + 向后 Euler.
    """

    def __init__(
        self,
        N: int = 20,
        L: float = 1.0,
        velocity: float = 1.0,
        diffusion: float = 0.01,
        porosity: float = 0.4,
        a_iso: float = 1.0,
        b_iso: float = 0.5,
        k_mass: float = 1.0,
    ):
        self.N = N
        self.L = L
        self.dx = L / N
        self.velocity = velocity
        self.diffusion = diffusion
        self.porosity = porosity
        self.a_iso = a_iso
        self.b_iso = b_iso
        self.k_mass = k_mass

    def rhs(
        self,
        C: np.ndarray,
        Q: np.ndarray,
        C_in: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """右端项 (ODE)."""
        N = self.N
        dC = np.zeros(N)
        dQ = np.zeros(N)

        # 对流 + 扩散
        for i in range(N):
            # 中心差分
            if i == 0:
                C_left = C_in
                C_right = C[1] if N > 1 else C_in
            elif i == N - 1:
                C_left = C[i - 1]
                C_right = C[i]  # Neumann
            else:
                C_left = C[i - 1]
                C_right = C[i + 1]
            dCdx = (C_right - C_left) / (2.0 * self.dx)
            d2Cdx2 = (C_right - 2.0 * C[i] + C_left) / (self.dx ** 2)
            # 吸附源项
            Q_star = langmuir_isotherm(np.array([C[i]]), self.a_iso, self.b_iso)[0]
            dQdt = self.k_mass * (Q_star - Q[i])
            dC[i] = -self.velocity * dCdx + self.diffusion * d2Cdx2 - (1.0 - self.porosity) / max(self.porosity, 1e-10) * dQdt
            dQ[i] = dQdt

        return dC, dQ

    def step(
        self,
        C: np.ndarray,
        Q: np.ndarray,
        C_in: float,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """RK4 时间步."""
        k1C, k1Q = self.rhs(C, Q, C_in)
        k2C, k2Q = self.rhs(C + 0.5 * dt * k1C, Q + 0.5 * dt * k1Q, C_in)
        k3C, k3Q = self.rhs(C + 0.5 * dt * k2C, Q + 0.5 * dt * k2Q, C_in)
        k4C, k4Q = self.rhs(C + dt * k3C, Q + dt * k3Q, C_in)
        C_new = C + (dt / 6.0) * (k1C + 2.0 * k2C + 2.0 * k3C + k4C)
        Q_new = Q + (dt / 6.0) * (k1Q + 2.0 * k2Q + 2.0 * k3Q + k4Q)
        return C_new, Q_new


# =============================================================================
# 四区 SMB 系统
# =============================================================================
class SimulatedMovingBed:
    """
    四区 SMB 色谱系统。
    4 柱 (每区 1 柱) 串联，周期性切换。
    """

    def __init__(
        self,
        n_columns: int = 4,
        n_cells: int = 10,
        L: float = 1.0,
        column_params: Optional[Dict] = None,
    ):
        self.n_columns = n_columns
        self.n_cells = n_cells
        self.L = L
        params = column_params or {}
        self.columns = [
            ChromatographyColumn(N=n_cells, L=L, **params)
            for _ in range(n_columns)
        ]
        # 状态
        self.C = [np.zeros(n_cells) for _ in range(n_columns)]
        self.Q = [np.zeros(n_cells) for _ in range(n_columns)]
        # 区域配置
        self.zones = {
            1: [0],       # 解吸 (extract)
            2: [1],       # 精馏
            3: [2],       # 吸附 (feed)
            4: [3],       # 缓冲 (raffinate)
        }

    def set_flow_rates(
        self,
        Q_extract: float,
        Q_feed: float,
        Q_desorb: float,
        Q_raff: float,
    ):
        """设置各区流速."""
        self.Q_extract = Q_extract
        self.Q_feed = Q_feed
        self.Q_desorb = Q_desorb
        self.Q_raff = Q_raff
        # 柱内流速
        self.v = [Q_extract, Q_extract + Q_feed, Q_desorb, Q_raff]

    def switch(self):
        """切换进出口 (端口移动)."""
        # 循环移位
        self.C = [self.C[-1]] + self.C[:-1]
        self.Q = [self.Q[-1]] + self.Q[:-1]

    def step(
        self,
        C_feed: float,
        C_desorb: float,
        dt: float,
    ) -> Tuple[float, float]:
        """
        单时间步 SMB 仿真。
        返回 (extract_conc, raffinate_conc).
        """
        # Zone I (解吸): 入口 = 解吸剂
        self.C[0], self.Q[0] = self.columns[0].step(
            self.C[0], self.Q[0], C_desorb, dt
        )
        # Zone II: 入口 = Zone I 出口
        C_in_II = self.C[0][-1]
        self.C[1], self.Q[1] = self.columns[1].step(
            self.C[1], self.Q[1], C_in_II, dt
        )
        # Zone III: 入口 = Zone II 出口 + 进料
        C_in_III = self.C[1][-1] + C_feed
        self.C[2], self.Q[2] = self.columns[2].step(
            self.C[2], self.Q[2], C_in_III, dt
        )
        # Zone IV: 入口 = Zone III 出口
        C_in_IV = self.C[2][-1]
        self.C[3], self.Q[3] = self.columns[3].step(
            self.C[3], self.Q[3], C_in_IV, dt
        )
        extract = self.C[1][-1]  # Zone II 出口
        raffinate = self.C[3][-1]  # Zone IV 出口
        return extract, raffinate

    def run(
        self,
        C_feed: float,
        C_desorb: float,
        T: float,
        t_switch: float,
        dt: float = 0.01,
    ) -> Dict[str, np.ndarray]:
        """运行 SMB 仿真."""
        N_steps = int(T / dt)
        N_switches = int(T / t_switch)
        t_arr = np.linspace(0.0, T, N_steps + 1)
        extract_arr = np.zeros(N_steps + 1)
        raff_arr = np.zeros(N_steps + 1)

        switch_counter = 0
        for i in range(N_steps):
            ext, raff = self.step(C_feed, C_desorb, dt)
            extract_arr[i + 1] = ext
            raff_arr[i + 1] = raff
            # 切换检查
            if (i + 1) * dt >= (switch_counter + 1) * t_switch:
                self.switch()
                switch_counter += 1

        return {
            "t": t_arr,
            "extract": extract_arr,
            "raffinate": raff_arr,
        }


# =============================================================================
# 性能指标
# =============================================================================
def compute_purity(extract: np.ndarray, raffinate: np.ndarray) -> Tuple[float, float]:
    """计算 extract 和 raffinate 纯度 (简化)."""
    ext_purity = np.mean(extract[-100:]) / (np.mean(extract[-100:]) + 1e-10)
    raff_purity = 1.0 - np.mean(raffinate[-100:]) / (np.mean(raffinate[-100:]) + 1e-10)
    return float(np.clip(ext_purity, 0.0, 1.0)), float(np.clip(raff_purity, 0.0, 1.0))


def compute_productivity(
    extract: np.ndarray,
    T: float,
    volume: float = 1.0,
) -> float:
    """生产率 = 总提取量 / (时间 * 体积)."""
    return float(np.sum(extract) * extract.size / T / max(volume, 1e-10))


# ----------------------------------------------------------------------
# 自检
# ----------------------------------------------------------------------
if __name__ == "__main__":
    smb = SimulatedMovingBed(n_columns=4, n_cells=8, L=1.0)
    smb.set_flow_rates(Q_extract=1.0, Q_feed=0.5, Q_desorb=1.0, Q_raff=0.5)
    result = smb.run(C_feed=1.0, C_desorb=0.0, T=2.0, t_switch=0.5, dt=0.05)
    print(f"SMB run: t in [0, {result['t'][-1]:.2f}]")
    ext_p, raff_p = compute_purity(result["extract"], result["raffinate"])
    prod = compute_productivity(result["extract"], 2.0)
    print(f"Extract purity: {ext_p:.4f}")
    print(f"Raffinate purity: {raff_p:.4f}")
    print(f"Productivity: {prod:.4f}")
