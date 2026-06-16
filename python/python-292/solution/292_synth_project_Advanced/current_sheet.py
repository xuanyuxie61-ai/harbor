"""
current_sheet.py — Harris 电流片平衡态
=========================================

融合自:
  - 204_compass_search: 平衡参数无导数优化

物理背景:
  Harris 电流片是磁重联研究的标准初始平衡:
    B_z(y) = B0 * tanh(y / a)     (反平行磁场分量)
    n(y) = n0 * sech^2(y / a) + n_bg  (密度剖面)
    p(y) = p0 * sech^2(y / a) + p_bg  (压力平衡)

  力平衡条件 (MHD 平衡):
    dp/dy + B_z * dB_z/dy / mu0 = 0
    => p(y) = p0 * sech^2(y/a) + const

  电流密度:
    J_x = -(1/mu0) * dB_z/dy = -(B0/(mu0*a)) * sech^2(y/a)

  等离子体 beta:
    beta(y) = 2*mu0*p(y) / B^2(y)
"""

import numpy as np


class CurrentSheetEquilibrium:
    """Harris 电流片平衡态求解器。"""

    def __init__(self, B0=1.0, n0=1.0, T_e=0.1, T_i=0.1, a_sheet=0.5):
        self.B0 = B0        # 渐近磁场强度
        self.n0 = n0         # 峰值密度
        self.T_e = T_e       # 电子温度
        self.T_i = T_i       # 离子温度
        self.a_sheet = a_sheet  # 电流片半厚度
        self.mu0 = 1.0       # 归一化真空磁导率
        self.n_bg = 0.2 * n0  # 背景密度

        self.By = None       # 反平行磁场分量 B_z (这里用 By 表示)
        self.density = None  # 密度剖面
        self.pressure = None  # 总压力
        self.Jx = None       # 电流密度
        self.beta = None     # 等离子体 beta
        self.J_max = None
        self.beta_min = None
        self.force_balance_residual = None

    def compute_harris_profile(self, y):
        """
        计算 Harris 平衡剖面。

        B_z(y) = B0 * tanh(y / a)
        n(y) = n0 * sech^2(y/a) + n_bg
        p(y) = (B0^2 / (2*mu0)) * sech^2(y/a) + p_bg
        """
        a = self.a_sheet
        tanh_ya = np.tanh(y / a)
        sech2_ya = 1.0 - tanh_ya ** 2  # sech^2 = 1 - tanh^2

        self.By = self.B0 * tanh_ya
        self.density = self.n0 * sech2_ya + self.n_bg
        self.pressure = (self.B0 ** 2 / (2 * self.mu0)) * sech2_ya + \
                        (self.n_bg * (self.T_e + self.T_i))

        # 电流密度: Jx = -(1/mu0) * dBy/dy = -(B0/(mu0*a)) * sech^2(y/a)
        self.Jx = -(self.B0 / (self.mu0 * a)) * sech2_ya
        self.J_max = float(np.max(np.abs(self.Jx)))

    def check_force_balance(self):
        """
        验证力平衡: dp/dy + By * dBy/dy / mu0 = 0

        残差 = max |dp/dy + By*dBy/dy/mu0|
        """
        dy = 0.01  # 差分步长
        a = self.a_sheet
        y_test = np.linspace(-3 * a, 3 * a, 200)
        sech2 = 1.0 - np.tanh(y_test / a) ** 2
        p_test = (self.B0 ** 2 / (2 * self.mu0)) * sech2 + self.n_bg * (self.T_e + self.T_i)
        By_test = self.B0 * np.tanh(y_test / a)

        dp_dy = np.gradient(p_test, dy)
        dBy_dy = np.gradient(By_test, dy)
        force_residual = dp_dy + By_test * dBy_dy / self.mu0
        self.force_balance_residual = float(np.max(np.abs(force_residual)))

    def compute_beta_profile(self):
        """
        计算等离子体 beta = 2*mu0*p / B^2。
        在 y=0 处 B=0，beta 发散；在远处 beta -> 0。
        """
        B2 = self.By ** 2 + 1e-10  # 避免除零
        self.beta = 2 * self.mu0 * self.pressure / B2
        # 取远离电流片区域的 beta (|y| > a)
        mask = np.abs(np.tanh(np.linspace(-3, 3, len(self.By)) / self.a_sheet)) > 0.5
        if np.any(mask):
            self.beta_min = float(np.min(self.beta[mask]))
        else:
            self.beta_min = float(np.min(self.beta))

    def get_perturbation_field(self, y, x, k_mode=1, amplitude=1e-3):
        """
        返回撕裂模扰动磁场 (用于初始化 MHD 演化)。

        磁通函数扰动:
          psi_1(x, y) = amplitude * cos(k*x) * sech(y/a)

        扰动磁场:
          b_x = dpsi_1/dy = -amplitude * cos(k*x) * sech(y/a)*tanh(y/a)/a
          b_y = -dpsi_1/dx = amplitude * k * sin(k*x) * sech(y/a)
        """
        a = self.a_sheet
        k = 2 * np.pi * k_mode / 12.8  # 波长适配计算域
        sech_ya = 1.0 / np.cosh(y[:, None] / a) if y.ndim == 1 else 1.0 / np.cosh(y / a)
        cos_kx = np.cos(k * x[None, :]) if x.ndim == 1 else np.cos(k * x)
        sin_kx = np.sin(k * x[None, :]) if x.ndim == 1 else np.sin(k * x)

        if y.ndim == 1 and x.ndim == 1:
            sech_ya = sech_ya[:, None]
            tanh_ya = np.tanh(y / a)[:, None]
        else:
            tanh_ya = np.tanh(y / a)

        b_x = -amplitude * cos_kx * sech_ya * tanh_ya / a
        b_y = amplitude * k * sin_kx * sech_ya
        return b_x, b_y
