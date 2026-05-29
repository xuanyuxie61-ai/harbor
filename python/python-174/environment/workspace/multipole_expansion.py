"""
multipole_expansion.py
多极展开模块

融合种子项目:
- 1132_spherical_harmonic (球谐函数展开)
- 777_monomial_value (单项式求值用于多极矩)
- 1409_wedge_integrals (楔形体积分思想, 用于区域积分)

科学背景:
多极展开是FMM的核心。将一组粒子在远场产生的势能用球谐函数展开:

公式推导:
    对于位于 x_j (|x_j| < a) 的粒子, 在远场点 x (|x| > a) 的势能:

    Phi(x) = sum_j q_j / |x - x_j|
           = sum_j q_j * sum_{l=0}^{inf} sum_{m=-l}^{l} (r_j^l / r^{l+1}) * Y_l^m(theta_j, phi_j) * Y_l^{m*}(theta, phi)
           = sum_{l=0}^{inf} sum_{m=-l}^{l} M_l^m / r^{l+1} * Y_l^{m*}(theta, phi)

    其中多极矩 M_l^m 定义为:
        M_l^m = sum_j q_j * r_j^l * Y_l^m(theta_j, phi_j)

    归一化球谐函数:
        Y_l^m(theta, phi) = N_l^m * P_l^m(cos(theta)) * exp(i*m*phi)
        N_l^m = sqrt( (2l+1)*(l-m)! / (4*pi*(l+m)!) )

    实数形式 (更便于数值计算):
        Y_l^{m,c} = N_l^m * P_l^m(cos(theta)) * cos(m*phi)
        Y_l^{m,s} = N_l^m * P_l^m(cos(theta)) * sin(m*phi)

    多极矩 likewise 分为实部和虚部:
        M_l^{m,c} = sum_j q_j * r_j^l * Y_l^{m,c}(theta_j, phi_j)
        M_l^{m,s} = sum_j q_j * r_j^l * Y_l^{m,s}(theta_j, phi_j)

    远场势能:
        Phi(x) = sum_{l=0}^{L} sum_{m=0}^{l} (1/r^{l+1}) * [ M_l^{m,c} * Y_l^{m,c}(theta,phi) + M_l^{m,s} * Y_l^{m,s}(theta,phi) ]
               (对m=0项, 虚部为零)

截断误差分析:
    |Phi_exact - Phi_L| <= (Q * a^{L+1}) / (r^{L+2} - a*r^{L+1})
    其中 Q = sum_j |q_j|, a 为源区域半径, r 为观察点距离
    当 r > 2a 时, 误差随 L 指数衰减

楔形体积分思想:
    对于连续电荷分布 rho(r, theta, phi) 在楔形体区域 W 上:
        M_l^m = integral_W rho(r,theta,phi) * r^l * Y_l^m(theta,phi) dV
    其中 dV = r^2 * sin(theta) dr dtheta dphi
"""

import numpy as np
from spherical_geometry import legendre_associated_normalized


class MultipoleExpansion:
    """
    多极展开类
    
    管理一组粒子的多极矩, 并支持远场势能/力计算
    """

    def __init__(self, center, order):
        """
        参数:
            center: ndarray (3,), 展开中心
            order: int, 截断阶数 L
        """
        self.center = np.asarray(center, dtype=float)
        self.order = int(order)
        if self.order < 0:
            raise ValueError("order必须非负")
        # 存储多极矩: 对每个 (l, m), 存储 (real, imag)
        # 使用列表: moments[l][m] = (real, imag), m = 0..l
        self.moments_real = []
        self.moments_imag = []
        for l in range(self.order + 1):
            self.moments_real.append(np.zeros(l + 1))
            self.moments_imag.append(np.zeros(l + 1))
        self.total_charge = 0.0

    def _shift_to_local(self, points):
        """将点坐标转换到以center为原点的局部坐标"""
        return points - self.center

    def add_particles(self, points, charges):
        """
        累加粒子的多极矩贡献
        
        公式:
            M_l^m += sum_i q_i * |x_i - c|^l * Y_l^m(theta_i, phi_i)
        """
        points = np.atleast_2d(points)
        charges = np.asarray(charges, dtype=float)
        if points.shape[0] != charges.shape[0]:
            raise ValueError("points和charges长度不匹配")

        local = self._shift_to_local(points)
        r = np.linalg.norm(local, axis=1)
        # 避免r=0导致0^0问题 (在计算中0^0视为1)
        theta = np.arccos(np.clip(local[:, 2] / (r + 1e-15), -1.0, 1.0))
        phi = np.arctan2(local[:, 1], local[:, 0])
        phi = np.where(phi < 0, phi + 2.0 * np.pi, phi)

        # TODO(Hole_1): 实现多极矩计算的核心科学公式
        # 公式: M_l^m = sum_i q_i * |x_i - c|^l * Y_l^m(theta_i, phi_i)
        # 需要遍历每个粒子, 计算各(l,m)阶的多极矩实部和虚部
        # 注意: r=0时仅l=0项有贡献; m=0时虚部为零
        # 提示: 使用 legendre_associated_normalized(l, m, cos(theta)) 获取归一化Legendre值
        raise NotImplementedError("Hole_1: 请实现add_particles中的多极矩计算")

    def evaluate_potential(self, target):
        """
        在目标点评估多极展开势能
        
        公式:
            Phi(x) = sum_{l=0}^{L} sum_{m=0}^{l} (1/R^{l+1}) * [
                         M_l^{m,r} * Y_l^{m,r}(Theta,Phi) +
                         M_l^{m,i} * Y_l^{m,i}(Theta,Phi)
                     ]
        
        参数:
            target: ndarray (3,) 或 (N, 3)
        
        返回:
            ndarray (N,), 势能值
        """
        target = np.atleast_2d(target)
        local = target - self.center
        R = np.linalg.norm(local, axis=1)
        # 边界: 若R太小, 多极展开不适用
        R = np.where(R < 1e-15, 1e-15, R)
        theta = np.arccos(np.clip(local[:, 2] / R, -1.0, 1.0))
        phi = np.arctan2(local[:, 1], local[:, 0])
        phi = np.where(phi < 0, phi + 2.0 * np.pi, phi)

        N = target.shape[0]
        potential = np.zeros(N)

        for l in range(self.order + 1):
            inv_R_lp1 = 1.0 / (R ** (l + 1))
            plm_0 = np.array([legendre_associated_normalized(l, 0, np.cos(t)) for t in theta])
            y_0 = plm_0[:, l]
            potential += self.moments_real[l][0] * inv_R_lp1 * y_0
            for m in range(1, l + 1):
                plm_m = np.array([legendre_associated_normalized(l, m, np.cos(t)) for t in theta])
                y_real = plm_m[:, l] * np.cos(m * phi)
                y_imag = plm_m[:, l] * np.sin(m * phi)
                potential += inv_R_lp1 * (
                    self.moments_real[l][m] * y_real +
                    self.moments_imag[l][m] * y_imag
                )
        return potential

    def evaluate_field(self, target):
        """
        在目标点评估多极展开电场 (负梯度)
        
        公式:
            E = -grad Phi
        
        使用数值差分近似梯度
        """
        target = np.atleast_2d(target)
        N = target.shape[0]
        h = 1e-6
        field = np.zeros((N, 3))
        for d in range(3):
            offset = np.zeros(3)
            offset[d] = h
            phi_plus = self.evaluate_potential(target + offset)
            phi_minus = self.evaluate_potential(target - offset)
            field[:, d] = -(phi_plus - phi_minus) / (2.0 * h)
        return field

    def get_moments_l2_norm(self):
        """
        计算多极矩的L2范数
        
        公式:
            ||M||^2 = sum_{l=0}^{L} sum_{m=0}^{l} (|M_l^{m,r}|^2 + |M_l^{m,i}|^2)
        """
        norm_sq = 0.0
        for l in range(self.order + 1):
            for m in range(l + 1):
                norm_sq += self.moments_real[l][m] ** 2 + self.moments_imag[l][m] ** 2
        return np.sqrt(norm_sq)

    def truncation_error_bound(self, target, total_charge_magnitude, source_radius):
        """
        估计截断误差上界
        
        公式:
            |error| <= Q * a^{L+1} / (R^{L+2} - a * R^{L+1})
            其中 R = |target - center|, a = source_radius, Q = total_charge_magnitude
        
        参数:
            target: ndarray (3,) 或 (N, 3)
            total_charge_magnitude: float, 总电荷绝对值之和
            source_radius: float, 源区域半径
        
        返回:
            ndarray (N,), 误差上界
        """
        target = np.atleast_2d(target)
        R = np.linalg.norm(target - self.center, axis=1)
        R = np.where(R < 1e-15, 1e-15, R)
        L = self.order
        a = source_radius
        Q = total_charge_magnitude
        denom = R ** (L + 2) - a * (R ** (L + 1))
        denom = np.where(denom < 1e-15, 1e-15, denom)
        error = Q * (a ** (L + 1)) / denom
        return error
