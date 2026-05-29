"""
stability_analysis.py
爆轰波线性稳定性分析模块
融合来源：315_double_well_ode（双阱势能思想 → 反应区扰动势能分析）
           861_pendulum_nonlinear_ode（非线性振动 → 爆轰波振荡模态）

基于 ZND 剖面的小扰动理论，计算增长率和振荡频率。
"""
import numpy as np
from combustion_utils import check_positive, check_nonnegative


class DetonationStability:
    r"""
    一维爆轰波线性稳定性分析器。

    理论基础：对 ZND 解施加小扰动:
        U(x,t) = U_0(x) + epsilon * U'(x) * exp(i*k*x + sigma*t)
    其中 sigma = alpha + i*omega 为复增长率。

    简化为在反应区上求解特征值问题:
        det(M - sigma*I) = 0
    其中 M 为近似 Jacobian 矩阵。
    """

    def __init__(self, xi, znd_sol, gamma=1.4, Q=2.5e6):
        r"""
        输入:
            xi:      ZND 空间坐标数组
            znd_sol: [rho, u, p, lambda] 解矩阵
        """
        self.xi = np.asarray(xi, dtype=float)
        self.znd = np.asarray(znd_sol, dtype=float)
        self.gamma = gamma
        self.Q = Q
        self.npts = len(xi)

    def _compute_local_jacobian(self, idx):
        r"""
        在 ZND 剖面第 idx 点处计算局部线性化 Jacobian。

        状态向量: W = [rho, u, p, lambda]^T
        线性化方程（在波坐标系中）:
            dW/dt = J * W
        其中 J 由 Euler 方程和反应方程线性化得到。
        """
        rho, u, p, lam = self.znd[idx]
        if rho <= 0.0 or p <= 0.0:
            return np.zeros((4, 4))

        a = np.sqrt(self.gamma * p / rho)
        v_rel = u  # 简化：假设在波坐标系中 D 已减去

        J = np.zeros((4, 4))
        # 质量方程线性化: drho/dt = -u*drho/dx - rho*du/dx
        J[0, 0] = -v_rel / (self.xi[1] - self.xi[0]) if self.npts > 1 else 0.0
        J[0, 1] = -rho / (self.xi[1] - self.xi[0]) if self.npts > 1 else 0.0

        # 动量方程线性化
        J[1, 0] = 0.0
        J[1, 1] = -v_rel / (self.xi[1] - self.xi[0]) if self.npts > 1 else 0.0
        J[1, 2] = -1.0 / (rho * (self.xi[1] - self.xi[0])) if self.npts > 1 else 0.0

        # 能量方程线性化
        cv = 1.0 / (self.gamma - 1.0) * (8.314 / 0.029)
        T = p / (rho * (8.314 / 0.029))
        k = 1.0e8 * np.exp(-8.314e4 / (8.314 * max(T, 100.0)))
        dlam_dt = -k * max(1.0 - lam, 0.0)

        J[2, 0] = (self.gamma - 1.0) * T * v_rel
        J[2, 1] = -v_rel * v_rel
        J[2, 2] = -v_rel / (self.xi[1] - self.xi[0]) if self.npts > 1 else 0.0
        J[2, 3] = self.Q * rho * k  # 化学能释放对压力的贡献

        # 反应进度方程
        J[3, 3] = -k if lam < 1.0 else 0.0

        return J

    def global_stability_matrix(self):
        r"""
        构建全局稳定性矩阵（分块近似）。
        对每个空间点计算局部 Jacobian 后按块组合。
        为简化，取反应区上 Jacobian 的积分平均。
        """
        J_avg = np.zeros((4, 4))
        dx_total = self.xi[-1] - self.xi[0]
        if dx_total <= 0.0:
            dx_total = 1.0

        for i in range(self.npts - 1):
            dx = self.xi[i + 1] - self.xi[i]
            Ji = self._compute_local_jacobian(i)
            J_avg += Ji * dx
        J_avg /= dx_total
        return J_avg

    def eigenvalue_analysis(self):
        r"""
        对全局稳定性矩阵进行特征值分解。
        返回:
            eigenvalues: 复特征值数组
            eigenvectors: 对应特征向量
        """
        J = self.global_stability_matrix()
        eigenvalues, eigenvectors = np.linalg.eig(J)
        # 按实部排序（从最大增长率到最稳定）
        idx = np.argsort(-eigenvalues.real)
        return eigenvalues[idx], eigenvectors[:, idx]

    def instability_modes(self):
        r"""
        识别不稳定模态（实部为正的特征值）。
        返回不稳定模态列表，每个包含:
            growth_rate: 增长率 alpha
            frequency:   振荡频率 omega/(2*pi)
        """
        evals, evecs = self.eigenvalue_analysis()
        modes = []
        for ev in evals:
            alpha = ev.real
            omega = ev.imag
            if alpha > 1.0e-6:
                modes.append({
                    'growth_rate': alpha,
                    'frequency': abs(omega) / (2.0 * np.pi),
                    'eigenvalue': ev
                })
        return modes

    def pulsation_frequency_estimate(self):
        r"""
        估算爆轰波头振荡频率（pulsation frequency）。

        对稳定爆轰（无正实部特征值），取特征值虚部最大者
        作为振荡主模态频率。
        """
        evals, _ = self.eigenvalue_analysis()
        # 找虚部绝对值最大的特征值
        idx = np.argmax(np.abs(evals.imag))
        return abs(evals[idx].imag) / (2.0 * np.pi)
