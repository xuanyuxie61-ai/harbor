"""
signal_reconstruction.py
信号重建与稳定性分析模块

融合 hermite_interpolant (带导数信息的 Hermite 插值)
与 boundary_locus (ODE 数值方法稳定性区域分析)。

核心科学模型：
  脉冲序列重建连续信号：
    给定脉冲时刻 {t_i} 与对应导数信息 (脉冲发放率的变化率)，
    构造 Hermite 插值多项式 H(t) 使得:
      H(t_i) = s_i          (信号值)
      H'(t_i) = s'_i        (导数值)

    差分表构造 (Newton-Hermite 形式):
      x_{2i}   = x_{2i+1} = t_i
      y_{2i}   = s_i
      y_{2i+1} = s'_i
      第一层差分:
        d_1 = (y_{j} - y_{j-1}) / (x_j - x_{j-1})  (j 为奇数时 x_j = x_{j-1}, 取 s'_i)
      递归:
        d_k^{(j)} = (d_{k-1}^{(j)} - d_{k-1}^{(j-1)}) / (x_j - x_{j-k+1})

    插值多项式:
      H(t) = sum_{k=0}^{2n-1} c_k * prod_{j=0}^{k-1} (t - x_j)

  重建误差分析：
    |f(t) - H(t)| <= M_{2n} / (2n)! * |omega_{2n}(t)|
    其中 omega_{2n}(t) = prod_{i=1}^n (t - t_i)^2
    M_{2n} = max |f^{(2n)}(xi)|

  数值稳定性分析 (boundary_locus 思想)：
    将脉冲发放动力学视为线性化 ODE: y' = lambda y
    应用 RK4 方法后的放大因子:
      R(z) = 1 + z + z^2/2 + z^3/6 + z^4/24,  z = h*lambda
    稳定性区域: { z in C : |R(z)| <= 1 }

    对于脉冲神经元的阈值动力学，lambda 与膜电位斜率相关:
      lambda(V) = (1/C_m) * [ -g_Na m^3 h - g_K n^4 - g_L + I_ext'(V) ]

    为保证数值稳定性，要求时间步长 dt 满足:
      z = dt * max|lambda(V)| 位于稳定性区域内。
"""

import numpy as np


class HermiteInterpolator:
    """
    Hermite 插值器，融合 hermite_interpolant 的差分表构造。
    """

    def __init__(self, t_nodes, values, derivatives):
        """
        t_nodes: 节点时刻 (长度 n)
        values: 节点处函数值
        derivatives: 节点处导数值
        """
        self.t = np.asarray(t_nodes, dtype=float)
        self.y = np.asarray(values, dtype=float)
        self.yp = np.asarray(derivatives, dtype=float)
        if not (len(self.t) == len(self.y) == len(self.yp)):
            raise ValueError("t_nodes, values, derivatives must have same length.")
        if len(self.t) < 1:
            raise ValueError("At least one node required.")
        if not np.all(np.diff(self.t) > 0):
            raise ValueError("t_nodes must be strictly increasing.")
        self.n = len(self.t)
        self._build_divided_difference_table()

    def _build_divided_difference_table(self):
        """
        构建 Hermite 差分表。
        输出:
          self.xd: 扩展节点 (每个 t_i 重复两次)
          self.yd: 差分表第一列
        """
        n = self.n
        nd = 2 * n
        self.xd = np.zeros(nd)
        self.xd[0::2] = self.t
        self.xd[1::2] = self.t

        self.yd = np.zeros(nd)
        self.yd[0] = self.y[0]
        # 第一层差分
        for i in range(1, n):
            self.yd[2 * i - 1] = (self.y[i] - self.y[i - 1]) / (self.t[i] - self.t[i - 1])
            self.yd[2 * i] = self.yp[i]
        self.yd[1] = self.yp[0]

        # 高阶差分
        for i in range(3, nd + 1):
            for j in range(nd - 1, i - 2, -1):
                denom = self.xd[j] - self.xd[j - i + 1]
                if np.abs(denom) < 1e-14:
                    self.yd[j] = 0.0
                else:
                    self.yd[j] = (self.yd[j] - self.yd[j - 1]) / denom

    def evaluate(self, t_query):
        """
        在 t_query 处求插值多项式值。
        """
        t_query = np.atleast_1d(t_query)
        result = np.zeros_like(t_query, dtype=float)
        for idx, t in enumerate(t_query):
            # 外推警告
            if t < self.xd[0] - 1e-10 or t > self.xd[-1] + 1e-10:
                # 允许微小外推
                pass
            # Horner 法则求 Newton 形式
            nd = len(self.xd)
            val = self.yd[nd - 1]
            for k in range(nd - 2, -1, -1):
                val = self.yd[k] + (t - self.xd[k]) * val
            result[idx] = val
        return result

    def evaluate_derivative(self, t_query):
        """
        在 t_query 处求插值多项式导数。
        """
        # 利用差分表对导数插值 (简化：数值微分)
        eps = 1e-6
        return (self.evaluate(t_query + eps) - self.evaluate(t_query - eps)) / (2.0 * eps)

    def error_bound(self, t_query, M_2n):
        """
        计算插值误差上界。
        M_2n: |f^{(2n)}| 的上界
        """
        t_query = np.atleast_1d(t_query)
        omega = np.ones_like(t_query)
        for ti in self.t:
            omega *= np.abs(t_query - ti) ** 2
        fact = 1.0
        for k in range(1, 2 * self.n + 1):
            fact *= k
        return M_2n / fact * omega


class StabilityAnalyzer:
    """
    数值稳定性分析器，融合 boundary_locus 的稳定性区域计算。
    """

    @staticmethod
    def rk4_amplification_factor(z):
        """
        RK4 方法的放大因子 R(z)。
        z: 复数，z = dt * lambda
        """
        z = np.asarray(z, dtype=complex)
        return 1.0 + z + z ** 2 / 2.0 + z ** 3 / 6.0 + z ** 4 / 24.0

    @staticmethod
    def stability_region_grid(xa, xb, ya, yb, nptsx=201, nptsy=201):
        """
        在复平面上计算 RK4 稳定性区域。
        返回:
          X, Y, Rabs: |R(z)| 的网格数据
        """
        x = np.linspace(xa, xb, nptsx)
        y = np.linspace(ya, yb, nptsy)
        X, Y = np.meshgrid(x, y)
        Z = X + 1j * Y
        Rval = StabilityAnalyzer.rk4_amplification_factor(Z)
        Rabs = np.abs(Rval)
        return X, Y, Rabs

    @staticmethod
    def is_stable(z):
        """判断 z 是否在 RK4 稳定性区域内。"""
        Rval = StabilityAnalyzer.rk4_amplification_factor(z)
        return np.abs(Rval) <= 1.0 + 1e-10

    @staticmethod
    def max_stable_timestep(lambda_max, method='rk4'):
        """
        TODO (Hole 2a): 给定特征值上界 lambda_max，返回最大稳定时间步长。
        RK4 的稳定性区间在实轴上约为 [-2.78, 0]。
        注意: 本方法中的 RK4 实轴稳定性边界需与 spike_neuron.py 中使用的 RK4 方法一致。
        """
        raise NotImplementedError("Hole 2a: 请补全 RK4 最大稳定步长计算")

    @staticmethod
    def neuron_linearized_eigenvalue(V, m, h, n, g_Na=120.0, g_K=36.0, g_L=0.3):
        """
        TODO (Hole 2b): 计算 HH 方程在平衡点附近的线性化特征值 (简化实部)。
        lambda ≈ -(g_Na m^3 h + g_K n^4 + g_L) / C_m
        注意: 本公式中的电导参数和门控变量幂次需与 spike_neuron.py 中 _dVdt 的实现严格一致。
        """
        raise NotImplementedError("Hole 2b: 请补全 HH 线性化特征值公式")


class SignalReconstructor:
    """
    从脉冲序列重建连续信号的综合模块。
    """

    def __init__(self, spike_times, spike_values=None):
        self.spike_times = np.asarray(spike_times, dtype=float)
        self.n_spikes = len(self.spike_times)
        if self.n_spikes < 2:
            raise ValueError("At least two spikes required for reconstruction.")
        if spike_values is None:
            # 使用脉冲间隔的倒数作为伪信号值 (发放率)
            intervals = np.diff(self.spike_times)
            values = 1.0 / (intervals + 1e-6)
            # 将间隔值映射到对应节点：每个间隔值赋给左端点
            self.spike_values = np.zeros(self.n_spikes)
            self.spike_values[:-1] = values
            self.spike_values[-1] = values[-1]
        else:
            self.spike_values = np.asarray(spike_values, dtype=float)

    def reconstruct_hermite(self, t_query):
        """
        使用 Hermite 插值重建信号。
        导数信息通过相邻脉冲值的有限差分近似。
        """
        n = len(self.spike_times)
        if n < 2:
            return np.zeros_like(np.atleast_1d(t_query), dtype=float)

        # 若脉冲数 >= 4，去掉首尾；否则使用全部节点
        if n >= 4:
            nodes_t = self.spike_times[1:-1]
            nodes_v = self.spike_values[1:-1]
            offset = 1
        else:
            nodes_t = self.spike_times
            nodes_v = self.spike_values
            offset = 0

        # 导数近似: dv/dt
        dt_nodes = np.diff(self.spike_times)
        dv = np.diff(self.spike_values)
        derivatives = np.zeros(len(nodes_t))
        for i in range(len(nodes_t)):
            idx = i + offset
            # 取左右平均导数
            left_dv = dv[idx - 1] / dt_nodes[idx - 1] if (idx > 0 and dt_nodes[idx - 1] > 0) else 0.0
            right_dv = dv[idx] / dt_nodes[idx] if (idx < len(dv) and dt_nodes[idx] > 0) else 0.0
            if idx == 0:
                derivatives[i] = right_dv
            elif idx >= len(dv):
                derivatives[i] = left_dv
            else:
                derivatives[i] = 0.5 * (left_dv + right_dv)

        interp = HermiteInterpolator(nodes_t, nodes_v, derivatives)
        return interp.evaluate(t_query)

    def reconstruction_quality(self, t_grid, signal_true):
        """
        计算重建质量指标：SNR, MSE, 相关系数。
        """
        signal_recon = self.reconstruct_hermite(t_grid)
        mse = np.mean((signal_true - signal_recon) ** 2)
        var_true = np.var(signal_true)
        snr = 10.0 * np.log10(var_true / (mse + 1e-12))
        corr = np.corrcoef(signal_true, signal_recon)[0, 1]
        return {'MSE': mse, 'SNR_dB': snr, 'Correlation': corr}


def demo_hermite_reconstruction():
    """Hermite 插值重建 demo。"""
    # 模拟脉冲序列
    t_nodes = np.array([0.0, 2.0, 5.0, 8.0, 12.0, 16.0, 20.0])
    v_nodes = np.array([1.0, 2.5, 1.8, 3.0, 2.2, 1.5, 2.0])
    dv_nodes = np.array([0.5, -0.3, 0.4, -0.2, -0.1, 0.3, 0.1])
    interp = HermiteInterpolator(t_nodes, v_nodes, dv_nodes)
    t_fine = np.linspace(0, 20, 200)
    v_recon = interp.evaluate(t_fine)
    return t_fine, v_recon


def demo_stability_analysis():
    """稳定性分析 demo。"""
    stab = StabilityAnalyzer()
    X, Y, Rabs = stab.stability_region_grid(-4.0, 2.0, -3.0, 3.0, nptsx=101, nptsy=101)
    # 查找稳定性边界 (|R|=1 的等高线近似)
    boundary_points = []
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if np.abs(Rabs[i, j] - 1.0) < 0.05:
                boundary_points.append((X[i, j], Y[i, j]))
    return X, Y, Rabs, boundary_points
