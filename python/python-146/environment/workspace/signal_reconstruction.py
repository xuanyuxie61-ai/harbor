
import numpy as np


class HermiteInterpolator:

    def __init__(self, t_nodes, values, derivatives):
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
        n = self.n
        nd = 2 * n
        self.xd = np.zeros(nd)
        self.xd[0::2] = self.t
        self.xd[1::2] = self.t

        self.yd = np.zeros(nd)
        self.yd[0] = self.y[0]

        for i in range(1, n):
            self.yd[2 * i - 1] = (self.y[i] - self.y[i - 1]) / (self.t[i] - self.t[i - 1])
            self.yd[2 * i] = self.yp[i]
        self.yd[1] = self.yp[0]


        for i in range(3, nd + 1):
            for j in range(nd - 1, i - 2, -1):
                denom = self.xd[j] - self.xd[j - i + 1]
                if np.abs(denom) < 1e-14:
                    self.yd[j] = 0.0
                else:
                    self.yd[j] = (self.yd[j] - self.yd[j - 1]) / denom

    def evaluate(self, t_query):
        t_query = np.atleast_1d(t_query)
        result = np.zeros_like(t_query, dtype=float)
        for idx, t in enumerate(t_query):

            if t < self.xd[0] - 1e-10 or t > self.xd[-1] + 1e-10:

                pass

            nd = len(self.xd)
            val = self.yd[nd - 1]
            for k in range(nd - 2, -1, -1):
                val = self.yd[k] + (t - self.xd[k]) * val
            result[idx] = val
        return result

    def evaluate_derivative(self, t_query):

        eps = 1e-6
        return (self.evaluate(t_query + eps) - self.evaluate(t_query - eps)) / (2.0 * eps)

    def error_bound(self, t_query, M_2n):
        t_query = np.atleast_1d(t_query)
        omega = np.ones_like(t_query)
        for ti in self.t:
            omega *= np.abs(t_query - ti) ** 2
        fact = 1.0
        for k in range(1, 2 * self.n + 1):
            fact *= k
        return M_2n / fact * omega


class StabilityAnalyzer:

    @staticmethod
    def rk4_amplification_factor(z):
        z = np.asarray(z, dtype=complex)
        return 1.0 + z + z ** 2 / 2.0 + z ** 3 / 6.0 + z ** 4 / 24.0

    @staticmethod
    def stability_region_grid(xa, xb, ya, yb, nptsx=201, nptsy=201):
        x = np.linspace(xa, xb, nptsx)
        y = np.linspace(ya, yb, nptsy)
        X, Y = np.meshgrid(x, y)
        Z = X + 1j * Y
        Rval = StabilityAnalyzer.rk4_amplification_factor(Z)
        Rabs = np.abs(Rval)
        return X, Y, Rabs

    @staticmethod
    def is_stable(z):
        Rval = StabilityAnalyzer.rk4_amplification_factor(z)
        return np.abs(Rval) <= 1.0 + 1e-10

    @staticmethod
    def max_stable_timestep(lambda_max, method='rk4'):
        raise NotImplementedError("Hole 2a: 请补全 RK4 最大稳定步长计算")

    @staticmethod
    def neuron_linearized_eigenvalue(V, m, h, n, g_Na=120.0, g_K=36.0, g_L=0.3):
        raise NotImplementedError("Hole 2b: 请补全 HH 线性化特征值公式")


class SignalReconstructor:

    def __init__(self, spike_times, spike_values=None):
        self.spike_times = np.asarray(spike_times, dtype=float)
        self.n_spikes = len(self.spike_times)
        if self.n_spikes < 2:
            raise ValueError("At least two spikes required for reconstruction.")
        if spike_values is None:

            intervals = np.diff(self.spike_times)
            values = 1.0 / (intervals + 1e-6)

            self.spike_values = np.zeros(self.n_spikes)
            self.spike_values[:-1] = values
            self.spike_values[-1] = values[-1]
        else:
            self.spike_values = np.asarray(spike_values, dtype=float)

    def reconstruct_hermite(self, t_query):
        n = len(self.spike_times)
        if n < 2:
            return np.zeros_like(np.atleast_1d(t_query), dtype=float)


        if n >= 4:
            nodes_t = self.spike_times[1:-1]
            nodes_v = self.spike_values[1:-1]
            offset = 1
        else:
            nodes_t = self.spike_times
            nodes_v = self.spike_values
            offset = 0


        dt_nodes = np.diff(self.spike_times)
        dv = np.diff(self.spike_values)
        derivatives = np.zeros(len(nodes_t))
        for i in range(len(nodes_t)):
            idx = i + offset

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
        signal_recon = self.reconstruct_hermite(t_grid)
        mse = np.mean((signal_true - signal_recon) ** 2)
        var_true = np.var(signal_true)
        snr = 10.0 * np.log10(var_true / (mse + 1e-12))
        corr = np.corrcoef(signal_true, signal_recon)[0, 1]
        return {'MSE': mse, 'SNR_dB': snr, 'Correlation': corr}


def demo_hermite_reconstruction():

    t_nodes = np.array([0.0, 2.0, 5.0, 8.0, 12.0, 16.0, 20.0])
    v_nodes = np.array([1.0, 2.5, 1.8, 3.0, 2.2, 1.5, 2.0])
    dv_nodes = np.array([0.5, -0.3, 0.4, -0.2, -0.1, 0.3, 0.1])
    interp = HermiteInterpolator(t_nodes, v_nodes, dv_nodes)
    t_fine = np.linspace(0, 20, 200)
    v_recon = interp.evaluate(t_fine)
    return t_fine, v_recon


def demo_stability_analysis():
    stab = StabilityAnalyzer()
    X, Y, Rabs = stab.stability_region_grid(-4.0, 2.0, -3.0, 3.0, nptsx=101, nptsy=101)

    boundary_points = []
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if np.abs(Rabs[i, j] - 1.0) < 0.05:
                boundary_points.append((X[i, j], Y[i, j]))
    return X, Y, Rabs, boundary_points
