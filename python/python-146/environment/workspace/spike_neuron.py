
import numpy as np


class HHNeuron:


    C_M = 1.0
    G_NA = 120.0
    G_K = 36.0
    G_L = 0.3
    E_NA = 50.0
    E_K = -77.0
    E_L = -54.387
    V_REST = -65.0
    V_TH = -50.0
    V_RESET = -65.0
    TAU_REF = 2.0

    def __init__(self, dt=0.01):
        if dt <= 0.0:
            raise ValueError("dt must be positive.")
        if dt > 0.1:
            raise ValueError("dt too large for HH stability (require dt <= 0.1 ms).")
        self.dt = dt
        self.V = self.V_REST
        self.m = self._alpha_m(self.V_REST) / (self._alpha_m(self.V_REST) + self._beta_m(self.V_REST))
        self.h = self._alpha_h(self.V_REST) / (self._alpha_h(self.V_REST) + self._beta_h(self.V_REST))
        self.n = self._alpha_n(self.V_REST) / (self._alpha_n(self.V_REST) + self._beta_n(self.V_REST))
        self.refractory = 0
        self.spike_times = []




    @staticmethod
    def _alpha_m(V):


        eps = 1e-6
        denom = 1.0 - np.exp(-(V + 40.0) / 10.0)
        if np.abs(denom) < eps:
            return 1.0
        return 0.1 * (V + 40.0) / denom

    @staticmethod
    def _beta_m(V):
        return 4.0 * np.exp(-(V + 65.0) / 18.0)

    @staticmethod
    def _alpha_h(V):
        return 0.07 * np.exp(-(V + 65.0) / 20.0)

    @staticmethod
    def _beta_h(V):
        return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))

    @staticmethod
    def _alpha_n(V):

        eps = 1e-6
        denom = 1.0 - np.exp(-(V + 55.0) / 10.0)
        if np.abs(denom) < eps:
            return 0.1
        return 0.01 * (V + 55.0) / denom

    @staticmethod
    def _beta_n(V):
        return 0.125 * np.exp(-(V + 65.0) / 80.0)




    def _gate_derivatives(self, V, m, h, n):
        dmdt = self._alpha_m(V) * (1.0 - m) - self._beta_m(V) * m
        dhdt = self._alpha_h(V) * (1.0 - h) - self._beta_h(V) * h
        dndt = self._alpha_n(V) * (1.0 - n) - self._beta_n(V) * n
        return np.array([dmdt, dhdt, dndt])




    def _dVdt(self, V, m, h, n, I_syn, I_ext):
        raise NotImplementedError("Hole 1: 请补全 HH 膜电位导数公式")




    def step(self, t, I_syn=0.0, I_ext=0.0):

        if self.refractory > 0:
            self.refractory -= 1
            self.V = self.V_RESET
            return False

        dt = self.dt
        V0, m0, h0, n0 = self.V, self.m, self.h, self.n


        dV1 = self._dVdt(V0, m0, h0, n0, I_syn, I_ext)
        dg1 = self._gate_derivatives(V0, m0, h0, n0)


        V2 = V0 + 0.5 * dt * dV1
        m2 = m0 + 0.5 * dt * dg1[0]
        h2 = h0 + 0.5 * dt * dg1[1]
        n2 = n0 + 0.5 * dt * dg1[2]
        dV2 = self._dVdt(V2, m2, h2, n2, I_syn, I_ext)
        dg2 = self._gate_derivatives(V2, m2, h2, n2)


        V3 = V0 + 0.5 * dt * dV2
        m3 = m0 + 0.5 * dt * dg2[0]
        h3 = h0 + 0.5 * dt * dg2[1]
        n3 = n0 + 0.5 * dt * dg2[2]
        dV3 = self._dVdt(V3, m3, h3, n3, I_syn, I_ext)
        dg3 = self._gate_derivatives(V3, m3, h3, n3)


        V4 = V0 + dt * dV3
        m4 = m0 + dt * dg3[0]
        h4 = h0 + dt * dg3[1]
        n4 = n0 + dt * dg3[2]
        dV4 = self._dVdt(V4, m4, h4, n4, I_syn, I_ext)
        dg4 = self._gate_derivatives(V4, m4, h4, n4)


        self.V = V0 + dt / 6.0 * (dV1 + 2.0 * dV2 + 2.0 * dV3 + dV4)
        self.m = m0 + dt / 6.0 * (dg1[0] + 2.0 * dg2[0] + 2.0 * dg3[0] + dg4[0])
        self.h = h0 + dt / 6.0 * (dg1[1] + 2.0 * dg2[1] + 2.0 * dg3[1] + dg4[1])
        self.n = n0 + dt / 6.0 * (dg1[2] + 2.0 * dg2[2] + 2.0 * dg3[2] + dg4[2])


        self.m = np.clip(self.m, 0.0, 1.0)
        self.h = np.clip(self.h, 0.0, 1.0)
        self.n = np.clip(self.n, 0.0, 1.0)


        if self.V >= self.V_TH:
            self.V = self.V_RESET
            self.refractory = int(np.round(self.TAU_REF / dt))
            self.spike_times.append(t)
            return True
        return False


class NeuronPopulation:

    def __init__(self, N_exc, N_inh, dt=0.01, p_conn=0.2):
        self.N_exc = N_exc
        self.N_inh = N_inh
        self.N = N_exc + N_inh
        self.dt = dt
        self.neurons = [HHNeuron(dt) for _ in range(self.N)]
        self.W = self._build_weight_matrix(p_conn)
        self.spike_record = [[] for _ in range(self.N)]

    def _build_weight_matrix(self, p_conn):
        np.random.seed(42)
        W = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                if i == j:
                    continue
                if np.random.rand() < p_conn:
                    if j < self.N_exc:
                        W[i, j] = np.random.uniform(0.5, 2.0)
                    else:
                        W[i, j] = np.random.uniform(-2.0, -0.5)
        return W

    def compute_synaptic_current(self, spike_vector):
        return self.W @ spike_vector

    def simulate(self, T_total, I_ext_per_neuron=None):
        n_steps = int(np.round(T_total / self.dt))
        if I_ext_per_neuron is None:
            I_ext_per_neuron = np.zeros(self.N)
        if len(I_ext_per_neuron) != self.N:
            raise ValueError("I_ext_per_neuron length must match N.")

        voltage_trace = np.zeros((self.N, n_steps))
        spike_raster = np.zeros((self.N, n_steps), dtype=int)

        for step_idx in range(n_steps):
            t = step_idx * self.dt

            spike_vec = spike_raster[:, step_idx - 1] if step_idx > 0 else np.zeros(self.N)
            I_syn = self.compute_synaptic_current(spike_vec)

            for i in range(self.N):
                fired = self.neurons[i].step(t, I_syn[i], I_ext_per_neuron[i])
                voltage_trace[i, step_idx] = self.neurons[i].V
                if fired:
                    spike_raster[i, step_idx] = 1
                    self.spike_record[i].append(t)

        return voltage_trace, spike_raster


def demo_single_neuron():
    neuron = HHNeuron(dt=0.01)
    T = 50.0
    n_steps = int(T / neuron.dt)
    V_trace = np.zeros(n_steps)
    I_ext = 10.0
    for k in range(n_steps):
        t = k * neuron.dt
        neuron.step(t, I_ext=I_ext)
        V_trace[k] = neuron.V
    return V_trace, neuron.spike_times


def demo_population():
    pop = NeuronPopulation(N_exc=10, N_inh=5, dt=0.01, p_conn=0.3)
    I_ext = np.concatenate([
        np.random.uniform(5.0, 12.0, pop.N_exc),
        np.random.uniform(3.0, 8.0, pop.N_inh)
    ])
    voltage_trace, spike_raster = pop.simulate(T_total=30.0, I_ext_per_neuron=I_ext)
    return voltage_trace, spike_raster, pop.spike_record
