
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


class CoagulationNetwork:

    SPECIES_NAMES = [
        "TF_VIIa",
        "IXa",
        "Xa",
        "Va",
        "IIa",
        "Fibrin",
        "APC",
        "ATIII_Xa",
        "TFPI",
        "Plasmin",
        "tPA",
        "Platelet_act",
    ]

    def __init__(self, params=None):
        if params is None:
            params = self._default_params()
        self.params = params
        self.n_species = len(self.SPECIES_NAMES)

    def _default_params(self):
        return {

            "K_M_IX": 150.0,
            "K_M_X": 250.0,
            "K_M_II": 100.0,
            "K_M_PC": 50.0,

            "k_cat_TF_VIIa_IX": 1.2,
            "k_cat_IXa_X": 6.5,
            "k_cat_TF_VIIa_X": 0.8,
            "k_cat_Xa_II": 25.0,
            "k_cat_IIa_Fibrin": 15.0,

            "k_inact_ATIII": 0.00005,
            "k_TFPI_inact": 0.005,
            "k_APC_inact_Va": 0.01,
            "k_clear": 0.001,

            "k_polymerization": 2.0,
            "n_Hill": 3.0,
            "K_poly_half": 50.0,

            "k_plasminogen_act": 0.02,
            "k_fibrin_lysis": 0.008,

            "k_PLT_act": 0.1,
            "K_PLT_half": 10.0,

            "tot_IX": 180.0,
            "tot_X": 300.0,
            "tot_II": 1500.0,
            "tot_PC": 80.0,
            "tot_ATIII": 3400.0,
            "tot_TFPI": 2.5,
            "tot_tPA": 0.07,
            "tot_TM": 1.0,
        }

    def rhs(self, y, t=0.0):
        p = self.params
        y = np.asarray(y, dtype=float)
        if y.ndim != 1 or y.shape[0] != self.n_species:
            raise ValueError(f"y 必须为长度 {self.n_species} 的一维数组")
        if np.any(y < 0):

            y = np.maximum(y, 0.0)


        TF_VIIa, IXa, Xa, Va, IIa, Fibrin, APC, ATIII_Xa, TFPI_free, Plasmin, tPA_free, PLT_act = y


        IX = max(p["tot_IX"] - IXa, 0.0)
        X = max(p["tot_X"] - Xa, 0.0)
        II = max(p["tot_II"] - IIa, 0.0)
        PC = max(p["tot_PC"] - APC, 0.0)
        ATIII = max(p["tot_ATIII"] - ATIII_Xa, 0.0)
        tPA_tot = p["tot_tPA"]




        pass

    def jacobian(self, y, t=0.0):
        eps_jac = 1e-7
        n = self.n_species
        J = np.zeros((n, n))
        for j in range(n):
            h_j = eps_jac * max(1.0, abs(y[j]))
            y_plus = y.copy()
            y_minus = y.copy()
            y_plus[j] += h_j
            y_minus[j] -= h_j
            f_plus = self.rhs(y_plus, t)
            f_minus = self.rhs(y_minus, t)
            J[:, j] = (f_plus - f_minus) / (2.0 * h_j)
        return J


def trapezoidal_solve(network, y0, t_span, n_steps=1000, tol=1e-8, max_iter=20):
    from scipy.integrate import solve_ivp

    t0, t1 = t_span
    if t0 >= t1:
        raise ValueError("t_span 必须满足 t0 < t1")
    if n_steps < 1:
        raise ValueError("n_steps 必须 >= 1")

    def ode_func(t, y):
        return network.rhs(y, t)

    t_eval = np.linspace(t0, t1, n_steps + 1)
    sol = solve_ivp(
        ode_func,
        t_span,
        y0,
        method='BDF',
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-9,
        jac=lambda t, y: network.jacobian(y, t),
        dense_output=True
    )

    if not sol.success:
        raise RuntimeError(f"ODE求解失败: {sol.message}")

    return sol.t, sol.y.T


def simulate_coagulation(t_end=1200.0, n_steps=2000):
    net = CoagulationNetwork()
    y0 = np.zeros(net.n_species)
    y0[0] = 5.0
    y0[net.SPECIES_NAMES.index("Va")] = 0.5
    y0[net.SPECIES_NAMES.index("IIa")] = 0.01
    y0[net.SPECIES_NAMES.index("ATIII_Xa")] = 0.0
    y0[net.SPECIES_NAMES.index("TFPI")] = net.params["tot_TFPI"]
    y0[net.SPECIES_NAMES.index("tPA")] = net.params["tot_tPA"]

    t, y = trapezoidal_solve(net, y0, (0.0, t_end), n_steps=n_steps)
    return t, y, net


if __name__ == "__main__":
    t, y, net = simulate_coagulation(t_end=300.0, n_steps=500)
    print("模拟完成，时间点:", t[0], "...", t[-1])
    print("最终凝血酶浓度 (IIa):", y[-1, 4], "nM")
    print("最终纤维蛋白浓度:", y[-1, 5], "nM")
