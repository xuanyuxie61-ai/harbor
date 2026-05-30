
import numpy as np

from reactor_mesh import generate_cylindrical_mesh, mesh_quality_report
from stochastic_inlet import generate_inlet_conditions, generate_perturbed_profile
from nonlinear_solver import (fixed_point_iteration, newton_solver,
                               reactor_algebraic_residual, reactor_jacobian)
from momentum_equations import (HartmannFlow, interphase_momentum_exchange,
                                 effective_viscosity_slurry)
from population_balance import (poisson_nucleation_events, qmom_integrate_pbe,
                                 breakage_frequency_lehr, wheeler_algorithm)
from catalyst_optimization import optimize_catalyst_loading
from numerical_linear_algebra import (steady_state_concentration_solver,
                                       power_iteration_eigenvector,
                                       estimate_condition_number)
from spectral_quadrature import gauss_legendre_integral, chebyshev_eval
from reactor_operations import reactor_operation_timeline


class SlurryBubbleColumnReactor:

    def __init__(self, R=0.15, H=3.0, Nr=10, Nz=30,
                 rho_l=800.0, rho_g=20.0, mu_l=0.002,
                 sigma=0.072, g=9.81,
                 T_in=523.0, P_in=2.5e6,
                 u_g_in=0.05, alpha_s=0.25,
                 k_FT=5.8e2, Ea=60000.0, dH_FT=-165e3,
                 Cp_mix=2300.0, k_eff=0.35):
        self.R = R
        self.H = H
        self.Nr = Nr
        self.Nz = Nz
        self.rho_l = rho_l
        self.rho_g = rho_g
        self.mu_l = mu_l
        self.sigma = sigma
        self.g = g
        self.T_in = T_in
        self.P_in = P_in
        self.u_g_in = u_g_in
        self.alpha_s = alpha_s
        self.k_FT = k_FT
        self.Ea = Ea
        self.dH_FT = dH_FT
        self.Cp_mix = Cp_mix
        self.k_eff = k_eff


        self.nodes, self.elements = generate_cylindrical_mesh(R, H, Nr, Nz)
        self.n_nodes = self.nodes.shape[0]
        self.n_elements = self.elements.shape[0]


        self.n_axial = Nz + 1
        self.dz = H / Nz


        self.alpha_g = np.ones(self.n_axial) * 0.25
        self.u_l = np.zeros(self.n_axial)
        self.p = np.ones(self.n_axial) * P_in
        self.T = np.ones(self.n_axial) * T_in


        V0 = np.pi / 6.0 * (5e-3) ** 3
        self.moments = np.array([1.0, V0, V0 ** 2, V0 ** 3])
        self.moments_hist = [self.moments.copy()]


        self.inlet_data = None


        self.convergence_history = []

    def setup_inlet_conditions(self, n_samples=100, seed=42):
        self.inlet_data = generate_inlet_conditions(
            n_samples=n_samples,
            T_mean=self.T_in,
            T_std=3.0,
            yCO_mean=0.30,
            yH2_mean=0.60,
            y_std=0.015,
            Q_mean=np.pi * self.R ** 2 * self.u_g_in,
            Q_std=0.0005,
            seed=seed
        )

    def compute_flow_field(self, max_iter=50, tol=1e-6):
        for it in range(1, max_iter + 1):
            alpha_g_old = self.alpha_g.copy()

            for j in range(self.n_axial):



                raise NotImplementedError("Hole 2: 请实现 compute_flow_field 的循环体")

            diff = np.linalg.norm(self.alpha_g - alpha_g_old, ord=np.inf)
            self.convergence_history.append(diff)
            if diff < tol:
                return True, it, diff

        return False, max_iter, diff

    def update_pbe_moments(self, dt=0.1, n_steps=50):
        try:
            t_arr, m_hist = qmom_integrate_pbe(
                self.moments, (0.0, dt * n_steps), dt,
                n_nodes=2, rho_l=self.rho_l, sigma=self.sigma, epsilon=0.05
            )
            m_new = m_hist[-1]
            if not np.all(np.isfinite(m_new)) or np.any(m_new < 0):

                return
            self.moments = m_new
            self.moments_hist.append(self.moments.copy())
        except Exception:
            pass

    def compute_sauter_diameter(self):
        m2 = self.moments[2]
        m3 = self.moments[3]
        if not np.isfinite(m2) or not np.isfinite(m3) or m2 < 1e-30:
            return 5.0e-3
        d32 = m3 / m2

        if d32 < 1e-4 or d32 > 0.05 or not np.isfinite(d32):
            return 5.0e-3
        return d32

    def compute_interfacial_area(self):
        d32 = self.compute_sauter_diameter()
        a_i = 6.0 * self.alpha_g / max(d32, 1e-9)
        return a_i

    def compute_temperature_profile(self):
        R_gas = 8.314
        k = self.k_FT * np.exp(-self.Ea / (R_gas * self.T))
        z = np.arange(self.n_axial) * self.dz

        C_total = self.P_in / (R_gas * self.T)
        C_CO = 0.30 * np.exp(-0.5 * z / self.H) * C_total


        eta_eff = 0.08
        r_FT = eta_eff * k * C_CO



        rho_m = self.alpha_g * self.rho_g + (1.0 - self.alpha_g) * self.rho_l
        u_l_clip = np.clip(np.abs(self.u_l), 1e-6, None)

        T_new = np.zeros(self.n_axial)
        T_new[0] = self.T_in
        for j in range(1, self.n_axial):

            q_rxn = (-self.dH_FT) * r_FT[j]

            q_cool = 500.0 * (T_new[j - 1] - self.T_in)
            dTdz = (q_rxn - q_cool) / (rho_m[j] * self.Cp_mix * u_l_clip[j])
            T_new[j] = T_new[j - 1] + dTdz * self.dz


        T_new = np.clip(T_new, self.T_in, 623.0)
        self.T = T_new

    def compute_species_concentration(self):
        n = self.n_axial
        A = np.zeros((n, n))
        b = np.zeros(n)

        a_i = self.compute_interfacial_area()

        d32 = self.compute_sauter_diameter()
        u_slip = 0.23
        D_CO = 2.5e-9
        k_L = np.sqrt(4.0 * D_CO * u_slip / (np.pi * max(d32, 1e-9)))


        eta_mt = 0.001
        kLa = eta_mt * k_L * a_i
        k_r = self.k_FT * np.exp(-self.Ea / (8.314 * self.T))

        yCO_in = 0.30
        for j in range(n):
            u = max(abs(self.u_l[j]), 1e-6)
            adv = u / self.dz
            kLa_j = kLa[j] if np.isfinite(kLa[j]) else 0.0
            k_r_j = k_r[j] if np.isfinite(k_r[j]) else 0.0
            if j == 0:
                A[j, j] = adv + kLa_j + k_r_j
                b[j] = adv * yCO_in
            else:
                A[j, j - 1] = -adv
                A[j, j] = adv + kLa_j + k_r_j
                b[j] = 0.0


        if not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)):

            c = np.full(n, yCO_in * np.exp(-np.arange(n) * 0.1))
            return c, 0.0, 0, True


        try:
            cond = estimate_condition_number(A)
            if cond > 1e12:
                A = A + 1e-8 * np.eye(n)
        except Exception:
            A = A + 1e-8 * np.eye(n)

        c, res, it, conv = steady_state_concentration_solver(
            A, b, alpha_relax=0.6, max_iter=500, tol=1e-8
        )
        return c, res, it, conv

    def evaluate_catalyst_distribution(self):
        n_seg = min(self.n_axial, 10)
        T_profile = np.interp(
            np.linspace(0, self.n_axial - 1, n_seg),
            np.arange(self.n_axial),
            self.T
        )
        result = optimize_catalyst_loading(
            W_total=50.0,
            n_segments=n_seg,
            T_profile=T_profile,
            Q_gas=np.pi * self.R ** 2 * self.u_g_in,
            method='brute_force'
        )
        return result

    def run_simulation(self, verbose=False):
        if self.inlet_data is None:
            self.setup_inlet_conditions()


        conv_flow, it_flow, diff_flow = self.compute_flow_field()
        if verbose:
            print(f"[Flow] Converged={conv_flow}, it={it_flow}, diff={diff_flow:.3e}")


        self.update_pbe_moments()
        if verbose:
            print(f"[PBE] Moments: {self.moments}")


        self.compute_temperature_profile()
        if verbose:
            print(f"[Energy] T_max={self.T.max():.2f} K, T_min={self.T.min():.2f} K")


        c_CO, res_CO, it_CO, conv_CO = self.compute_species_concentration()
        if verbose:
            print(f"[Species] CO outlet={c_CO[-1]:.4f}, res={res_CO:.3e}, conv={conv_CO}")


        cat_result = self.evaluate_catalyst_distribution()
        if verbose:
            print(f"[Catalyst] Max value={cat_result['max_value']:.4f}")


        mesh_report = mesh_quality_report(self.nodes, self.elements)
        if verbose:
            print(f"[Mesh] Jacobian min={mesh_report['jacobian_min']:.3e}, "
                  f"neg_count={mesh_report['jacobian_negative_count']}")


        hart = HartmannFlow(G=1.0, Ha=2.0, Re=10.0, Rm=6.0)
        y_test = np.linspace(-0.9, 0.9, 5)
        ur, br = hart.residual_check(y_test)
        hartmann_error = max(np.max(np.abs(ur)), np.max(np.abs(br)))
        if verbose:
            print(f"[Hartmann] Residual max={hartmann_error:.3e}")


        timeline = reactor_operation_timeline(
            (2024, 1, 1), (2024, 12, 31)
        )
        if verbose:
            print(f"[Ops] Operating days={timeline['total_days']}, "
                  f"max cycles={timeline['max_cycles']}")

        results = {
            'converged_flow': conv_flow,
            'flow_iterations': it_flow,
            'moments': self.moments.copy(),
            'sauter_diameter': self.compute_sauter_diameter(),
            'interfacial_area_mean': float(np.mean(self.compute_interfacial_area())),
            'temperature_max': float(self.T.max()),
            'temperature_min': float(self.T.min()),
            'CO_outlet': float(c_CO[-1]),
            'CO_conversion': float(1.0 - c_CO[-1] / max(c_CO[0], 1e-12)),
            'species_residual': res_CO,
            'catalyst_result': cat_result,
            'mesh_report': mesh_report,
            'hartmann_benchmark_error': float(hartmann_error),
            'operational_timeline': timeline,
            'inlet_statistics': self.inlet_data['statistics'] if self.inlet_data else None,
        }
        return results
