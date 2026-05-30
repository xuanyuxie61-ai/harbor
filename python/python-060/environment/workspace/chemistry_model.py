
import numpy as np
from typing import Dict, Tuple, Optional


R_GAS = 8.314
N_A = 6.022e23
H_PLANCK = 6.626e-34
C_LIGHT = 2.998e8
K_BOLTZMANN = 1.381e-23


class StratosphericChemistry:

    def __init__(self, num_altitude_levels: int = 80,
                 t_min: float = 180.0, t_max: float = 270.0):
        if num_altitude_levels < 10:
            raise ValueError("高度层数必须 >= 10")
        if t_min <= 0 or t_max <= 0:
            raise ValueError("温度必须为正")

        self.nz = num_altitude_levels
        self.z = np.linspace(10000.0, 50000.0, num_altitude_levels)
        self.dz = self.z[1] - self.z[0]


        self.T = self._temperature_profile(self.z)
        self.T = np.clip(self.T, t_min, t_max)


        self.M_air = 0.0289644


        self.P = self._pressure_profile(self.z)
        self.P = np.clip(self.P, 1.0, 1013.25)


        self.rho_air = self.P * 100.0 / (R_GAS * self.T)
        self.rho_air = np.clip(self.rho_air, 1e-6, 100.0)


        self.species = {
            'O': np.ones(num_altitude_levels) * 1e6,
            'O3': np.ones(num_altitude_levels) * 1e12,
            'O2': np.ones(num_altitude_levels) * 5e18,
            'N2': np.ones(num_altitude_levels) * 2e19,
            'NO': np.ones(num_altitude_levels) * 1e9,
            'NO2': np.ones(num_altitude_levels) * 5e8,
            'Cl': np.ones(num_altitude_levels) * 1e3,
            'ClO': np.ones(num_altitude_levels) * 5e2,
            'OH': np.ones(num_altitude_levels) * 1e6,
            'HO2': np.ones(num_altitude_levels) * 5e6,
            'H': np.ones(num_altitude_levels) * 1e2,
            'Br': np.ones(num_altitude_levels) * 1e2,
            'BrO': np.ones(num_altitude_levels) * 5e1,
        }


        self.reactions = self._init_reactions()


        self.J_rates = self._init_photolysis_rates()


        self.Kzz = self._init_eddy_diffusion()


        self.emission_profile = self._init_emission_profile()

    def _temperature_profile(self, z: np.ndarray) -> np.ndarray:
        T = np.zeros_like(z)
        z_km = z / 1000.0
        for i, zk in enumerate(z_km):
            if zk <= 11.0:
                T[i] = 216.65
            elif zk <= 20.0:
                T[i] = 216.65 + 1.0 * (zk - 11.0)
            elif zk <= 32.0:
                T[i] = 216.65 + 1.0 * 9.0 + 2.8 * (zk - 20.0)
            elif zk <= 47.0:
                T[i] = 216.65 + 9.0 + 33.6 - 2.8 * (zk - 32.0)
            else:
                T[i] = 270.65 - 2.0 * (zk - 47.0)
        return T

    def _pressure_profile(self, z: np.ndarray) -> np.ndarray:
        g = 9.81
        H_scale = R_GAS * np.mean(self._temperature_profile(z)) / (self.M_air * g)
        P = 1013.25 * np.exp(-z / (H_scale * 1000.0))
        return P

    def _init_reactions(self) -> Dict:
        reactions = {

            'R1': {'A': 6.0e-34, 'Ea': 0.0, 'type': 'termolecular',
                   'reactants': ['O', 'O2', 'M'], 'products': ['O3']},
            'R2': {'A': 8.0e-12, 'Ea': 2060.0 * R_GAS, 'type': 'bimolecular',
                   'reactants': ['O', 'O3'], 'products': ['O2', 'O2']},

            'R3': {'A': 1.8e-12, 'Ea': 1370.0 * R_GAS, 'type': 'bimolecular',
                   'reactants': ['NO', 'O3'], 'products': ['NO2', 'O2']},
            'R4': {'A': 9.3e-12, 'Ea': 0.0, 'type': 'bimolecular',
                   'reactants': ['NO2', 'O'], 'products': ['NO', 'O2']},
            'R5': {'A': 1.0e-20, 'Ea': 0.0, 'type': 'termolecular',
                   'reactants': ['NO', 'O', 'M'], 'products': ['NO2']},

            'R6': {'A': 2.9e-11, 'Ea': 260.0 * R_GAS, 'type': 'bimolecular',
                   'reactants': ['Cl', 'O3'], 'products': ['ClO', 'O2']},
            'R7': {'A': 2.8e-11, 'Ea': 0.0, 'type': 'bimolecular',
                   'reactants': ['ClO', 'O'], 'products': ['Cl', 'O2']},

            'R8': {'A': 1.7e-12, 'Ea': 940.0 * R_GAS, 'type': 'bimolecular',
                   'reactants': ['OH', 'O3'], 'products': ['HO2', 'O2']},
            'R9': {'A': 3.0e-11, 'Ea': 200.0 * R_GAS, 'type': 'bimolecular',
                   'reactants': ['HO2', 'O'], 'products': ['OH', 'O2']},
            'R10': {'A': 7.2e-11, 'Ea': 0.0, 'type': 'bimolecular',
                    'reactants': ['OH', 'O'], 'products': ['H', 'O2']},

            'R11': {'A': 1.7e-12, 'Ea': 600.0 * R_GAS, 'type': 'bimolecular',
                    'reactants': ['Br', 'O3'], 'products': ['BrO', 'O2']},
            'R12': {'A': 1.5e-11, 'Ea': 0.0, 'type': 'bimolecular',
                    'reactants': ['BrO', 'O'], 'products': ['Br', 'O2']},
        }
        return reactions

    def _init_photolysis_rates(self) -> Dict[str, np.ndarray]:
        J = {}
        theta_sza = np.deg2rad(45.0)
        sec_theta = 1.0 / np.cos(theta_sza)


        tau_o2 = np.cumsum(self.species['O2'] * 1e6 * self.dz * 1e-5)
        tau_o2 = np.clip(tau_o2, 0.0, 50.0)
        J['J_O2'] = 1e-10 * np.exp(-tau_o2 * sec_theta)


        tau_o3 = np.cumsum(self.species['O3'] * 1e-20 * self.dz * 1e-2)
        tau_o3 = np.clip(tau_o3, 0.0, 30.0)
        J['J_O3'] = 1e-2 * np.exp(-tau_o3 * sec_theta)


        J['J_NO2'] = 1e-2 * np.exp(-tau_o3 * sec_theta * 0.5)


        J['J_ClO'] = 1e-5 * np.ones(self.nz)

        return J

    def _init_eddy_diffusion(self) -> np.ndarray:
        z_km = self.z / 1000.0
        Kzz = 0.5 + 2.0 * np.exp(-((z_km - 25.0) / 10.0) ** 2)
        Kzz = np.clip(Kzz, 0.01, 50.0)
        return Kzz

    def _init_emission_profile(self) -> Dict[str, np.ndarray]:
        profiles = {}
        z_km = self.z / 1000.0


        profiles['N2O'] = np.exp(-((z_km - 0.0) / 5.0) ** 2)
        profiles['N2O'] = profiles['N2O'] / (np.sum(profiles['N2O']) + 1e-30)


        profiles['CFC11'] = np.exp(-((z_km - 0.0) / 3.0) ** 2)
        profiles['CFC11'] = profiles['CFC11'] / (np.sum(profiles['CFC11']) + 1e-30)


        profiles['NOx_aircraft'] = np.exp(-((z_km - 11.0) / 2.0) ** 2)
        profiles['NOx_aircraft'] = profiles['NOx_aircraft'] / (np.sum(profiles['NOx_aircraft']) + 1e-30)

        return profiles

    def arrhenius_rate(self, A: float, Ea: float, T: float) -> float:
        if T <= 0:
            raise ValueError("温度必须大于零")
        k = A * np.exp(-Ea / (R_GAS * T))
        return max(k, 1e-40)

    def termolecular_rate(self, k0: float, kinf: float, M: float,
                          T: float, F: float = 0.6) -> float:
        if M <= 0 or T <= 0:
            return 0.0

        k0_T = k0 * (T / 300.0) ** (-2.0)
        kinf_T = kinf * (T / 300.0) ** (-1.0)

        k0M = k0_T * M
        ratio = k0M / (kinf_T + 1e-40)

        N = 0.75 - 1.27 * np.log10(F)
        c = -0.4 - 0.67 * np.log10(ratio + 1e-40)

        log_ratio = np.log10(ratio + 1e-40)
        d = 1.0 + (log_ratio / N) ** 2

        k = k0M / (1.0 + ratio) * F ** (1.0 / d)
        return max(k, 1e-40)

    def compute_reaction_rates(self) -> Dict[str, np.ndarray]:
        rates = {}
        nz = self.nz

        for key, rxn in self.reactions.items():
            rate = np.zeros(nz)
            T_local = self.T

            if rxn['type'] == 'bimolecular':
                A = rxn['A']
                Ea = rxn['Ea']
                r1 = rxn['reactants'][0]
                r2 = rxn['reactants'][1]
                for iz in range(nz):
                    k = self.arrhenius_rate(A, Ea, T_local[iz])
                    conc1 = max(self.species[r1][iz], 0.0)
                    conc2 = max(self.species[r2][iz], 0.0)
                    rate[iz] = k * conc1 * conc2

            elif rxn['type'] == 'termolecular':
                A = rxn['A']
                Ea = rxn['Ea']
                r1 = rxn['reactants'][0]
                r2 = rxn['reactants'][1]

                for iz in range(nz):
                    k = self.arrhenius_rate(A, Ea, T_local[iz])
                    conc1 = max(self.species[r1][iz], 0.0)
                    conc2 = max(self.species[r2][iz], 0.0)
                    M = self.rho_air[iz] * 1e-6
                    rate[iz] = k * conc1 * conc2 * M

            rates[key] = np.clip(rate, 0.0, 1e30)

        return rates

    def compute_production_loss(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        rates = self.compute_reaction_rates()
        nz = self.nz

        production = {s: np.zeros(nz) for s in self.species}
        loss = {s: np.zeros(nz) for s in self.species}


        for iz in range(nz):

            production['O'][iz] += 2.0 * self.J_rates['J_O2'][iz] * max(self.species['O2'][iz], 0.0)
            loss['O2'][iz] += self.J_rates['J_O2'][iz] * max(self.species['O2'][iz], 0.0)


            production['O'][iz] += self.J_rates['J_O3'][iz] * max(self.species['O3'][iz], 0.0)
            production['O2'][iz] += self.J_rates['J_O3'][iz] * max(self.species['O3'][iz], 0.0)
            loss['O3'][iz] += self.J_rates['J_O3'][iz] * max(self.species['O3'][iz], 0.0)


            production['NO'][iz] += self.J_rates['J_NO2'][iz] * max(self.species['NO2'][iz], 0.0)
            production['O'][iz] += self.J_rates['J_NO2'][iz] * max(self.species['NO2'][iz], 0.0)
            loss['NO2'][iz] += self.J_rates['J_NO2'][iz] * max(self.species['NO2'][iz], 0.0)


        for key, rate_arr in rates.items():
            rxn = self.reactions[key]
            reacs = rxn['reactants']
            prods = rxn['products']

            for iz in range(nz):
                r = max(rate_arr[iz], 0.0)

                for reac in reacs:
                    if reac in loss and reac != 'M':
                        loss[reac][iz] += r

                for prod in prods:
                    if prod in production:
                        production[prod][iz] += r


        for s in production:
            production[s] = np.clip(production[s], 0.0, 1e30)
            loss[s] = np.clip(loss[s], 0.0, 1e30)

        return production, loss

    def ozone_tendency(self) -> np.ndarray:
        production, loss = self.compute_production_loss()
        tendency = production['O3'] - loss['O3']
        return np.clip(tendency, -1e20, 1e20)

    def ozone_column_density(self) -> float:

        n_o3 = self.species['O3']

        dz_cm = self.dz * 100.0
        toc = np.trapezoid(n_o3, dx=dz_cm)

        du = toc / 2.69e16
        return du

    def update_species(self, dt: float, production: Dict[str, np.ndarray],
                       loss: Dict[str, np.ndarray]) -> None:
        if dt <= 0:
            raise ValueError("时间步长必须为正")

        for s in self.species:
            if s in ['O', 'OH', 'Cl', 'Br']:

                k_loss = np.clip(loss[s] / (self.species[s] + 1e-30), 1e-30, 1e30)
                self.species[s] = np.clip(production[s] / k_loss, 1e-3, 1e15)
            else:

                k_loss = np.clip(loss[s] / (self.species[s] + 1e-30), 0.0, 1e30)
                n_new = (self.species[s] + dt * production[s]) / (1.0 + dt * k_loss)
                self.species[s] = np.clip(n_new, 1e-3, 1e20)

    def get_state_vector(self) -> np.ndarray:
        state = []
        for s in sorted(self.species.keys()):
            state.extend(self.species[s])
        return np.array(state)

    def set_state_vector(self, state: np.ndarray) -> None:
        n_species = len(self.species)
        expected_len = n_species * self.nz
        if len(state) != expected_len:
            raise ValueError(f"状态向量长度不匹配: {len(state)} != {expected_len}")

        idx = 0
        for s in sorted(self.species.keys()):
            self.species[s] = np.clip(state[idx:idx + self.nz], 1e-30, 1e25)
            idx += self.nz

    def compute_jacobian_diagonal(self, species_name: Optional[str] = None) -> np.ndarray:
        production, loss = self.compute_production_loss()
        if species_name is not None:
            if species_name not in self.species:
                raise KeyError(f"未知物种: {species_name}")
            diag = -np.clip(loss[species_name] / (self.species[species_name] + 1e-30), 0.0, 1e20)
            return diag
        jac_diag = []
        for s in sorted(self.species.keys()):
            diag = -np.clip(loss[s] / (self.species[s] + 1e-30), 0.0, 1e20)
            jac_diag.extend(diag)
        return np.array(jac_diag)
