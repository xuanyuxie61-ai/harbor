
import math
from typing import Dict, Any





DEFAULT_CONFIG: Dict[str, Any] = {
    "planet": {
        "name": "Earth",
        "radius_cmb": 3480.0e3,
        "radius_icb": 1221.0e3,
        "rotation_rate": 7.2921159e-5,
        "gravity_cmb": 10.68,
    },
    "core_physics": {
        "magnetic_diffusivity": 2.0,
        "kinematic_viscosity": 1.0e-6,
        "thermal_diffusivity": 5.0e-6,
        "density": 1.05e4,
        "electrical_conductivity": 5.0e5,
        "alpha_effect_amplitude": 0.5,
        "omega_effect_shear": 1.0,
    },
    "numerics": {
        "l_max": 8,
        "n_radial": 32,
        "t_end_years": 1.0e6,
        "dt_init_years": 100.0,
        "adaptive_tol": 1.0e-4,
        "cg_tol": 1.0e-10,
        "cg_maxiter": 2000,
    },
    "uncertainty_quantification": {
        "enable": True,
        "sparse_grid_level": 3,
        "uq_dim": 4,
        "param_ranges": {
            "eta": [1.0, 3.0],
            "alpha": [0.3, 0.7],
            "omega_shear": [0.5, 1.5],
            "l_max": [6, 10],
        }
    },
    "io": {
        "output_prefix": "dynamo_043",
        "save_interval_years": 1.0e4,
    }
}





class ConfigParser:

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config if config is not None else DEFAULT_CONFIG.copy()
        self._validate()

    def _validate(self):
        p = self.config["planet"]
        c = self.config["core_physics"]
        n = self.config["numerics"]
        uq = self.config["uncertainty_quantification"]


        assert 0.0 < p["radius_icb"] < p["radius_cmb"], \
            "ICB 半径必须小于 CMB 半径"
        assert p["radius_cmb"] < 1.0e8, \
            "行星半径过大，疑似单位错误"


        assert c["magnetic_diffusivity"] > 0.0, \
            "磁扩散系数必须为正"
        assert c["thermal_diffusivity"] > 0.0, \
            "热扩散系数必须为正"


        assert n["l_max"] >= 1 and n["l_max"] <= 32, \
            "球谐截断阶数必须在 [1, 32] 之间"
        assert n["n_radial"] >= 4 and n["n_radial"] <= 256, \
            "径向网格数必须在 [4, 256] 之间"
        assert n["adaptive_tol"] > 0.0 and n["adaptive_tol"] < 1.0, \
            "自适应容限必须在 (0, 1) 之间"


        if uq["enable"]:
            assert uq["sparse_grid_level"] >= 1 and uq["sparse_grid_level"] <= 6, \
                "稀疏网格层级必须在 [1, 6] 之间"
            assert uq["uq_dim"] >= 1 and uq["uq_dim"] <= 8, \
                "不确定维度必须在 [1, 8] 之间"

    def get(self, section: str, key: str, default: Any = None) -> Any:
        sec = self.config.get(section, {})
        return sec.get(key, default)

    def planet_radius_cmb(self) -> float:
        return float(self.config["planet"]["radius_cmb"])

    def planet_radius_icb(self) -> float:
        return float(self.config["planet"]["radius_icb"])

    def shell_thickness(self) -> float:
        return self.planet_radius_cmb() - self.planet_radius_icb()

    def rotation_rate(self) -> float:
        return float(self.config["planet"]["rotation_rate"])

    def magnetic_diffusivity(self) -> float:
        return float(self.config["core_physics"]["magnetic_diffusivity"])

    def alpha_amplitude(self) -> float:
        return float(self.config["core_physics"]["alpha_effect_amplitude"])

    def omega_shear(self) -> float:
        return float(self.config["core_physics"]["omega_effect_shear"])

    def l_max(self) -> int:
        return int(self.config["numerics"]["l_max"])

    def n_radial(self) -> int:
        return int(self.config["numerics"]["n_radial"])

    def t_end_seconds(self) -> float:
        years = float(self.config["numerics"]["t_end_years"])
        return years * 365.25 * 24.0 * 3600.0

    def dt_init_seconds(self) -> float:
        years = float(self.config["numerics"]["dt_init_years"])
        return years * 365.25 * 24.0 * 3600.0

    def save_interval_seconds(self) -> float:
        years = float(self.config["io"]["save_interval_years"])
        return years * 365.25 * 24.0 * 3600.0

    def magnetic_reynolds_number(self) -> float:
        U = self.alpha_amplitude() * self.omega_shear() * self.rotation_rate() * self.shell_thickness()
        L = self.shell_thickness()
        eta = self.magnetic_diffusivity()
        return U * L / eta if eta > 0.0 else 0.0

    def ekman_number(self) -> float:
        nu = self.config["core_physics"]["kinematic_viscosity"]
        Omega = self.rotation_rate()
        L = self.shell_thickness()
        return nu / (2.0 * Omega * L * L)

    def magnetic_rossby_number(self) -> float:
        U = self.alpha_amplitude() * self.omega_shear() * self.rotation_rate() * self.shell_thickness()
        return U / (2.0 * self.rotation_rate() * self.shell_thickness())

    def to_dict(self) -> Dict[str, Any]:
        return self.config.copy()


def load_config(config_dict: Dict[str, Any] = None) -> ConfigParser:
    return ConfigParser(config_dict)





def _self_test():
    cfg = load_config()
    assert cfg.planet_radius_cmb() == 3480.0e3
    assert cfg.planet_radius_icb() == 1221.0e3
    assert cfg.shell_thickness() == 3480.0e3 - 1221.0e3
    assert cfg.magnetic_reynolds_number() > 0.0
    assert cfg.ekman_number() > 0.0
    print(f"config_parser: Rm={cfg.magnetic_reynolds_number():.4e}, E={cfg.ekman_number():.4e}")
    print("config_parser: self-test passed.")


if __name__ == "__main__":
    _self_test()
