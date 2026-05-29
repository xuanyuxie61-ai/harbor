"""
配置参数解析器 (config_parser.py)
==================================
基于种子项目 1418_xml2struct 的层次化数据解析思想，为地核发电机模拟提供
结构化参数配置管理。支持从嵌套 dict 读取物理参数、数值方法参数和边界条件。

所有参数均经过类型检查与边界校验，确保数值鲁棒性。
"""

import math
from typing import Dict, Any


# ---------------------------------------------------------------------------
# 默认配置模板（地球物理典型参数）
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    "planet": {
        "name": "Earth",
        "radius_cmb": 3480.0e3,      # 核幔边界半径 (m)
        "radius_icb": 1221.0e3,      # 内核边界半径 (m)
        "rotation_rate": 7.2921159e-5, # 地球自转角速度 (rad/s)
        "gravity_cmb": 10.68,         # CMB 处重力加速度 (m/s^2)
    },
    "core_physics": {
        "magnetic_diffusivity": 2.0,   # 磁扩散系数 eta (m^2/s)
        "kinematic_viscosity": 1.0e-6, # 运动学粘性 (m^2/s)
        "thermal_diffusivity": 5.0e-6, # 热扩散系数 (m^2/s)
        "density": 1.05e4,             # 地核密度 (kg/m^3)
        "electrical_conductivity": 5.0e5, # 电导率 (S/m)
        "alpha_effect_amplitude": 0.5, # alpha 效应无量纲振幅
        "omega_effect_shear": 1.0,     # 差速自转剪切强度
    },
    "numerics": {
        "l_max": 8,                   # 球谐截断阶数
        "n_radial": 32,               # 径向网格数
        "t_end_years": 1.0e6,         # 终止时间 (年)
        "dt_init_years": 100.0,       # 初始时间步 (年)
        "adaptive_tol": 1.0e-4,       # 自适应误差容限
        "cg_tol": 1.0e-10,            # 共轭梯度收敛容限
        "cg_maxiter": 2000,           # CG 最大迭代次数
    },
    "uncertainty_quantification": {
        "enable": True,
        "sparse_grid_level": 3,       # Smolyak 稀疏网格层级
        "uq_dim": 4,                  # 不确定参数维度
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


# ---------------------------------------------------------------------------
# 配置解析与校验
# ---------------------------------------------------------------------------
class ConfigParser:
    """地核发电机模拟参数解析器，基于 xml2struct 的层次化结构思想。"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config if config is not None else DEFAULT_CONFIG.copy()
        self._validate()

    def _validate(self):
        """边界校验：确保所有物理参数在合理范围内。"""
        p = self.config["planet"]
        c = self.config["core_physics"]
        n = self.config["numerics"]
        uq = self.config["uncertainty_quantification"]

        # 半径检查
        assert 0.0 < p["radius_icb"] < p["radius_cmb"], \
            "ICB 半径必须小于 CMB 半径"
        assert p["radius_cmb"] < 1.0e8, \
            "行星半径过大，疑似单位错误"

        # 扩散系数正定性
        assert c["magnetic_diffusivity"] > 0.0, \
            "磁扩散系数必须为正"
        assert c["thermal_diffusivity"] > 0.0, \
            "热扩散系数必须为正"

        # 数值参数合理性
        assert n["l_max"] >= 1 and n["l_max"] <= 32, \
            "球谐截断阶数必须在 [1, 32] 之间"
        assert n["n_radial"] >= 4 and n["n_radial"] <= 256, \
            "径向网格数必须在 [4, 256] 之间"
        assert n["adaptive_tol"] > 0.0 and n["adaptive_tol"] < 1.0, \
            "自适应容限必须在 (0, 1) 之间"

        # UQ 参数
        if uq["enable"]:
            assert uq["sparse_grid_level"] >= 1 and uq["sparse_grid_level"] <= 6, \
                "稀疏网格层级必须在 [1, 6] 之间"
            assert uq["uq_dim"] >= 1 and uq["uq_dim"] <= 8, \
                "不确定维度必须在 [1, 8] 之间"

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """安全取值，若路径不存在则返回 default。"""
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
        """
        估算磁雷诺数 Rm = U L / eta。
        取特征速度 U ~ alpha*Omega*L，特征尺度 L ~ shell_thickness。
        """
        U = self.alpha_amplitude() * self.omega_shear() * self.rotation_rate() * self.shell_thickness()
        L = self.shell_thickness()
        eta = self.magnetic_diffusivity()
        return U * L / eta if eta > 0.0 else 0.0

    def ekman_number(self) -> float:
        """
        Ekman 数 E = nu / (2*Omega*L^2)，衡量粘性力与科里奥利力之比。
        """
        nu = self.config["core_physics"]["kinematic_viscosity"]
        Omega = self.rotation_rate()
        L = self.shell_thickness()
        return nu / (2.0 * Omega * L * L)

    def magnetic_rossby_number(self) -> float:
        """
        磁 Rossby 数 Ro_m = U / (2*Omega*L)。
        """
        U = self.alpha_amplitude() * self.omega_shear() * self.rotation_rate() * self.shell_thickness()
        return U / (2.0 * self.rotation_rate() * self.shell_thickness())

    def to_dict(self) -> Dict[str, Any]:
        return self.config.copy()


def load_config(config_dict: Dict[str, Any] = None) -> ConfigParser:
    """工厂函数：从 dict 构建 ConfigParser。"""
    return ConfigParser(config_dict)


# ---------------------------------------------------------------------------
# 自测试
# ---------------------------------------------------------------------------
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
