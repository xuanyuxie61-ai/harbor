"""
config_io.py — 地核发电机模拟参数配置解析模块

原项目映射: 1418_xml2struct — XML结构解析与参数配置读取
改造思路: 将MATLAB的xml2struct改写为Python字典解析器，
          用于读取地核发电机模拟的物理参数、数值参数与输出控制参数。

本模块解析地核发电机模型的全部配置，包括：
  - 几何参数: 内核半径 r_i, 外核半径 r_o
  - 物理参数: 磁扩散率 η, 差分旋转强度 C_Ω, α效应强度 C_α, 平衡场强 B_eq
  - 数值参数: 径向网格数 N_r, 角度网格数 N_θ, 时间积分容差 tol, 最大时间 t_max
  - 输出参数: 采样间隔、SVD分析模态数、稀疏网格层级等
"""

import xml.etree.ElementTree as ET
from typing import Dict, Any


class ConfigIO:
    """
    配置参数读取与管理类。
    支持XML格式配置解析，并提供完整的默认值回退机制。
    """

    def __init__(self, xml_string: str = None):
        """
        初始化配置，从XML字符串或默认配置加载参数。
        """
        if xml_string is not None:
            self.params = self._parse_xml(xml_string)
        else:
            self.params = self._default_params()
        self._validate_params()

    def _default_params(self) -> Dict[str, Any]:
        """
        默认配置参数 —— 基于地球物理典型值的无量纲化参数。

        物理背景:
          地核外核厚度约 2260 km, 地球半径 R ≈ 6371 km,
          内核半径 r_i ≈ 0.35 R, 外核半径 r_o = R.
          磁扩散时间 τ_η = L²/η ≈ 2×10⁵ 年 (L = r_o - r_i).
        """
        return {
            "geometry": {
                "r_inner": 0.35,   # 内核边界 (核幔边界内半径)
                "r_outer": 1.00,   # 核幔边界 (外核外半径)
            },
            "physics": {
                "magnetic_diffusivity": 1.0,      # η, 磁扩散率 (无量纲, 基于 τ_η)
                "differential_rotation_amplitude": 150.0,  # C_Ω, 差分旋转强度
                "alpha_effect_amplitude": 8.0,    # C_α, α效应强度
                "equipartition_field_strength": 15.0,  # B_eq, 能量均分场强
                "alpha_quenching_exponent": 2.0,  # α淬灭指数
                "differential_rotation_profile": "solar_type",  # 差分旋转类型
            },
            "numerics": {
                "nr": 8,            # 径向网格点数 (粗网格快速演示)
                "ntheta": 12,       # 极角网格点数
                "time_tolerance": 1.0,   # 自适应RK局部截断误差容差 (大系统需放宽)
                "max_time": 12.0,   # 最大无量纲演化时间 (约 12 个磁扩散时间)
                "cfl_safety_factor": 0.45,  # CFL安全因子
                "min_dt": 1e-10,    # 最小时间步长
                "max_dt": 0.02,     # 最大时间步长
            },
            "output": {
                "snapshot_interval": 0.05,   # 场快照输出间隔
                "svd_rank": 6,               # SVD降维保留的模态数
                "sparse_grid_level": 3,      # 稀疏网格配置层级 (UQ)
                "enable_reversal_detection": True,
                "enable_parameter_scan": True,
            },
            "parallel": {
                "task_chunks": 4,    # 参数扫描任务分块数
            }
        }

    def _parse_xml(self, xml_string: str) -> Dict[str, Any]:
        """
        解析XML配置字符串，将嵌套XML结构转换为嵌套字典。
        改造自 xml2struct.m 的核心递归解析逻辑。
        """
        root = ET.fromstring(xml_string)
        return self._element_to_dict(root)

    def _element_to_dict(self, element: ET.Element) -> Any:
        """
        递归将XML元素转换为Python字典或标量值。
        处理嵌套元素、属性和文本内容。
        """
        result: Dict[str, Any] = {}

        # 解析属性
        if element.attrib:
            result["_attributes"] = dict(element.attrib)

        # 解析子元素
        children = list(element)
        if children:
            for child in children:
                child_data = self._element_to_dict(child)
                tag = child.tag.replace("-", "_").replace(":", "_")
                if tag in result:
                    # 同名元素转为列表
                    if not isinstance(result[tag], list):
                        result[tag] = [result[tag]]
                    result[tag].append(child_data)
                else:
                    result[tag] = child_data

        # 解析文本内容
        text = (element.text or "").strip()
        if text:
            # 尝试转换为数值类型
            result["_text"] = self._convert_scalar(text)

        # 简化：如果只有_text，直接返回标量
        if set(result.keys()) == {"_text"}:
            return result["_text"]

        return result if result else None

    @staticmethod
    def _convert_scalar(value: str) -> Any:
        """
        尝试将字符串转换为数值类型 (int/float)，失败则保持字符串。
        """
        try:
            if "." in value or "e" in value.lower():
                return float(value)
            return int(value)
        except ValueError:
            return value

    def _validate_params(self):
        """
        验证并修正配置参数的边界条件与数值鲁棒性。
        """
        geo = self.params["geometry"]
        phy = self.params["physics"]
        num = self.params["numerics"]

        # 几何参数约束: 0 < r_i < r_o
        if not (0.0 < geo["r_inner"] < geo["r_outer"]):
            geo["r_inner"] = 0.35
            geo["r_outer"] = 1.0

        # 物理参数非负性
        phy["magnetic_diffusivity"] = max(phy.get("magnetic_diffusivity", 1.0), 1e-10)
        phy["equipartition_field_strength"] = max(phy.get("equipartition_field_strength", 10.0), 1e-6)

        # 网格参数边界
        num["nr"] = max(4, min(num.get("nr", 28), 512))
        num["ntheta"] = max(4, min(num.get("ntheta", 56), 1024))
        num["time_tolerance"] = max(1e-12, min(num.get("time_tolerance", 1e-5), 1e-1))

    def get(self, section: str, key: str, default=None):
        """
        安全获取嵌套配置参数。
        """
        return self.params.get(section, {}).get(key, default)

    def dump(self) -> str:
        """
        将当前配置序列化为可读字符串。
        """
        lines = ["=" * 60, "地核发电机模拟参数配置", "=" * 60]
        for section, entries in self.params.items():
            lines.append(f"\n[{section}]")
            for k, v in entries.items():
                lines.append(f"  {k:40s} = {v}")
        lines.append("=" * 60)
        return "\n".join(lines)
