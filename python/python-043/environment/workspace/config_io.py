
import xml.etree.ElementTree as ET
from typing import Dict, Any


class ConfigIO:

    def __init__(self, xml_string: str = None):
        if xml_string is not None:
            self.params = self._parse_xml(xml_string)
        else:
            self.params = self._default_params()
        self._validate_params()

    def _default_params(self) -> Dict[str, Any]:
        return {
            "geometry": {
                "r_inner": 0.35,
                "r_outer": 1.00,
            },
            "physics": {
                "magnetic_diffusivity": 1.0,
                "differential_rotation_amplitude": 150.0,
                "alpha_effect_amplitude": 8.0,
                "equipartition_field_strength": 15.0,
                "alpha_quenching_exponent": 2.0,
                "differential_rotation_profile": "solar_type",
            },
            "numerics": {
                "nr": 8,
                "ntheta": 12,
                "time_tolerance": 1.0,
                "max_time": 12.0,
                "cfl_safety_factor": 0.45,
                "min_dt": 1e-10,
                "max_dt": 0.02,
            },
            "output": {
                "snapshot_interval": 0.05,
                "svd_rank": 6,
                "sparse_grid_level": 3,
                "enable_reversal_detection": True,
                "enable_parameter_scan": True,
            },
            "parallel": {
                "task_chunks": 4,
            }
        }

    def _parse_xml(self, xml_string: str) -> Dict[str, Any]:
        root = ET.fromstring(xml_string)
        return self._element_to_dict(root)

    def _element_to_dict(self, element: ET.Element) -> Any:
        result: Dict[str, Any] = {}


        if element.attrib:
            result["_attributes"] = dict(element.attrib)


        children = list(element)
        if children:
            for child in children:
                child_data = self._element_to_dict(child)
                tag = child.tag.replace("-", "_").replace(":", "_")
                if tag in result:

                    if not isinstance(result[tag], list):
                        result[tag] = [result[tag]]
                    result[tag].append(child_data)
                else:
                    result[tag] = child_data


        text = (element.text or "").strip()
        if text:

            result["_text"] = self._convert_scalar(text)


        if set(result.keys()) == {"_text"}:
            return result["_text"]

        return result if result else None

    @staticmethod
    def _convert_scalar(value: str) -> Any:
        try:
            if "." in value or "e" in value.lower():
                return float(value)
            return int(value)
        except ValueError:
            return value

    def _validate_params(self):
        geo = self.params["geometry"]
        phy = self.params["physics"]
        num = self.params["numerics"]


        if not (0.0 < geo["r_inner"] < geo["r_outer"]):
            geo["r_inner"] = 0.35
            geo["r_outer"] = 1.0


        phy["magnetic_diffusivity"] = max(phy.get("magnetic_diffusivity", 1.0), 1e-10)
        phy["equipartition_field_strength"] = max(phy.get("equipartition_field_strength", 10.0), 1e-6)


        num["nr"] = max(4, min(num.get("nr", 28), 512))
        num["ntheta"] = max(4, min(num.get("ntheta", 56), 1024))
        num["time_tolerance"] = max(1e-12, min(num.get("time_tolerance", 1e-5), 1e-1))

    def get(self, section: str, key: str, default=None):
        return self.params.get(section, {}).get(key, default)

    def dump(self) -> str:
        lines = ["=" * 60, "地核发电机模拟参数配置", "=" * 60]
        for section, entries in self.params.items():
            lines.append(f"\n[{section}]")
            for k, v in entries.items():
                lines.append(f"  {k:40s} = {v}")
        lines.append("=" * 60)
        return "\n".join(lines)
