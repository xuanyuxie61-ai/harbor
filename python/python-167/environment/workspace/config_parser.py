"""
config_parser.py
机器人参数配置解析模块。
融入种子项目 1418_xml2struct（XML 递归解析器）。
将 XML 格式的机器人 URDF/SDF 参数文件解析为 Python dict 结构。
"""

import xml.etree.ElementTree as ET
import re
from typing import Any, Dict, Union


def xml2dict(element: ET.Element) -> Dict[str, Any]:
    """
    递归将 XML Element 解析为 dict，保留属性和子元素。
    思想来源于 xml2struct.m：将 XML 节点树映射为嵌套字典。

    命名空间中的特殊字符替换规则：
        '-' → '_dash_'
        ':' → '_colon_'
        '.' → '_dot_'
    """
    def sanitize(name: str) -> str:
        name = name.replace('-', '_dash_')
        name = name.replace(':', '_colon_')
        name = name.replace('.', '_dot_')
        return name

    node_dict: Dict[str, Any] = {}

    # 解析属性
    if element.attrib:
        node_dict['Attributes'] = {sanitize(k): v for k, v in element.attrib.items()}

    # 解析文本内容
    text = element.text
    if text is not None:
        text = text.strip()
        if text:
            node_dict['Text'] = text

    # 递归解析子节点
    for child in element:
        child_name = sanitize(child.tag)
        child_data = xml2dict(child)

        if child_name in node_dict:
            # 同名元素合并为列表
            if not isinstance(node_dict[child_name], list):
                node_dict[child_name] = [node_dict[child_name]]
            node_dict[child_name].append(child_data)
        else:
            node_dict[child_name] = child_data

    return node_dict


def parse_robot_config(xml_string: str) -> Dict[str, Any]:
    """
    解析机器人 XML 配置字符串，返回完整参数字典。
    期望结构包含 link、joint、inertial、collision 等 URDF 标准标签。
    """
    root = ET.fromstring(xml_string)
    return xml2dict(root)


def extract_link_params(config: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    从解析后的配置中提取各 link 的物理参数：质量、质心、惯性张量。

    科学公式：
    刚体惯性张量（关于质心坐标系）：
        I = [ Ixx  Ixy  Ixz
              Iyx  Iyy  Iyz
              Izx  Izy  Izz ]
    其中 Iij = ∫_V ρ(r) (r^2 δ_{ij} - x_i x_j) dV。
    """
    links = {}
    robot = config.get('robot', config)
    link_nodes = robot.get('link', [])
    if not isinstance(link_nodes, list):
        link_nodes = [link_nodes]

    for link in link_nodes:
        name = link.get('Attributes', {}).get('name', 'unknown')
        inertial = link.get('inertial', {})
        mass_node = inertial.get('mass', {})
        mass_text = mass_node.get('Text', '0.0')
        mass = float(mass_text) if mass_text else 0.0

        inertia_node = inertial.get('inertia', {})
        attr = inertia_node.get('Attributes', {})
        inertia = {
            'ixx': float(attr.get('ixx', 0.0)),
            'ixy': float(attr.get('ixy', 0.0)),
            'ixz': float(attr.get('ixz', 0.0)),
            'iyy': float(attr.get('iyy', 0.0)),
            'iyz': float(attr.get('iyz', 0.0)),
            'izz': float(attr.get('izz', 0.0)),
        }

        origin = inertial.get('origin', {})
        xyz_str = origin.get('Attributes', {}).get('xyz', '0 0 0')
        xyz = [float(v) for v in xyz_str.split()]

        links[name] = {
            'mass': mass,
            'com': np.array(xyz),
            'inertia': inertia,
        }
    return links


import numpy as np
