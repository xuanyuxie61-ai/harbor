
import xml.etree.ElementTree as ET
import re
from typing import Any, Dict, Union


def xml2dict(element: ET.Element) -> Dict[str, Any]:
    def sanitize(name: str) -> str:
        name = name.replace('-', '_dash_')
        name = name.replace(':', '_colon_')
        name = name.replace('.', '_dot_')
        return name

    node_dict: Dict[str, Any] = {}


    if element.attrib:
        node_dict['Attributes'] = {sanitize(k): v for k, v in element.attrib.items()}


    text = element.text
    if text is not None:
        text = text.strip()
        if text:
            node_dict['Text'] = text


    for child in element:
        child_name = sanitize(child.tag)
        child_data = xml2dict(child)

        if child_name in node_dict:

            if not isinstance(node_dict[child_name], list):
                node_dict[child_name] = [node_dict[child_name]]
            node_dict[child_name].append(child_data)
        else:
            node_dict[child_name] = child_data

    return node_dict


def parse_robot_config(xml_string: str) -> Dict[str, Any]:
    root = ET.fromstring(xml_string)
    return xml2dict(root)


def extract_link_params(config: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
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
