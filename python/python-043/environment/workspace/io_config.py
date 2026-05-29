"""
io_config.py — 配置解析与数据输入输出

融合以下种子项目：
- 1418_xml2struct : XML 配置解析
- 1062_scip_solution_read : 解文件读取（参数解）
- 351_fd_to_tec : 有限差分数据格式化输出
- 1310_triangle_io : Triangle 网格数据读写

功能：
1. 读取 XML 格式的地核发电机模型配置
2. 读写数值模拟结果（节点值、单元值、时间序列）
3. 格式化输出为类 TECPLOT 的 ASCII 数据文件
4. 参数解的解析与验证
"""

import numpy as np
import xml.etree.ElementTree as ET


def xml2struct(file_path):
    """
    将 XML 文件转换为嵌套字典（源自 1418_xml2struct）。
    支持属性、文本内容和多层嵌套。
    """
    def elem_to_dict(elem):
        result = {}
        if elem.attrib:
            result['Attributes'] = dict(elem.attrib)
        if elem.text and elem.text.strip():
            result['Text'] = elem.text.strip()
        for child in elem:
            child_dict = elem_to_dict(child)
            if child.tag in result:
                if not isinstance(result[child.tag], list):
                    result[child.tag] = [result[child.tag]]
                result[child.tag].append(child_dict)
            else:
                result[child.tag] = child_dict
        return result

    tree = ET.parse(file_path)
    root = tree.getroot()
    return {root.tag: elem_to_dict(root)}


def struct2xml(data, file_path):
    """
    将嵌套字典写回 XML 文件。
    """
    def dict_to_elem(parent, tag, value):
        if isinstance(value, dict):
            child = ET.SubElement(parent, tag)
            for k, v in value.items():
                if k == 'Attributes':
                    for attr_key, attr_val in v.items():
                        child.set(attr_key, str(attr_val))
                elif k == 'Text':
                    child.text = str(v)
                else:
                    dict_to_elem(child, k, v)
        elif isinstance(value, list):
            for item in value:
                dict_to_elem(parent, tag, item)
        else:
            child = ET.SubElement(parent, tag)
            child.text = str(value)

    root_tag = list(data.keys())[0]
    root = ET.Element(root_tag)
    dict_to_elem(root, root_tag, data[root_tag])

    tree = ET.ElementTree(root)
    tree.write(file_path, encoding='utf-8', xml_declaration=True)


class DynamoConfig:
    """
    地核发电机模型配置管理器。
    """

    def __init__(self, config_dict=None):
        self.params = {
            'simulation': {
                't_end': 5.0,
                'dt_init': 0.001,
                'tol': 1e-6,
                'integrator': 'rk12',
            },
            'physics': {
                'eta': 0.02,
                'nu': 1e-4,
                'kappa': 5e-5,
                'alpha_g': 1e-3,
                'Omega': 1.0,
                'Ra': 1e6,
                'alpha_effect': 0.1,
            },
            'mesh': {
                'n_radial': 10,
                'n_theta': 12,
                'n_phi': 16,
                'r_icb': 0.35,
                'r_cmb': 1.0,
                'use_cvt': True,
            },
            'stochastic': {
                'enabled': True,
                'dim_random': 3,
                'sg_level': 3,
                'sigma_alpha': 0.05,
            },
            'output': {
                'save_interval': 0.1,
                'output_dir': './output',
            }
        }
        if config_dict is not None:
            self._merge_dict(self.params, config_dict)

    def _merge_dict(self, base, update):
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_dict(base[key], value)
            else:
                base[key] = value

    @classmethod
    def from_xml(cls, file_path):
        data = xml2struct(file_path)
        config_dict = data.get('DynamoConfig', {})
        return cls(config_dict)

    def to_xml(self, file_path):
        struct2xml({'DynamoConfig': self.params}, file_path)

    def get(self, section, key, default=None):
        return self.params.get(section, {}).get(key, default)


def read_scip_solution(filename, nx):
    """
    读取 SCIP 优化解文件（源自 1062_scip_solution_read）。
    此处用于读取优化后的边界条件参数。
    """
    x = np.zeros(nx, dtype=int)
    with open(filename, 'r') as f:
        lines = f.readlines()

    # 跳过前 2 行
    for line in lines[2:]:
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) >= 1 and parts[0].startswith('x'):
            try:
                idx = int(parts[0][1:])
                if 1 <= idx <= nx:
                    x[idx - 1] = 1
            except ValueError:
                continue

    return x


def write_tecplot_ascii(filename, nodes, values_dict, elements=None):
    """
    输出类 TECPLOT ASCII 数据文件（源自 351_fd_to_tec）。

    格式：
      TITLE = "..."
      VARIABLES = "X", "Y", "Z", "Var1", ...
      ZONE N=..., E=..., DATAPACKING=POINT, ZONETYPE=FETETRAHEDRON
      <节点数据>
      <单元连接>
    """
    n_nodes = len(nodes)
    var_names = ['X', 'Y', 'Z'] + list(values_dict.keys())
    n_vars = len(var_names)

    has_elements = elements is not None and elements.size > 0
    n_elements = len(elements) if has_elements else 0

    with open(filename, 'w') as f:
        f.write(f'TITLE = "Core Dynamo Simulation Output"\n')
        f.write('VARIABLES = ' + ', '.join([f'"{v}"' for v in var_names]) + '\n')

        zone_type = 'FETETRAHEDRON' if has_elements else 'FEBRICK'
        f.write(f'ZONE N={n_nodes}, E={n_elements}, DATAPACKING=POINT, '
                f'ZONETYPE={zone_type}\n')

        # 节点数据
        for i in range(n_nodes):
            line_vals = [nodes[i, 0], nodes[i, 1], nodes[i, 2]]
            for key in values_dict:
                val = values_dict[key]
                if val.ndim > 1 and val.shape[1] == 3:
                    line_vals.extend([val[i, 0], val[i, 1], val[i, 2]])
                else:
                    line_vals.append(val[i])
            f.write(' '.join([f'{v:.8e}' for v in line_vals]) + '\n')

        # 单元数据
        if has_elements:
            for elem in elements:
                f.write(' '.join([str(int(idx) + 1) for idx in elem]) + '\n')


def read_fd_data(filename, n_cols_expected=None):
    """
    读取有限差分格式数据文件（源自 351_fd_to_tec）。
    返回 (n_points, n_cols) 数组。
    """
    data = []
    with open(filename, 'r') as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            try:
                vals = [float(x) for x in stripped.split()]
                if n_cols_expected and len(vals) != n_cols_expected:
                    continue
                data.append(vals)
            except ValueError:
                continue

    return np.array(data)


def save_timeseries(filename, times, data_dict):
    """
    保存时间序列数据。
    """
    with open(filename, 'w') as f:
        f.write('# Time ' + ' '.join(data_dict.keys()) + '\n')
        for i, t in enumerate(times):
            line = f'{t:.8e}'
            for key in data_dict:
                val = data_dict[key]
                if i < len(val):
                    line += f' {val[i]:.8e}'
                else:
                    line += ' 0.0'
            f.write(line + '\n')


def load_timeseries(filename):
    """
    加载时间序列数据。
    """
    with open(filename, 'r') as f:
        header = f.readline().strip()
    names = header.replace('#', '').strip().split()
    data = np.loadtxt(filename)
    times = data[:, 0]
    result = {}
    for j, name in enumerate(names[1:], start=1):
        result[name] = data[:, j]
    return times, result
