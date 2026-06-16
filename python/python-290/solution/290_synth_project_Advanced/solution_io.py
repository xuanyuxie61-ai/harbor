"""
solution_io.py - 模拟数据 I/O 与特征值结果解析模块

本模块处理模拟数据的输入输出，包括:
  - 模拟结果的保存和加载 (JSON 格式)
  - 特征值解文件的解析
  - 网格数据的读写

核心算法融合了以下种子项目:
  - cplex_solution_read (224): XML/结构化数据解析
  - gmsh_io (474): 网格数据格式处理
  - triangulation_display (1336): 数据文件读取工具

作者: DA 博士级合成项目 PROJECT_290
"""

import json
import os
import numpy as np


class SimulationResultWriter:
    """
    模拟结果写入器。

    将模拟的关键结果保存为结构化 JSON 文件，
    便于后续分析和可复现性验证。
    """

    def __init__(self, output_dir):
        """
        参数:
          output_dir: str, 输出目录路径
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_eigenvalue_result(self, eigen_result, filename='eigenvalue_result.json'):
        """
        保存本征值分析结果。

        改编自 cplex_solution_read (224) 的结构化数据保存思想。

        参数:
          eigen_result: dict, 本征值分析结果
          filename: str, 输出文件名
        """
        filepath = os.path.join(self.output_dir, filename)

        serializable = {
            'max_growth_rate': float(eigen_result['max_growth_rate']),
            'is_stable': bool(eigen_result['is_stable']),
            'n_unstable': int(eigen_result['n_unstable']),
            'dominant_mode_idx': int(eigen_result['dominant_mode_idx']),
            'n_modes': len(eigen_result['eigenvalues']),
        }

        # 保存前 N 个本征值
        n_save = min(20, len(eigen_result['eigenvalues']))
        serializable['top_eigenvalues'] = []
        for i in range(n_save):
            ev = eigen_result['eigenvalues'][i]
            serializable['top_eigenvalues'].append({
                'index': i,
                'real': float(ev.real),
                'imag': float(ev.imag),
                'growth_rate': float(eigen_result['growth_rates'][i]),
                'frequency_hz': float(eigen_result['frequencies_hz'][i]),
            })

        with open(filepath, 'w') as f:
            json.dump(serializable, f, indent=2)

        return filepath

    def save_simulation_history(self, history, filename='simulation_history.json'):
        """
        保存模拟历史数据。

        参数:
          history: dict, 包含 energy, time 等序列
          filename: str
        """
        filepath = os.path.join(self.output_dir, filename)

        serializable = {}
        for key, values in history.items():
            if isinstance(values, (list, np.ndarray)):
                serializable[key] = [float(v) for v in values]

        with open(filepath, 'w') as f:
            json.dump(serializable, f, indent=2)

        return filepath

    def save_stability_result(self, stability_result, filename='stability_result.json'):
        """
        保存稳定性分析结果。
        """
        filepath = os.path.join(self.output_dir, filename)

        serializable = {}
        for key, value in stability_result.items():
            if isinstance(value, (np.floating, float)):
                serializable[key] = float(value)
            elif isinstance(value, (np.integer, int)):
                serializable[key] = int(value)
            elif isinstance(value, (np.bool_, bool)):
                serializable[key] = bool(value)

        with open(filepath, 'w') as f:
            json.dump(serializable, f, indent=2)

        return filepath

    def save_plasma_parameters(self, params, filename='plasma_params.json'):
        """
        保存等离子体参数。
        """
        filepath = os.path.join(self.output_dir, filename)

        param_dict = {}
        for attr in dir(params):
            if not attr.startswith('_') and not callable(getattr(params, attr)):
                val = getattr(params, attr)
                if isinstance(val, (int, float, np.floating, np.integer)):
                    param_dict[attr] = float(val)

        with open(filepath, 'w') as f:
            json.dump(param_dict, f, indent=2)

        return filepath


class SolutionFileReader:
    """
    特征值/优化解文件读取器。

    改编自 cplex_solution_read (224) 的解文件解析思想。
    支持解析 JSON 格式和简单文本格式的结果文件。
    """

    def __init__(self, input_dir):
        self.input_dir = input_dir

    def read_json_solution(self, filename):
        """
        读取 JSON 格式的解文件。

        参数:
          filename: str, 文件名

        返回:
          data: dict, 解析后的数据
        """
        filepath = os.path.join(self.input_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"文件不存在: {filepath}")

        with open(filepath, 'r') as f:
            data = json.load(f)

        return data

    def read_eigenvalue_file(self, filename):
        """
        读取特征值结果文件并重构复数本征值。

        参数:
          filename: str

        返回:
          eigenvalues: ndarray, 复数本征值数组
        """
        data = self.read_json_solution(filename)

        eigenvalues = []
        if 'top_eigenvalues' in data:
            for ev_data in data['top_eigenvalues']:
                ev = complex(ev_data['real'], ev_data['imag'])
                eigenvalues.append(ev)

        return np.array(eigenvalues)

    def read_mesh_file(self, filename):
        """
        读取网格数据文件。

        改编自 gmsh_io (474) 和 triangulation_display (1336)
        的网格数据读取方法。

        支持简单的文本格式:
          第1行: node_num element_num
          后续行: node_id x y z
          然后: elem_id type n1 n2 n3 ...

        参数:
          filename: str

        返回:
          nodes: ndarray, 节点坐标
          elements: ndarray, 单元连接
        """
        filepath = os.path.join(self.input_dir, filename)
        if not os.path.exists(filepath):
            return None, None

        with open(filepath, 'r') as f:
            lines = f.readlines()

        if len(lines) < 2:
            return None, None

        header = lines[0].strip().split()
        n_nodes = int(header[0])
        n_elements = int(header[1])

        nodes = np.zeros((n_nodes, 3))
        for i in range(n_nodes):
            parts = lines[i + 1].strip().split()
            nodes[i, 0] = float(parts[1])
            nodes[i, 1] = float(parts[2])
            nodes[i, 2] = float(parts[3]) if len(parts) > 3 else 0.0

        elements = []
        for i in range(n_elements):
            line_idx = n_nodes + 1 + i
            if line_idx < len(lines):
                parts = lines[line_idx].strip().split()
                elem_nodes = [int(p) for p in parts[2:]]
                elements.append(elem_nodes)

        return nodes, elements


def format_result_table(headers, rows, precision=6):
    """
    格式化结果表格为文本。

    参数:
      headers: list of str, 列标题
      rows: list of list, 数据行
      precision: int, 浮点精度

    返回:
      text: str, 格式化后的表格
    """
    col_widths = [len(h) for h in headers]

    str_rows = []
    for row in rows:
        str_row = []
        for i, val in enumerate(row):
            if isinstance(val, float):
                s = f"{val:.{precision}e}"
            elif isinstance(val, complex):
                s = f"({val.real:.{precision}e}+{val.imag:.{precision}e}j)"
            else:
                s = str(val)
            str_row.append(s)
            col_widths[i] = max(col_widths[i], len(s))
        str_rows.append(str_row)

    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"

    lines = [sep]
    header_line = "|"
    for i, h in enumerate(headers):
        header_line += f" {h:^{col_widths[i]}} |"
    lines.append(header_line)
    lines.append(sep)

    for str_row in str_rows:
        row_line = "|"
        for i, s in enumerate(str_row):
            row_line += f" {s:>{col_widths[i]}} |"
        lines.append(row_line)

    lines.append(sep)
    return "\n".join(lines)
