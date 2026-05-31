"""
混合整数线性规划解解析模块
============================
基于种子项目:
  - 224_cplex_solution_read : CPLEX XML解文件的解析与数值清洗

核心数学模型:
  1. CPLEX解格式解析:
     - 单解: <variable name="x1" index="0" value="1.0"/>
     - 多解: <CPLEXSolution> 封装多个 <variable> 标签
     - 数值清洗: abs(round(x)) 消除 -0 和微小噪声

  2. MILP在轨迹规划中的应用:
     混合整数线性规划用于离散决策，例如:
     - 选择通过哪个避障走廊（二元变量）
     - 速度档位选择（整数变量）
     - 时间离散化步长选择

  3. 标准MILP形式:
     min  c^T x + d^T y
     s.t. A x + B y ≤ b
          x ∈ ℤ^n,  y ∈ ℝ^m
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import xml.etree.ElementTree as ET


class MILPSolution:
    r"""
    混合整数线性规划解的数据结构。
    """

    def __init__(self, objective: float, integer_vars: Dict[str, int],
                 continuous_vars: Dict[str, float]):
        self.objective = float(objective)
        self.integer_vars = dict(integer_vars)
        self.continuous_vars = dict(continuous_vars)

    def get_var(self, name: str, default=0.0):
        if name in self.integer_vars:
            return self.integer_vars[name]
        return self.continuous_vars.get(name, default)

    def __repr__(self):
        return f"MILPSolution(obj={self.objective:.4f}, n_int={len(self.integer_vars)}, n_cont={len(self.continuous_vars)})"


def cplex_solution_read(xml_string: str) -> List[MILPSolution]:
    r"""
    解析CPLEX风格的XML解字符串，提取所有解。
    兼容单解和多解格式。
    """
    solutions = []
    try:
        root = ET.fromstring(xml_string)
    except ET.ParseError as e:
        # 尝试包装在根元素中
        try:
            root = ET.fromstring(f"<root>{xml_string}</root>")
        except ET.ParseError:
            return []

    # 查找所有解节点
    sol_nodes = root.findall('.//CPLEXSolution')
    if not sol_nodes:
        sol_nodes = [root]

    seen_hashes = set()
    for sol_node in sol_nodes:
        obj_node = sol_node.find('header')
        objective = 0.0
        if obj_node is not None:
            obj_val = obj_node.get('objectiveValue')
            if obj_val is not None:
                objective = float(obj_val)

        int_vars = {}
        cont_vars = {}
        for var in sol_node.findall('.//variable'):
            name = var.get('name', '')
            val_str = var.get('value', '0')
            try:
                val = float(val_str)
            except ValueError:
                continue
            # 数值清洗：消除微小噪声和-0
            if abs(val) < 1e-8:
                val = 0.0
            # 判断整数变量（名字以'y'或'i'开头，或index在整数段）
            idx_str = var.get('index', '')
            # 启发式：若值接近整数，则视为整数
            if abs(val - round(val)) < 1e-5:
                int_vars[name] = int(round(val))
            else:
                cont_vars[name] = val

        sol = MILPSolution(objective, int_vars, cont_vars)
        h = hash((sol.objective, tuple(sorted(sol.integer_vars.items()))))
        if h not in seen_hashes:
            seen_hashes.add(h)
            solutions.append(sol)
    return solutions


def generate_example_cplex_xml() -> str:
    r"""
    生成一个示例CPLEX解XML，模拟机械臂避障走廊选择的MILP结果。
    决策变量:
      y_i ∈ {0,1}: 是否选择第i个避障走廊
      x_j ∈ ℝ:    第j个关节的连续速度
    """
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<CPLEXSolutions version="1.2">
  <CPLEXSolution version="1.2">
    <header problemName="trajectory_milp" objectiveValue="12.3456"/>
    <variables>
      <variable name="y_corridor_1" index="0" value="1"/>
      <variable name="y_corridor_2" index="1" value="0"/>
      <variable name="y_corridor_3" index="2" value="0.0000001"/>
      <variable name="x_vel_j1" index="3" value="0.523"/>
      <variable name="x_vel_j2" index="4" value="-0.314"/>
      <variable name="x_vel_j3" index="5" value="0.785"/>
      <variable name="x_vel_j4" index="6" value="-0.200"/>
      <variable name="x_vel_j5" index="7" value="0.150"/>
      <variable name="x_vel_j6" index="8" value="-0.100"/>
      <variable name="x_vel_j7" index="9" value="0.050"/>
    </variables>
  </CPLEXSolution>
</CPLEXSolutions>"""
    return xml


def parse_milp_trajectory_decision(xml_string: str) -> Dict[str, any]:
    r"""
    从CPLEX解中解析轨迹规划决策。
    返回字典:
      {
        'selected_corridors': [str],
        'joint_velocities': {str: float},
        'objective': float
      }
    """
    sols = cplex_solution_read(xml_string)
    if not sols:
        return {'selected_corridors': [], 'joint_velocities': {}, 'objective': 0.0}
    sol = sols[0]
    corridors = [k for k, v in sol.integer_vars.items() if v == 1 and 'corridor' in k]
    velocities = {k: v for k, v in sol.continuous_vars.items() if 'vel' in k}
    return {
        'selected_corridors': corridors,
        'joint_velocities': velocities,
        'objective': sol.objective
    }
