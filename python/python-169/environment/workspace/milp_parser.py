
import numpy as np
from typing import Dict, List, Tuple, Optional
import xml.etree.ElementTree as ET


class MILPSolution:

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
    solutions = []
    try:
        root = ET.fromstring(xml_string)
    except ET.ParseError as e:

        try:
            root = ET.fromstring(f"<root>{xml_string}</root>")
        except ET.ParseError:
            return []


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

            if abs(val) < 1e-8:
                val = 0.0

            idx_str = var.get('index', '')

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
