# -*- coding: utf-8 -*-
"""
heterostructure_topology.py — 异质结构拓扑描述
=================================================
核心科学问题: 描述多层异质结构的层状拓扑,
包括材料排列、界面、应变传递.

融合种子项目:
  - 1059_Glyphosate_crystallization: 分子拓扑与坐标生成
  - 754_mesh_display: 网格拓扑读取 (去掉可视化)
"""

import numpy as np


class HeterostructureTopology:
    """
    多层异质结拓扑描述.

    每层用 (材料, 厚度, 应变状态) 描述.
    界面用 (上层, 下层, 界面类型) 描述.
    """

    def __init__(self):
        self.layers = []
        self.interfaces = []
        self.substrate = None

    def add_layer(self, material, thickness, strain_state='pseudomorphic'):
        """
        添加一层.

        参数
        ----
        material : Material2D
            材料对象
        thickness : float
            层厚度 [m]
        strain_state : str
            应变状态 ('pseudomorphic', 'relaxed', 'partial')
        """
        self.layers.append({
            'material': material,
            'thickness': thickness,
            'strain_state': strain_state,
            'z_start': 0.0,
            'z_end': 0.0,
        })

    def set_substrate(self, material):
        """设置衬底材料."""
        self.substrate = material

    def compute_z_positions(self, z_start=0.0):
        """计算各层的 z 位置."""
        z = z_start
        for layer in self.layers:
            layer['z_start'] = z
            layer['z_end'] = z + layer['thickness']
            z = layer['z_end']
        return z

    def compute_interfaces(self):
        """计算界面位置和性质."""
        self.interfaces = []
        for i in range(len(self.layers) - 1):
            mat1 = self.layers[i]['material']
            mat2 = self.layers[i + 1]['material']
            z_int = self.layers[i]['z_end']

            # 晶格失配
            mismatch = (mat2.a_lattice - mat1.a_lattice) / mat1.a_lattice

            # 界面类型
            if abs(mismatch) < 0.01:
                itype = 'coherent'
            elif abs(mismatch) < 0.05:
                itype = 'semi-coherent'
            else:
                itype = 'incoherent'

            self.interfaces.append({
                'z': z_int,
                'material_above': mat1.name,
                'material_below': mat2.name,
                'mismatch': mismatch,
                'type': itype,
            })

        return self.interfaces

    def total_thickness(self):
        """总厚度."""
        if not self.layers:
            return 0.0
        return sum(l['thickness'] for l in self.layers)

    def critical_thickness(self, layer_idx):
        """
        Matthews-Blakeslee 临界厚度:
            h_c = (b/(2πf)) * (1-ν/4) * ln(h_c/b + 1)

        其中 b 是 Burgers 矢量, f 是失配度, ν 是泊松比.
        """
        if layer_idx >= len(self.layers):
            return float('inf')

        layer = self.layers[layer_idx]
        mat = layer['material']

        # Burgers 矢量 ≈ 晶格常数
        b = mat.a_lattice * 1e-10  # [m]

        # 失配度
        if self.substrate is not None:
            f = abs(mat.a_lattice - self.substrate.a_lattice) / self.substrate.a_lattice
        elif layer_idx > 0:
            f = abs(mat.a_lattice -
                    self.layers[layer_idx - 1]['material'].a_lattice) / mat.a_lattice
        else:
            f = 0.0

        if f < 1e-6:
            return float('inf')

        # 迭代求解
        nu = 0.3  # 泊松比
        h_c = b / f  # 初始估计
        for _ in range(20):
            h_c_new = (b / (2 * np.pi * f)) * (1 - nu / 4) * np.log(h_c / b + 1)
            if abs(h_c_new - h_c) < 1e-15:
                break
            h_c = h_c_new

        return h_c

    def strain_distribution(self, z_points):
        """
        计算应变沿 z 的分布.

        对于赝晶生长:
            eps_xx = (a_sub - a_layer) / a_layer
        对于弛豫:
            eps_xx = 0
        """
        eps_xx = np.zeros_like(z_points)

        for layer in self.layers:
            z_s = layer['z_start']
            z_e = layer['z_end']
            mat = layer['material']
            strain_state = layer['strain_state']

            mask = (z_points >= z_s) & (z_points <= z_e)

            if strain_state == 'pseudomorphic' and self.substrate:
                eps = (self.substrate.a_lattice - mat.a_lattice) / mat.a_lattice
            elif strain_state == 'partial':
                eps = 0.5 * (self.substrate.a_lattice - mat.a_lattice) / mat.a_lattice
            else:
                eps = 0.0

            eps_xx[mask] = eps

        return eps_xx

    def to_dict(self):
        """转换为字典 (序列化)."""
        return {
            'n_layers': len(self.layers),
            'total_thickness': self.total_thickness(),
            'layers': [{'material': l['material'].name,
                        'thickness': l['thickness'],
                        'strain_state': l['strain_state']}
                       for l in self.layers],
            'interfaces': [{'z': i['z'],
                            'mismatch': i['mismatch'],
                            'type': i['type']}
                           for i in self.interfaces],
        }


def create_typical_heterostructure(het_type='MoS2_WSe2'):
    """
    创建典型异质结结构.
    """
    from material_parameters import MOS2, WSE2, HBN, MOSE2, WS2, GASE

    topo = HeterostructureTopology()

    if het_type == 'MoS2_WSe2':
        topo.set_substrate(HBN)
        topo.add_layer(HBN, 10e-10, 'relaxed')       # hBN 衬底层
        topo.add_layer(MOS2, 6.15e-10, 'pseudomorphic')  # MoS2
        topo.add_layer(WSE2, 6.30e-10, 'pseudomorphic')  # WSe2
        topo.add_layer(HBN, 10e-10, 'relaxed')       # hBN 覆盖层

    elif het_type == 'MoS2_MoSe2':
        topo.set_substrate(HBN)
        topo.add_layer(HBN, 10e-10, 'relaxed')
        topo.add_layer(MOS2, 6.15e-10, 'pseudomorphic')
        topo.add_layer(MOSE2, 6.35e-10, 'pseudomorphic')
        topo.add_layer(HBN, 10e-10, 'relaxed')

    elif het_type == 'WSe2_WS2':
        topo.set_substrate(HBN)
        topo.add_layer(HBN, 10e-10, 'relaxed')
        topo.add_layer(WSE2, 6.30e-10, 'pseudomorphic')
        topo.add_layer(WS2, 6.18e-10, 'pseudomorphic')
        topo.add_layer(HBN, 10e-10, 'relaxed')

    else:
        # 默认
        topo.set_substrate(HBN)
        topo.add_layer(HBN, 10e-10, 'relaxed')
        topo.add_layer(MOS2, 6.15e-10, 'pseudomorphic')
        topo.add_layer(WSE2, 6.30e-10, 'pseudomorphic')
        topo.add_layer(HBN, 10e-10, 'relaxed')

    topo.compute_z_positions()
    topo.compute_interfaces()

    return topo
