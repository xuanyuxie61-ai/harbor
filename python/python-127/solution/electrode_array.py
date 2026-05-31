"""
electrode_array.py
==================
人工耳蜗电极阵列配置模块

科学背景:
  现代人工耳蜗通常包含 12~22 个铂铱合金电极，
  以线性或perimodiolar方式植入鼓阶(scala tympani)。
  每个电极可独立施加双相脉冲电流刺激。

电极参数:
  - 电极半径: a_e ≈ 0.15 ~ 0.3 mm
  - 电极间距: d_e ≈ 0.7 ~ 1.1 mm
  - 刺激电流: I_e ≈ 10 ~ 2000 μA (临床常用 100~500 μA)
"""

import numpy as np


class ElectrodeArray:
    """
    人工耳蜗电极阵列模型。
    """

    def __init__(self, n_electrodes=12, electrode_radius=0.2,
                 spacing=1.0, insertion_depth_mm=25.0,
                 offset_from_modiolus=0.8):
        """
        Parameters
        ----------
        n_electrodes : int
            电极数量
        electrode_radius : float
            电极有效半径 (mm)
        spacing : float
            相邻电极中心间距 (mm)
        insertion_depth_mm : float
            插入深度，从蜗底算起 (mm)
        offset_from_modiolus : float
            电极阵列中心到蜗轴的距离 (mm)
        """
        if n_electrodes < 1:
            raise ValueError("电极数量至少为 1")
        if spacing <= 0:
            raise ValueError("电极间距必须为正")
        if electrode_radius <= 0:
            raise ValueError("电极半径必须为正")

        self.n_electrodes = int(n_electrodes)
        self.electrode_radius = float(electrode_radius)
        self.spacing = float(spacing)
        self.insertion_depth = float(insertion_depth_mm)
        self.offset = float(offset_from_modiolus)
        self.positions = None
        self.currents = None

    def place_along_modiolar_axis(self, geometry):
        """
        沿蜗轴螺旋放置电极。

        Parameters
        ----------
        geometry : CochleaGeometry
            耳蜗几何对象

        Returns
        -------
        positions : ndarray, shape (n_electrodes, 2)
            电极二维坐标
        """
        # 从蜗底开始，沿中心线累积弧长定位电极
        cl = geometry._centerline['points']
        tangents = geometry._centerline['tangent']
        normals = geometry._centerline['normal']

        # 计算累积弧长
        diffs = np.diff(cl, axis=0)
        seg_lengths = np.sqrt(np.sum(diffs**2, axis=1))
        arc_lengths = np.concatenate(([0.0], np.cumsum(seg_lengths)))

        # 电极从蜗底开始沿弧长均匀分布
        electrode_arc_positions = np.linspace(
            0.0, min(self.insertion_depth, arc_lengths[-1]), self.n_electrodes
        )

        positions = np.empty((self.n_electrodes, 2))
        for i, s in enumerate(electrode_arc_positions):
            idx = np.searchsorted(arc_lengths, s)
            idx = min(idx, len(cl) - 1)
            # 电极位于蜗轴外侧 offset 处
            positions[i] = cl[idx] + self.offset * normals[idx]

        self.positions = positions
        return positions

    def set_currents(self, currents_uA):
        """
        设置各电极刺激电流 (μA)。

        Parameters
        ----------
        currents_uA : ndarray, shape (n_electrodes,)
            电流值，单位 μA
        """
        currents_uA = np.asarray(currents_uA, dtype=float)
        if currents_uA.shape != (self.n_electrodes,):
            raise ValueError(
                f"currents_uA shape must be ({self.n_electrodes},), "
                f"got {currents_uA.shape}"
            )
        if np.any(np.abs(currents_uA) > 5000.0):
            raise ValueError("电流值超过安全上限 5000 μA")
        self.currents = currents_uA * 1e-6  # 转换为 A

    def monopolar_stimulus(self, active_electrode_idx, amplitude_uA):
        """
        单极刺激模式: 一个电极作为 active，体外接地作为 return。

        Parameters
        ----------
        active_electrode_idx : int
            活动电极索引
        amplitude_uA : float
            刺激幅度 (μA)
        """
        if not (0 <= active_electrode_idx < self.n_electrodes):
            raise ValueError("电极索引越界")
        currents = np.zeros(self.n_electrodes)
        currents[active_electrode_idx] = amplitude_uA
        self.set_currents(currents)

    def tripolar_stimulus(self, center_idx, amplitude_uA, fraction=0.5):
        """
        三极刺激模式: 中心电极发放阳极电流，相邻两电极作为回流。

        Parameters
        ----------
        center_idx : int
            中心电极索引
        amplitude_uA : float
            中心电极刺激幅度 (μA)
        fraction : float
            相邻电极分流比例
        """
        if not (0 <= center_idx < self.n_electrodes):
            raise ValueError("电极索引越界")
        if not (0.0 <= fraction <= 1.0):
            raise ValueError("fraction 必须在 [0, 1] 之间")

        currents = np.zeros(self.n_electrodes)
        currents[center_idx] = amplitude_uA
        if center_idx > 0:
            currents[center_idx - 1] = -amplitude_uA * fraction / 2.0
        if center_idx + 1 < self.n_electrodes:
            currents[center_idx + 1] = -amplitude_uA * fraction / 2.0
        self.set_currents(currents)

    def get_source_terms(self, mesh_nodes):
        """
        将电极电流转换为有限元网格上的源项。

        使用高斯近似点源:
            f(x) = I_e / (π * a_e^2) * exp(-||x - x_e||^2 / a_e^2)

        Parameters
        ----------
        mesh_nodes : ndarray, shape (N, 2)
            网格节点坐标

        Returns
        -------
        source : ndarray, shape (N,)
            源项分布 (A/mm^2)
        """
        if self.positions is None:
            raise RuntimeError("必须先调用 place_along_modiolar_axis()")
        if self.currents is None:
            raise RuntimeError("必须先设置刺激电流")

        mesh_nodes = np.asarray(mesh_nodes, dtype=float)
        source = np.zeros(mesh_nodes.shape[0])
        sigma = self.electrode_radius  # 高斯源宽度

        for pos, I in zip(self.positions, self.currents):
            if abs(I) < 1e-15:
                continue
            dists_sq = np.sum((mesh_nodes - pos)**2, axis=1)
            # 高斯点源近似
            gaussian = np.exp(-dists_sq / (sigma**2)) / (np.pi * sigma**2)
            source += I * gaussian

        return source
