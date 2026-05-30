
import numpy as np


class ElectrodeArray:

    def __init__(self, n_electrodes=12, electrode_radius=0.2,
                 spacing=1.0, insertion_depth_mm=25.0,
                 offset_from_modiolus=0.8):
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

        cl = geometry._centerline['points']
        tangents = geometry._centerline['tangent']
        normals = geometry._centerline['normal']


        diffs = np.diff(cl, axis=0)
        seg_lengths = np.sqrt(np.sum(diffs**2, axis=1))
        arc_lengths = np.concatenate(([0.0], np.cumsum(seg_lengths)))


        electrode_arc_positions = np.linspace(
            0.0, min(self.insertion_depth, arc_lengths[-1]), self.n_electrodes
        )

        positions = np.empty((self.n_electrodes, 2))
        for i, s in enumerate(electrode_arc_positions):
            idx = np.searchsorted(arc_lengths, s)
            idx = min(idx, len(cl) - 1)

            positions[i] = cl[idx] + self.offset * normals[idx]

        self.positions = positions
        return positions

    def set_currents(self, currents_uA):
        currents_uA = np.asarray(currents_uA, dtype=float)
        if currents_uA.shape != (self.n_electrodes,):
            raise ValueError(
                f"currents_uA shape must be ({self.n_electrodes},), "
                f"got {currents_uA.shape}"
            )
        if np.any(np.abs(currents_uA) > 5000.0):
            raise ValueError("电流值超过安全上限 5000 μA")
        self.currents = currents_uA * 1e-6

    def monopolar_stimulus(self, active_electrode_idx, amplitude_uA):
        if not (0 <= active_electrode_idx < self.n_electrodes):
            raise ValueError("电极索引越界")
        currents = np.zeros(self.n_electrodes)
        currents[active_electrode_idx] = amplitude_uA
        self.set_currents(currents)

    def tripolar_stimulus(self, center_idx, amplitude_uA, fraction=0.5):
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
        if self.positions is None:
            raise RuntimeError("必须先调用 place_along_modiolar_axis()")
        if self.currents is None:
            raise RuntimeError("必须先设置刺激电流")

        mesh_nodes = np.asarray(mesh_nodes, dtype=float)
        source = np.zeros(mesh_nodes.shape[0])
        sigma = self.electrode_radius

        for pos, I in zip(self.positions, self.currents):
            if abs(I) < 1e-15:
                continue
            dists_sq = np.sum((mesh_nodes - pos)**2, axis=1)

            gaussian = np.exp(-dists_sq / (sigma**2)) / (np.pi * sigma**2)
            source += I * gaussian

        return source
