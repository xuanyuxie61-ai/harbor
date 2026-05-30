
import numpy as np
from mesh_transform import rotation_matrix_2d, dilation_matrix_2d, affine_transform_2d


class Processor:
    def __init__(self, proc_id, proc_type, peak_gflops, memory_bw_gb_s,
                 power_idle_w, power_peak_w, thermal_resistance_k_w,
                 position_xy):
        self.proc_id = proc_id
        self.proc_type = proc_type
        self.peak_gflops = float(peak_gflops)
        self.memory_bw_gb_s = float(memory_bw_gb_s)
        self.power_idle_w = float(power_idle_w)
        self.power_peak_w = float(power_peak_w)
        self.thermal_resistance_k_w = float(thermal_resistance_k_w)
        self.position_xy = np.array(position_xy, dtype=float)
        self.current_temp = 300.0
        self.utilization = 0.0

    def effective_performance(self, ambient_temp=300.0, alpha_thermal=0.003):
        p_diss = self.power_idle_w + self.utilization * (self.power_peak_w - self.power_idle_w)
        self.current_temp = ambient_temp + self.thermal_resistance_k_w * p_diss
        factor = max(0.0, 1.0 - alpha_thermal * (self.current_temp - ambient_temp))
        return self.peak_gflops * factor

    def execution_time(self, workload_flops, compute_intensity):
        peak_time = workload_flops / max(self.effective_performance() * 1e9, 1e-6)
        mem_bound_flops = self.memory_bw_gb_s * 1e9 * max(compute_intensity, 0.0)
        mem_time = workload_flops / max(mem_bound_flops, 1e-6)
        return max(peak_time, mem_time)


class HeterogeneousPlatform:
    def __init__(self, ambient_temp=300.0):
        self.processors = []
        self.ambient_temp = float(ambient_temp)
        self.comm_latency_matrix = None
        self.topology_transform = np.eye(2)
        self.topology_offset = np.zeros(2)

    def add_processor(self, proc):
        self.processors.append(proc)

    def build_default_platform(self):

        self.add_processor(Processor(
            0, 'CPU', peak_gflops=500.0, memory_bw_gb_s=100.0,
            power_idle_w=50.0, power_peak_w=200.0,
            thermal_resistance_k_w=0.5, position_xy=[0.0, 0.0]
        ))

        self.add_processor(Processor(
            1, 'CPU', peak_gflops=500.0, memory_bw_gb_s=100.0,
            power_idle_w=50.0, power_peak_w=200.0,
            thermal_resistance_k_w=0.5, position_xy=[1.0, 0.0]
        ))

        self.add_processor(Processor(
            2, 'GPU', peak_gflops=10000.0, memory_bw_gb_s=900.0,
            power_idle_w=30.0, power_peak_w=300.0,
            thermal_resistance_k_w=0.3, position_xy=[0.5, 1.0]
        ))

        self.add_processor(Processor(
            3, 'FPGA', peak_gflops=2000.0, memory_bw_gb_s=50.0,
            power_idle_w=10.0, power_peak_w=75.0,
            thermal_resistance_k_w=0.8, position_xy=[0.5, -1.0]
        ))
        self._build_comm_matrix()

    def _build_comm_matrix(self):
        n = len(self.processors)
        self.comm_latency_matrix = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(
                    self.processors[i].position_xy - self.processors[j].position_xy
                )
                lat = 1e-6 + 1e-7 * dist
                self.comm_latency_matrix[i, j] = lat
                self.comm_latency_matrix[j, i] = lat

    def apply_topology_transform(self, A=None, b=None):
        n = len(self.processors)
        positions = np.column_stack([p.position_xy for p in self.processors])
        new_pos = affine_transform_2d(positions, A, b)
        for i in range(n):
            self.processors[i].position_xy = new_pos[:, i]
        self._build_comm_matrix()
        if A is not None:
            self.topology_transform = A @ self.topology_transform
        if b is not None:
            self.topology_offset = self.topology_offset + (b[:, 0] if b.ndim > 1 else b)

    def rotate_topology(self, angle_deg):
        A = rotation_matrix_2d(np.deg2rad(angle_deg))
        self.apply_topology_transform(A=A)

    def scale_topology(self, sx, sy):
        A = dilation_matrix_2d(sx, sy)
        self.apply_topology_transform(A=A)

    def get_total_power(self):
        return sum(
            p.power_idle_w + p.utilization * (p.power_peak_w - p.power_idle_w)
            for p in self.processors
        )

    def reset_utilization(self):
        for p in self.processors:
            p.utilization = 0.0

    def snapshot_state(self):
        return {
            'temps': [p.current_temp for p in self.processors],
            'effective_gflops': [p.effective_performance(self.ambient_temp) for p in self.processors],
            'total_power_w': self.get_total_power()
        }
