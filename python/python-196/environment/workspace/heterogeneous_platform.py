"""
heterogeneous_platform.py
异构计算平台建模模块

包含：
- 异构处理器（CPU/GPU/FPGA/TPU）的峰值性能与功耗建模
- 处理器拓扑的仿射变换表示（源自 hand 项目的几何变换思想）
- 内存带宽与通信延迟模型

科学背景：
现代HPC集群由多种处理器组成：
  - CPU: 低延迟、高单线程性能，适合不规则计算
  - GPU: 高吞吐、大规模并行，适合规则稠密计算
  - FPGA: 可重构、低功耗，适合定制流计算
  - TPU: 专用矩阵加速器，适合AI推理

处理器i的有效性能（考虑温度降频）：
    P_eff(i) = P_peak(i) * (1 - alpha_i * (T_i - T_ambient))

其中温度 T_i 由热阻网络模型给出：
    T_i = T_ambient + R_th(i) * P_diss(i)
"""

import numpy as np
from mesh_transform import rotation_matrix_2d, dilation_matrix_2d, affine_transform_2d


class Processor:
    """
    单个异构处理器模型。
    """
    def __init__(self, proc_id, proc_type, peak_gflops, memory_bw_gb_s,
                 power_idle_w, power_peak_w, thermal_resistance_k_w,
                 position_xy):
        self.proc_id = proc_id
        self.proc_type = proc_type  # 'CPU', 'GPU', 'FPGA', 'TPU'
        self.peak_gflops = float(peak_gflops)
        self.memory_bw_gb_s = float(memory_bw_gb_s)
        self.power_idle_w = float(power_idle_w)
        self.power_peak_w = float(power_peak_w)
        self.thermal_resistance_k_w = float(thermal_resistance_k_w)
        self.position_xy = np.array(position_xy, dtype=float)
        self.current_temp = 300.0  # K
        self.utilization = 0.0

    def effective_performance(self, ambient_temp=300.0, alpha_thermal=0.003):
        """
        考虑温度降频后的有效性能。

        P_eff = P_peak * max(0, 1 - alpha * (T - T_ambient))
        T = T_ambient + R_th * P_diss
        P_diss = P_idle + util * (P_peak - P_idle)
        """
        p_diss = self.power_idle_w + self.utilization * (self.power_peak_w - self.power_idle_w)
        self.current_temp = ambient_temp + self.thermal_resistance_k_w * p_diss
        factor = max(0.0, 1.0 - alpha_thermal * (self.current_temp - ambient_temp))
        return self.peak_gflops * factor

    def execution_time(self, workload_flops, compute_intensity):
        """
         Roofline模型估算执行时间:
            t_peak = workload / peak_flops
            t_mem  = workload / (memory_bw * intensity)
            t = max(t_peak, t_mem)
        其中 memory_bw [bytes/s] = memory_bw_gb_s * 1e9,
        intensity [FLOPs/byte] = compute_intensity,
        故 memory-bound 算力 = memory_bw [bytes/s] * intensity [FLOPs/byte] = [FLOPs/s].
        """
        peak_time = workload_flops / max(self.effective_performance() * 1e9, 1e-6)
        mem_bound_flops = self.memory_bw_gb_s * 1e9 * max(compute_intensity, 0.0)
        mem_time = workload_flops / max(mem_bound_flops, 1e-6)
        return max(peak_time, mem_time)


class HeterogeneousPlatform:
    """
    异构计算平台：包含多个处理器及其拓扑连接。
    """
    def __init__(self, ambient_temp=300.0):
        self.processors = []
        self.ambient_temp = float(ambient_temp)
        self.comm_latency_matrix = None  # 处理器间通信延迟矩阵
        self.topology_transform = np.eye(2)  # 拓扑映射的仿射变换
        self.topology_offset = np.zeros(2)

    def add_processor(self, proc):
        self.processors.append(proc)

    def build_default_platform(self):
        """
        构建一个默认的4处理器异构平台：
          2x CPU + 1x GPU + 1x FPGA
        """
        # CPU0
        self.add_processor(Processor(
            0, 'CPU', peak_gflops=500.0, memory_bw_gb_s=100.0,
            power_idle_w=50.0, power_peak_w=200.0,
            thermal_resistance_k_w=0.5, position_xy=[0.0, 0.0]
        ))
        # CPU1
        self.add_processor(Processor(
            1, 'CPU', peak_gflops=500.0, memory_bw_gb_s=100.0,
            power_idle_w=50.0, power_peak_w=200.0,
            thermal_resistance_k_w=0.5, position_xy=[1.0, 0.0]
        ))
        # GPU
        self.add_processor(Processor(
            2, 'GPU', peak_gflops=10000.0, memory_bw_gb_s=900.0,
            power_idle_w=30.0, power_peak_w=300.0,
            thermal_resistance_k_w=0.3, position_xy=[0.5, 1.0]
        ))
        # FPGA
        self.add_processor(Processor(
            3, 'FPGA', peak_gflops=2000.0, memory_bw_gb_s=50.0,
            power_idle_w=10.0, power_peak_w=75.0,
            thermal_resistance_k_w=0.8, position_xy=[0.5, -1.0]
        ))
        self._build_comm_matrix()

    def _build_comm_matrix(self):
        """
        基于欧氏距离构建通信延迟矩阵（简化模型）。
        latency_{ij} = base_latency + distance_{ij} / bandwidth
        """
        n = len(self.processors)
        self.comm_latency_matrix = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(
                    self.processors[i].position_xy - self.processors[j].position_xy
                )
                lat = 1e-6 + 1e-7 * dist  # 基础1us + 0.1us/m
                self.comm_latency_matrix[i, j] = lat
                self.comm_latency_matrix[j, i] = lat

    def apply_topology_transform(self, A=None, b=None):
        """
        对处理器拓扑施加仿射变换（源自 hand_linear 的几何变换思想）。
        用于模拟机柜布局旋转/缩放等场景。
        """
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
        """
        旋转处理器拓扑（源自 hand_rotation）。
        """
        A = rotation_matrix_2d(np.deg2rad(angle_deg))
        self.apply_topology_transform(A=A)

    def scale_topology(self, sx, sy):
        """
        缩放处理器拓扑（源自 hand_dilation）。
        """
        A = dilation_matrix_2d(sx, sy)
        self.apply_topology_transform(A=A)

    def get_total_power(self):
        """
        平台总功耗。
        """
        return sum(
            p.power_idle_w + p.utilization * (p.power_peak_w - p.power_idle_w)
            for p in self.processors
        )

    def reset_utilization(self):
        for p in self.processors:
            p.utilization = 0.0

    def snapshot_state(self):
        """
        返回平台状态字典。
        """
        return {
            'temps': [p.current_temp for p in self.processors],
            'effective_gflops': [p.effective_performance(self.ambient_temp) for p in self.processors],
            'total_power_w': self.get_total_power()
        }
