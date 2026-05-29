"""
reversal_detector.py — 地磁反转事件检测与统计分析

融合以下种子项目：
- 116_box_plot : 数据统计分箱与统计矩计算
- 1062_scip_solution_read : 读取优化解（此处用于读取事件标记）
- 351_fd_to_tec : 数据格式化输出

功能：
1. 从偶极矩时间序列中自动检测极性反转事件
2. 统计反转间隔时间、反转持续时间、反转速率
3. 计算古地磁学的统计量（VGP 离散度、虚拟偶极矩 VDM）
4. 分箱统计（box-counting）用于分析反转路径的分形特征

核心数学模型：
-------------
极性反转判据：
  设 m_z(t) 为轴向偶极矩，当 sign(m_z(t)) ≠ sign(m_z(t-Δt)) 时，
  记为一次反转事件。

虚拟地磁极 (VGP) 纬度：
  λ_VGP = arctan[0.5 tan(I)]，其中 I 为磁倾角
  tan(I) = 2 tan(λ)  (轴向偶极近似)

虚拟偶极矩 (VDM)：
  VDM = (4π/μ₀) · B_eq · r_cmb³ / sqrt(1 + 3 cos²θ_m)
  其中 B_eq 为赤道磁场强度，θ_m 为磁余纬度。

反转路径分形维数（Box-counting）：
  对反转路径 {θ(t), φ(t)} 在球面上用边长为 ε 的方格覆盖，
  计数 N(ε)，则分形维数 D_f = -lim_{ε→0} log N(ε) / log ε。
"""

import numpy as np
from utils import PHYSICAL_CONSTANTS


class ReversalDetector:
    """
    地磁反转事件检测器。
    """

    def __init__(self, dipole_moment, times, threshold=0.05):
        """
        参数：
          dipole_moment : (n_times,) 轴向偶极矩时间序列
          times         : (n_times,) 对应时间
          threshold     : 检测阈值（偶极矩绝对值低于 threshold 视为过渡期）
        """
        self.mz = np.array(dipole_moment)
        self.times = np.array(times)
        self.threshold = threshold
        self.n_times = len(times)

    def detect_reversals(self):
        """
        检测所有反转事件。

        返回：
          events : 列表，每个元素为字典 {
              'time': 反转时刻,
              'duration': 过渡期持续时间,
              'polarity_before': 反转前极性 (+1/-1),
              'polarity_after': 反转后极性,
              'amplitude_drop': 偶极矩下降幅度
          }
        """
        events = []
        sign_mz = np.sign(self.mz)
        # 处理零点附近的振荡
        in_transition = False
        transition_start = 0

        for i in range(1, self.n_times):
            if abs(self.mz[i]) < self.threshold and not in_transition:
                in_transition = True
                transition_start = i

            if in_transition:
                # 检查是否恢复为明确极性
                if abs(self.mz[i]) >= self.threshold:
                    # 比较反转前后的符号
                    idx_before = max(0, transition_start - 1)
                    idx_after = min(self.n_times - 1, i)

                    if sign_mz[idx_before] * sign_mz[idx_after] < 0:
                        event = {
                            'time': self.times[i],
                            'duration': self.times[i] - self.times[transition_start],
                            'polarity_before': int(sign_mz[idx_before]),
                            'polarity_after': int(sign_mz[idx_after]),
                            'amplitude_drop': abs(self.mz[idx_before]) - abs(self.mz[i]),
                        }
                        events.append(event)

                    in_transition = False

        return events

    def compute_chron_statistics(self):
        """
        计算地磁极性期 (Chron) 的统计特性。

        返回：
          mean_chron_length : 平均极性期长度
          std_chron_length  : 标准差
          reversal_rate     : 反转率（每单位时间的反转次数）
        """
        events = self.detect_reversals()
        if len(events) < 2:
            return 0.0, 0.0, 0.0

        chron_lengths = []
        for i in range(1, len(events)):
            length = events[i]['time'] - events[i - 1]['time']
            if length > 0:
                chron_lengths.append(length)

        if len(chron_lengths) == 0:
            return 0.0, 0.0, 0.0

        mean_len = np.mean(chron_lengths)
        std_len = np.std(chron_lengths)
        total_time = self.times[-1] - self.times[0]
        rate = len(events) / (total_time + 1e-30)

        return mean_len, std_len, rate

    def compute_box_counting_dimension(self, theta_path, phi_path, n_boxes_range=None):
        """
        计算反转路径的盒计数维数（源自 116_box_plot 的分箱思想）。

        参数：
          theta_path : (N,) 反转路径的极角序列
          phi_path   : (N,) 反转路径的方位角序列
          n_boxes_range : 盒数量范围列表

        返回：
          D_f : 估计的分形维数
        """
        if n_boxes_range is None:
            n_boxes_range = [4, 8, 16, 32, 64]

        counts = []
        inv_eps = []

        for n_box in n_boxes_range:
            # 将球面划分为 n_box × n_box 的经纬网格
            d_theta = np.pi / n_box
            d_phi = 2 * np.pi / n_box

            occupied = set()
            for t, p in zip(theta_path, phi_path):
                i_theta = int(t / d_theta)
                i_phi = int(p / d_phi)
                occupied.add((i_theta, i_phi))

            counts.append(len(occupied))
            inv_eps.append(1.0 / d_theta)

        # 线性回归 log(N) ~ D_f · log(1/ε)
        log_counts = np.log(np.array(counts) + 1e-15)
        log_inv_eps = np.log(np.array(inv_eps))

        # 最小二乘拟合
        A = np.vstack([log_inv_eps, np.ones(len(log_inv_eps))]).T
        D_f, _ = np.linalg.lstsq(A, log_counts, rcond=None)[0]

        return max(0.0, min(2.0, D_f))

    def compute_vdm_series(self, Br_equator, r_cmb=1.0):
        """
        计算虚拟偶极矩 (VDM) 时间序列。

        公式：
          VDM = (4π/μ₀) · Br_eq · r_cmb³
        """
        mu0 = PHYSICAL_CONSTANTS["mu_0"]
        vdm = (4.0 * np.pi / mu0) * np.array(Br_equator) * (r_cmb ** 3)
        return vdm

    def compute_reversal_speed(self, theta_path, phi_path, times_path):
        """
        计算反转路径的角速度（度/单位时间）。
        """
        dtheta = np.diff(theta_path)
        dphi = np.diff(phi_path)
        dt = np.diff(times_path)
        dt = np.where(dt < 1e-15, 1e-15, dt)

        # 球面距离
        ds = np.sqrt(dtheta ** 2 + (np.sin(theta_path[:-1]) * dphi) ** 2)
        speed = ds / dt
        return np.degrees(np.mean(speed))

    def export_reversal_data(self, filename, events):
        """
        将反转事件数据导出为结构化文本（源自 351_fd_to_tec 的数据格式化思想）。
        """
        with open(filename, 'w') as f:
            f.write("# Geomagnetic Reversal Events\n")
            f.write("# Format: Time | Duration | Polarity_Before | Polarity_After | Amplitude_Drop\n")
            for ev in events:
                f.write(f"{ev['time']:.6e} {ev['duration']:.6e} "
                        f"{ev['polarity_before']} {ev['polarity_after']} "
                        f"{ev['amplitude_drop']:.6e}\n")

    def read_reversal_data(self, filename):
        """
        读取反转事件数据（源自 1062_scip_solution_read 的解析思想）。
        """
        events = []
        with open(filename, 'r') as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith('#'):
                    continue
                parts = stripped.split()
                if len(parts) >= 5:
                    events.append({
                        'time': float(parts[0]),
                        'duration': float(parts[1]),
                        'polarity_before': int(parts[2]),
                        'polarity_after': int(parts[3]),
                        'amplitude_drop': float(parts[4]),
                    })
        return events


def statistical_moments(data, max_moment=4):
    """
    计算数据的前四阶统计矩（均值、方差、偏度、峰度）。
    源自 116_box_plot 的统计分箱思想。
    """
    data = np.array(data)
    mean = np.mean(data)
    var = np.var(data)
    std = np.sqrt(var + 1e-30)

    moments = {'mean': mean, 'variance': var, 'std': std}

    if max_moment >= 3:
        skewness = np.mean(((data - mean) / std) ** 3)
        moments['skewness'] = skewness

    if max_moment >= 4:
        kurtosis = np.mean(((data - mean) / std) ** 4) - 3.0
        moments['kurtosis'] = kurtosis

    return moments
