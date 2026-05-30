
import numpy as np
from utils import PHYSICAL_CONSTANTS


class ReversalDetector:

    def __init__(self, dipole_moment, times, threshold=0.05):
        self.mz = np.array(dipole_moment)
        self.times = np.array(times)
        self.threshold = threshold
        self.n_times = len(times)

    def detect_reversals(self):
        events = []
        sign_mz = np.sign(self.mz)

        in_transition = False
        transition_start = 0

        for i in range(1, self.n_times):
            if abs(self.mz[i]) < self.threshold and not in_transition:
                in_transition = True
                transition_start = i

            if in_transition:

                if abs(self.mz[i]) >= self.threshold:

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
        if n_boxes_range is None:
            n_boxes_range = [4, 8, 16, 32, 64]

        counts = []
        inv_eps = []

        for n_box in n_boxes_range:

            d_theta = np.pi / n_box
            d_phi = 2 * np.pi / n_box

            occupied = set()
            for t, p in zip(theta_path, phi_path):
                i_theta = int(t / d_theta)
                i_phi = int(p / d_phi)
                occupied.add((i_theta, i_phi))

            counts.append(len(occupied))
            inv_eps.append(1.0 / d_theta)


        log_counts = np.log(np.array(counts) + 1e-15)
        log_inv_eps = np.log(np.array(inv_eps))


        A = np.vstack([log_inv_eps, np.ones(len(log_inv_eps))]).T
        D_f, _ = np.linalg.lstsq(A, log_counts, rcond=None)[0]

        return max(0.0, min(2.0, D_f))

    def compute_vdm_series(self, Br_equator, r_cmb=1.0):
        mu0 = PHYSICAL_CONSTANTS["mu_0"]
        vdm = (4.0 * np.pi / mu0) * np.array(Br_equator) * (r_cmb ** 3)
        return vdm

    def compute_reversal_speed(self, theta_path, phi_path, times_path):
        dtheta = np.diff(theta_path)
        dphi = np.diff(phi_path)
        dt = np.diff(times_path)
        dt = np.where(dt < 1e-15, 1e-15, dt)


        ds = np.sqrt(dtheta ** 2 + (np.sin(theta_path[:-1]) * dphi) ** 2)
        speed = ds / dt
        return np.degrees(np.mean(speed))

    def export_reversal_data(self, filename, events):
        with open(filename, 'w') as f:
            f.write("# Geomagnetic Reversal Events\n")
            f.write("# Format: Time | Duration | Polarity_Before | Polarity_After | Amplitude_Drop\n")
            for ev in events:
                f.write(f"{ev['time']:.6e} {ev['duration']:.6e} "
                        f"{ev['polarity_before']} {ev['polarity_after']} "
                        f"{ev['amplitude_drop']:.6e}\n")

    def read_reversal_data(self, filename):
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
