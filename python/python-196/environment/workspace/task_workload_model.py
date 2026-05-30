
import numpy as np
from utils import safe_log


def alnorm_cdf(x, upper=False):
    a1 = 5.75885480458
    a2 = 2.62433121679
    a3 = 5.92885724438
    b1 = -29.8213557807
    b2 = 48.6959930692
    c1 = -0.000000038052
    c2 = 0.000398064794
    c3 = -0.151679116635
    c4 = 4.8385912808
    c5 = 0.742380924027
    c6 = 3.99019417011
    con = 1.28
    d1 = 1.00000615302
    d2 = 1.98615381364
    d3 = 5.29330324926
    d4 = -15.1508972451
    d5 = 30.789933034
    ltone = 7.0
    p = 0.39894228044
    q = 0.39990348504
    r = 0.398942280385
    utzero = 18.66

    up = bool(upper)
    z = float(x)

    if z < 0.0:
        up = not up
        z = -z

    if ltone < z and ((not up) or utzero < z):
        return 0.0 if up else 1.0

    y = 0.5 * z * z

    if z <= con:
        value = 0.5 - z * (p - q * y
                           / (y + a1 + b1
                              / (y + a2 + b2
                                 / (y + a3))))
    else:
        value = r * np.exp(-y) \
            / (z + c1 + d1
               / (z + c2 + d2
                  / (z + c3 + d3
                     / (z + c4 + d4
                        / (z + c5 + d5
                           / (z + c6))))))

    if not up:
        value = 1.0 - value

    return float(value)


def normal_cdf_inv(p):
    if p <= 0.0:
        return -1e10
    if p >= 1.0:
        return 1e10

    a1 = -3.969683028665376e+01
    a2 = 2.209460984245205e+02
    a3 = -2.759285104469687e+02
    a4 = 1.383577518672690e+02
    a5 = -3.066479806614716e+01
    a6 = 2.506628277459239e+00
    b1 = -5.447609879822406e+01
    b2 = 1.615858368580409e+02
    b3 = -1.556989798598866e+02
    b4 = 6.680131188771972e+01
    b5 = -1.328068155288572e+01
    c1 = -7.784894002430293e-03
    c2 = -3.223964580411365e-01
    c3 = -2.400758277161838e+00
    c4 = -2.549732539343734e+00
    c5 = 4.374664141464968e+00
    c6 = 2.938163982698783e+00
    d1 = 7.784695709041462e-03
    d2 = 3.224671290700398e-01
    d3 = 2.445134137142996e+00
    d4 = 3.754408661907416e+00
    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = np.sqrt(-2.0 * np.log(p))
        x = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) \
            / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        x = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q \
            / (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    else:
        q = np.sqrt(-2.0 * np.log(1.0 - p))
        x = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) \
            / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    return float(x)


def log_normal_pdf(x, mu, sigma):
    x = np.array(x, dtype=float)
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    pdf = np.zeros_like(x, dtype=float)
    mask = x > 0.0
    if np.any(mask):
        lx = np.log(x[mask])
        pdf[mask] = np.exp(-0.5 * ((lx - mu) / sigma) ** 2) \
            / (sigma * x[mask] * np.sqrt(2.0 * np.pi))
    return pdf if pdf.shape != () else float(pdf)


def log_normal_sample(mu, sigma, size=None, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    u = rng.random(size=size)

    u = np.clip(u, 1e-15, 1.0 - 1e-15)
    if size is None:
        z = normal_cdf_inv(float(u))
        return float(np.exp(z * sigma + mu))
    z = np.array([normal_cdf_inv(ui) for ui in u.flat]).reshape(u.shape)
    return np.exp(z * sigma + mu)


def log_normal_mean(mu, sigma):
    return float(np.exp(mu + 0.5 * sigma ** 2))


def log_normal_variance(mu, sigma):
    return float((np.exp(sigma ** 2) - 1.0) * np.exp(2.0 * mu + sigma ** 2))


class TaskWorkload:
    def __init__(self, task_id, base_flops, compute_intensity, memory_footprint,
                 mu_exec, sigma_exec, deadline=None, reference_peak_gflops=100.0):
        self.task_id = task_id
        self.base_flops = float(base_flops)
        self.compute_intensity = float(compute_intensity)
        self.memory_footprint = float(memory_footprint)
        self.mu_exec = float(mu_exec)
        self.sigma_exec = float(sigma_exec)
        self.deadline = float(deadline) if deadline is not None else None
        self.reference_peak_gflops = float(reference_peak_gflops)

    def sample_execution_time(self, processor_speed_ratio=1.0, rng=None):
        adjusted_mu = self.mu_exec - np.log(max(processor_speed_ratio, 1e-12))
        return float(log_normal_sample(adjusted_mu, self.sigma_exec, rng=rng))

    def reliability_probability(self, allocated_time, processor_speed_ratio=1.0):
        adjusted_mu = self.mu_exec - np.log(max(processor_speed_ratio, 1e-12))
        if allocated_time <= 0:
            return 0.0
        z = (np.log(allocated_time) - adjusted_mu) / self.sigma_exec
        return alnorm_cdf(z, upper=False)


def generate_task_set(n_tasks, seed=42):
    rng = np.random.default_rng(seed)
    tasks = []
    for i in range(n_tasks):

        mean_sec = rng.uniform(5.0, 120.0)
        cv = rng.uniform(0.05, 0.3)
        sigma = np.sqrt(np.log(1.0 + cv ** 2))
        mu = np.log(mean_sec) - 0.5 * sigma ** 2

        ref_peak_gflops = 100.0
        base_flops = mean_sec * ref_peak_gflops * 1e9
        compute_intensity = rng.uniform(0.1, 5.0)
        memory_footprint = rng.uniform(1e6, 1e9)
        deadline = mean_sec * rng.uniform(1.5, 4.0)
        tasks.append(TaskWorkload(
            task_id=i,
            base_flops=base_flops,
            compute_intensity=compute_intensity,
            memory_footprint=memory_footprint,
            mu_exec=mu,
            sigma_exec=sigma,
            deadline=deadline,
            reference_peak_gflops=ref_peak_gflops
        ))
    return tasks
