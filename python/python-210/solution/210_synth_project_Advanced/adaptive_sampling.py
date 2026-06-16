"""
adaptive_sampling.py - 自适应采样策略：CVT-Lloyd + 重要性采样

本模块实现基于 Lloyd-CVT 迭代的最优采样与自适应重要性采样算法，
用于高效估计小失效概率。

Lloyd 算法 (CVT 迭代):
  1. 给定 N 个生成点 {z_i}, 构造 Voronoi 划分
  2. 计算每个 Voronoi 区域的密度加权质心
  3. 用质心更新生成点
  4. 重复直至收敛

CVT 能量: E = sum_i int_{V_i} rho(x) ||x - z_i||^2 dx
最小化 CVT 能量等价于寻找最优的 L2 量化点集。

自适应重要性采样:
  h(u) = sum_i w_i * phi(u - z_i)   (混合高斯)
  P_f = (1/N) sum I(g(u_k)<=0) * phi(u_k) / h(u_k)

种子项目映射:
  247_cvt_2d_lumping → Lloyd 算法核心 + 密度加权质心
  252_cvt_box        → CVT 能量计算 + 边界投影
  241_cvt_3_movie    → 拒绝采样 + 区域判定
  182_circle_positive_distance → 球面采样
"""

import numpy as np
from random_variables import std_normal_pdf


def cvt_lloyd_2d(n_generators, n_samples, n_iter, density_fn=None, rng=None):
    """Lloyd 算法: 2D 质心 Voronoi 剖分 (CVT)

    种子项目 247_cvt_2d_lumping 的 Python 实现。

    Returns
    -------
    generators : np.ndarray (n_generators, 2)
    energy_history : np.ndarray
    motion_history : np.ndarray
    """
    rng = rng or np.random.default_rng(42)
    generators = 2.0 * rng.random((n_generators, 2)) - 1.0

    s1d = np.linspace(-1.0 + 1e-10, 1.0 - 1e-10, int(np.sqrt(max(n_samples, 4))))
    sx, sy = np.meshgrid(s1d, s1d)
    samples = np.column_stack([sx.ravel(), sy.ravel()])
    n_s = samples.shape[0]

    if density_fn is not None:
        rho = density_fn(sx, sy).ravel()
        rho = np.minimum(rho, 10.0)
    else:
        rho = np.ones(n_s)

    rho = rho ** 0.5  # 2D CVT 渐近: rho_eff = mu^4, 此处取 mu = rho^{1/4}

    energy_hist = []
    motion_hist = []

    for _ in range(n_iter):
        # 分配采样点到最近生成点
        dists = np.sum((samples[:, None, :] - generators[None, :, :]) ** 2, axis=2)
        k = np.argmin(dists, axis=1)

        generators_new = np.zeros_like(generators)
        for i in range(n_generators):
            mask = (k == i)
            mass = np.sum(rho[mask])
            if mass > 1e-15:
                generators_new[i, 0] = np.sum(rho[mask] * samples[mask, 0]) / mass
                generators_new[i, 1] = np.sum(rho[mask] * samples[mask, 1]) / mass
            else:
                generators_new[i] = generators[i]

        assigned = generators[k]
        energy = np.sum(rho * np.sum((samples - assigned) ** 2, axis=1)) / n_s
        motion = np.sum((generators_new - generators) ** 2) / n_generators
        energy_hist.append(energy)
        motion_hist.append(motion)
        generators = generators_new

    return generators, np.array(energy_hist), np.array(motion_hist)


def cvt_lloyd_nd(n_generators, n_samples, n_iter, n_dim=3, density_fn=None, rng=None):
    """N-D Lloyd CVT 算法"""
    rng = rng or np.random.default_rng(42)
    generators = 2.0 * rng.random((n_generators, n_dim)) - 1.0
    n_per_dim = max(int(n_samples ** (1.0 / n_dim)), 3)
    grids = [np.linspace(-1.0 + 1e-10, 1.0 - 1e-10, n_per_dim) for _ in range(n_dim)]
    mesh = np.meshgrid(*grids, indexing='ij')
    samples = np.column_stack([m.ravel() for m in mesh])
    n_s = samples.shape[0]
    rho = np.ones(n_s)
    if density_fn is not None:
        rho = np.clip(density_fn(samples), 0, 10.0)

    for _ in range(n_iter):
        block = max(1, min(5000, n_s))
        k = np.zeros(n_s, dtype=int)
        for s in range(0, n_s, block):
            e = min(s + block, n_s)
            d = np.sum((samples[s:e, None, :] - generators[None, :, :]) ** 2, axis=2)
            k[s:e] = np.argmin(d, axis=1)

        g_new = np.zeros_like(generators)
        for i in range(n_generators):
            mask = (k == i)
            mass = np.sum(rho[mask])
            if mass > 1e-15:
                for d in range(n_dim):
                    g_new[i, d] = np.sum(rho[mask] * samples[mask, d]) / mass
            else:
                g_new[i] = generators[i]
        generators = g_new

    return generators


def cvt_energy(generators, n_samples, n_dim=2, rng=None):
    """CVT 能量: E = (1/N) sum ||s_i - z_{k(i)}||^2"""
    rng = rng or np.random.default_rng()
    samples = 2.0 * rng.random((n_samples, n_dim)) - 1.0
    d = np.sum((samples[:, None, :] - generators[None, :, :]) ** 2, axis=2)
    k = np.argmin(d, axis=1)
    return float(np.mean(np.sum((samples - generators[k]) ** 2, axis=1)))


def rejection_sample_in_domain(sample_fn, inside_fn, n, max_ratio=10, rng=None):
    """拒绝采样 (种子项目 241_cvt_3_movie p08_sample)"""
    rng = rng or np.random.default_rng()
    pts = []
    n_rej = 0
    batch = min(1000, max(n, 100))
    while len(pts) < n:
        s = sample_fn(min(batch, n * 3))
        ok = inside_fn(s)
        for j in range(s.shape[0]):
            if ok[j]:
                pts.append(s[j])
                if len(pts) >= n:
                    break
            else:
                n_rej += 1
        if n_rej > max_ratio * n:
            break
    return np.array(pts[:n]) if pts else np.empty((0, 2))


class AdaptiveImportanceSampler:
    """自适应重要性采样器

    利用 Lloyd-CVT 迭代优化重要性采样密度, 核心公式:
      P_f = E_h[ I(g(u)<=0) * phi(u)/h(u) ]
      ≈ (1/N) sum I(g(u_k)<=0) * phi_n(u_k) / h(u_k)
    其中 h(u) = (1/K) sum phi_n(u - z_i) 为 CVT 混合密度。
    """

    def __init__(self, lsf, n_dim, n_generators=None, rng_seed=42):
        self.lsf = lsf
        self.n_dim = n_dim
        self.n_generators = n_generators or max(20, 5 * n_dim)
        self.rng = np.random.default_rng(rng_seed)
        self.generators = None
        self.energy_history = None

    def _setup_generators(self, n_cvt_iter=20, n_cvt_samples=None):
        if n_cvt_samples is None:
            n_cvt_samples = max(500, 50 * self.n_generators)
        if self.n_dim == 2:
            self.generators, self.energy_history, _ = cvt_lloyd_2d(
                self.n_generators, n_cvt_samples, n_cvt_iter, rng=self.rng
            )
        else:
            self.generators = cvt_lloyd_nd(
                self.n_generators, n_cvt_samples, n_cvt_iter,
                n_dim=self.n_dim, rng=self.rng
            )

    def _mixture_pdf(self, u):
        u = np.atleast_2d(u)
        K = self.generators.shape[0]
        log_sum = np.full(u.shape[0], -np.inf)
        for i in range(K):
            diff = u - self.generators[i]
            log_phi = -0.5 * np.sum(diff ** 2, axis=1) - 0.5 * self.n_dim * np.log(2 * np.pi)
            log_sum = np.logaddexp(log_sum, log_phi - np.log(K))
        return np.exp(log_sum)

    def _normal_pdf(self, u):
        u = np.atleast_2d(u)
        return np.exp(-0.5 * np.sum(u ** 2, axis=1) - 0.5 * self.n_dim * np.log(2 * np.pi))

    def importance_sampling(self, n_samples=10000):
        if self.generators is None:
            self._setup_generators()

        samples = np.zeros((n_samples, self.n_dim))
        comps = self.rng.integers(0, self.n_generators, size=n_samples)
        for k in range(n_samples):
            samples[k] = self.generators[comps[k]] + self.rng.standard_normal(self.n_dim)

        g_vals = self.lsf.evaluate(samples)
        fail = (g_vals <= 0).astype(float)
        phi = self._normal_pdf(samples)
        h = np.maximum(self._mixture_pdf(samples), 1e-300)
        w = phi / h

        wf = fail * w
        pf = float(np.mean(wf))
        if pf > 1e-30:
            cov = float(np.std(wf) / (np.sqrt(n_samples) * pf))
        else:
            cov = float('inf')
        return pf, cov, samples, w

    def adaptive_run(self, n_samples=10000, max_outer=5, cov_target=0.05, cvt_iter=20,
                     form_design_point=None):
        pf_hist, cov_hist = [], []
        for outer in range(max_outer):
            self._setup_generators(n_cvt_iter=cvt_iter)
            # 将生成点偏向 FORM 设计点 (若提供)
            if form_design_point is not None and outer == 0:
                u_star = np.asarray(form_design_point)
                beta = np.linalg.norm(u_star)
                # 将部分生成点移到设计点附近
                n_shift = min(self.n_generators // 2, self.n_generators)
                for i in range(n_shift):
                    self.generators[i] = u_star + 0.5 * (self.generators[i] - u_star)
            pf, cov, _, _ = self.importance_sampling(n_samples)
            pf_hist.append(pf)
            cov_hist.append(cov)
            if cov < cov_target and pf > 0:
                break
            if 0 < pf < 1:
                shift = 0.1 / (1.0 + outer)
                for i in range(self.n_generators):
                    grad = self.lsf.gradient(self.generators[i])
                    gn = np.linalg.norm(grad)
                    if gn > 1e-12:
                        self.generators[i] -= shift * grad / gn
        return {
            'pf': pf_hist[-1] if pf_hist else 0.0,
            'cov': cov_hist[-1] if cov_hist else float('inf'),
            'pf_history': pf_hist,
            'cov_history': cov_hist,
            'generators': self.generators.copy() if self.generators is not None else None
        }
