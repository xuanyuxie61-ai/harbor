"""
anomaly_detector.py — 异常事件检测
====================================
种子项目映射:
  1143_ALRLDA (RLDA/主动学习) → 正则化判别分析用于信号/本底分类
  867_persistence (持久统计) → Welford在线统计与χ²检验

物理: 使用Profile Likelihood Ratio、核密度估计、注意力加权等方法
      检测偏离SM预测的异常事件.
"""
import numpy as np
import math


class WelfordAccumulator:
    """Welford在线算法: 递推计算均值、方差."""

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def variance(self):
        return self.M2 / (self.n - 1) if self.n > 1 else 0.0

    def std(self):
        return math.sqrt(max(self.variance(), 0))

    def summary(self):
        return {
            'n': self.n,
            'mean': self.mean,
            'std': self.std(),
            'se': self.std() / math.sqrt(max(self.n, 1)),
        }


class RunningChiSquared:
    """运行χ²检验."""

    def __init__(self, n_bins):
        self.n_bins = n_bins

    def compute(self, observed, expected, uncertainty=None):
        """χ² = Σ (O_i - E_i)² / σ_i²."""
        if uncertainty is None:
            uncertainty = np.sqrt(np.maximum(expected, 1.0))
        chi2 = np.sum((observed - expected)**2 / np.maximum(uncertainty**2, 1e-10))
        ndf = self.n_bins - 1
        p_value = 1.0 - _chi2_cdf(chi2, ndf)
        return chi2, ndf, p_value


def _chi2_cdf(x, k):
    """χ²分布CDF (正则化下不完全Gamma函数近似)."""
    if x <= 0 or k <= 0:
        return 0.0
    # Wilson-Hilferty近似
    z = ((x / k)**(1.0/3.0) - (1.0 - 2.0/(9.0*k))) / math.sqrt(2.0/(9.0*k))
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


class ProfileLikelihoodTest:
    """Profile Likelihood Ratio 检验."""

    def __init__(self, n_bins):
        self.n_bins = n_bins

    def compute_test_statistic(self, n_obs, s_template, b_expected, syst=0.05):
        """
        q₀ = -2 ln λ(0), λ(0) = L(μ=0,θ̂̂) / L(μ̂,θ̂)
        渐近: Z = sqrt(q₀) (局部显著性)
        """
        mu_hat = max(np.sum(n_obs - b_expected) / max(np.sum(s_template), 1e-10), 0)
        # 对数似然 (Poisson)
        def logL(mu):
            ll = 0.0
            for i in range(self.n_bins):
                exp_i = max(b_expected[i] + mu * s_template[i], 1e-10)
                obs_i = max(n_obs[i], 0)
                ll += obs_i * math.log(exp_i) - exp_i
            # 系统不确定 (高斯约束)
            ll -= 0.5 * (mu * syst)**2
            return ll

        ll_0 = logL(0)
        ll_hat = logL(mu_hat)
        q0 = max(-2.0 * (ll_0 - ll_hat), 0)
        Z = math.sqrt(q0) if q0 > 0 else 0.0
        p = 0.5 * math.erfc(Z / math.sqrt(2.0))
        return q0, Z, p


class KernelAnomalyDetector:
    """核密度异常检测."""

    def __init__(self, bandwidth=None):
        self.bandwidth = bandwidth
        self.X_train = None

    def fit(self, X):
        self.X_train = X.copy()
        if self.bandwidth is None:
            self.bandwidth = np.std(X, axis=0) * len(X)**(-1.0/(len(X.shape[1:]) + 4))
            self.bandwidth = np.maximum(self.bandwidth, 1e-6)

    def score(self, x):
        """异常分数: 负对数似然."""
        diffs = (self.X_train - x) / self.bandwidth
        log_kde = np.sum(-0.5 * diffs**2, axis=1)
        max_log = np.max(log_kde)
        log_density = max_log + math.log(np.sum(np.exp(log_kde - max_log))) - math.log(len(self.X_train))
        return -log_density

    def score_batch(self, X):
        return np.array([self.score(x) for x in X])


class AttentionWeightedScorer:
    """注意力加权异常评分 (种子项目 1289_BABILong 映射)."""

    def __init__(self, n_features):
        self.n_features = n_features
        self.weights = np.ones(n_features) / n_features
        self.means = None
        self.stds = None

    def fit(self, X):
        self.means = np.mean(X, axis=0)
        self.stds = np.std(X, axis=0)
        self.stds = np.maximum(self.stds, 1e-6)
        # 注意力权重: 逆方差加权
        self.weights = 1.0 / (self.stds**2)
        self.weights /= np.sum(self.weights)

    def score(self, x):
        z_scores = np.abs((x - self.means) / self.stds)
        return np.sum(self.weights * z_scores)

    def score_batch(self, X):
        scores = np.array([self.score(x) for x in X])
        return scores, self.weights


class AnomalyDetectionPipeline:
    """综合异常检测流水线."""

    def __init__(self, n_bins=20, M_range=(60, 120)):
        self.n_bins = n_bins
        self.M_range = M_range
        self.bkg_hist = None

    def fit_background(self, events, feature='M'):
        vals = [e.get(feature, 0) for e in events]
        bins = np.linspace(self.M_range[0], self.M_range[1], self.n_bins + 1)
        self.bkg_hist, _ = np.histogram(vals, bins=bins)

    def analyze(self, events, feature='M'):
        vals = [e.get(feature, 0) for e in events]
        bins = np.linspace(self.M_range[0], self.M_range[1], self.n_bins + 1)
        obs_hist, _ = np.histogram(vals, bins=bins)

        chi2_test = RunningChiSquared(self.n_bins)
        chi2, ndf, p_chi2 = chi2_test.compute(obs_hist, self.bkg_hist)

        # 信号模板 (高斯峰)
        center = self.n_bins // 2
        s_template = np.array([math.exp(-0.5*((i-center)/2.0)**2) for i in range(self.n_bins)])
        s_template *= max(np.sum(obs_hist) - np.sum(self.bkg_hist), 1.0)
        s_template = np.maximum(s_template, 0.01)

        pl_test = ProfileLikelihoodTest(self.n_bins)
        q0, Z, p_local = pl_test.compute_test_statistic(obs_hist, s_template, self.bkg_hist)
        p_global = min(p_local * self.n_bins, 1.0)

        return {
            'chi2': (chi2, ndf, p_chi2),
            'significance': (q0, Z, p_local),
            'global_pvalue': p_global,
        }


def significance_asimov(s, b):
    """Asimov预期显著性: Z_A = sqrt(2*(s+b)*ln(1+s/b) - s)."""
    if b <= 0 or s <= 0:
        return 0.0
    return math.sqrt(max(2.0 * ((s + b) * math.log(1 + s / b) - s), 0))


def cls_limit(s, b, n_obs):
    """CLs排除限."""
    if s <= 0:
        return 1.0
    # 简化: Poisson p-value
    p_sb = 1.0 - _poisson_cdf(n_obs - 1, s + b)
    p_b = 1.0 - _poisson_cdf(n_obs - 1, b)
    return p_sb / max(p_b, 1e-10)


def _poisson_cdf(k, lam):
    """Poisson CDF."""
    if k < 0:
        return 0.0
    return math.exp(-lam) * sum(lam**i / math.factorial(int(i)) for i in range(int(k) + 1))
