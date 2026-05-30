
import numpy as np
from scipy.special import betainc, gammaln, factorial
from scipy.stats import beta as beta_dist


def noncentral_beta_cdf(x, a, b, lam, errmax=1e-10, max_iter=1000):
    if a <= 0 or b <= 0:
        return 0.0, 3
    if lam < 0:
        return 0.0, 3
    if x <= 0.0:
        return 0.0, 0
    if x >= 1.0:
        return 1.0, 0


    if lam < 54.0:
        c = 0.5 * lam
        cdf = 0.0
        j = 0
        while j < max_iter:
            pois_prob = np.exp(-c + j * np.log(c) - gammaln(j + 1.0))
            if pois_prob < errmax and j > c:
                break
            beta_cdf = betainc(a + j, b, x)
            cdf += pois_prob * beta_cdf
            j += 1
        return cdf, 0


    m = int(np.floor(0.5 * lam + 0.5))
    c = 0.5 * lam
    iterlo = max(0, m - int(5.0 * np.sqrt(m)))
    iterhi = m + int(5.0 * np.sqrt(m))

    t = -c + m * np.log(c) - gammaln(m + 1.0)
    q = np.exp(t)
    r = q
    psum = q

    beta_ln = gammaln(a + m) + gammaln(b) - gammaln(a + m + b)
    s1 = (a + m) * np.log(x) + b * np.log(1.0 - x) - np.log(a + m) - beta_ln
    gx = np.exp(s1)
    fx = gx
    ftemp = betainc(a + m, b, x)
    sum_val = q * ftemp


    iter1 = m
    while iter1 > iterlo and q > errmax:
        q = q * iter1 / c
        gx = (a + iter1) / (x * (a + b + iter1 - 1.0)) * gx
        iter1 -= 1
        temp = ftemp + gx
        psum += q
        sum_val += q * temp


    q = r
    temp = ftemp
    gx = fx
    iter2 = m
    while iter2 < iterhi:
        ebd = (1.0 - psum) * temp
        if ebd < errmax:
            break
        iter2 += 1
        q = q * c / iter2
        psum += q
        temp = temp - gx
        gx = x * (a + b + iter2 - 1.0) / (a + iter2) * gx
        sum_val += q * temp

    return sum_val, 0


class PatientVariabilityModel:

    def __init__(self, sigma_mean=0.3, sigma_std=0.08,
                 survival_alpha=5.0, survival_beta=2.0, survival_lambda=2.0):
        self.sigma_mean = float(sigma_mean)
        self.sigma_std = float(sigma_std)
        self.survival_alpha = float(survival_alpha)
        self.survival_beta = float(survival_beta)
        self.survival_lambda = float(survival_lambda)

    def sample_conductivity(self, n_samples=1):
        samples = np.random.randn(n_samples) * self.sigma_std + self.sigma_mean
        samples = np.maximum(samples, 0.05)
        return samples

    def sample_neural_survival_rate(self, n_samples=1):


        c = 0.5 * self.survival_lambda
        n = np.random.poisson(c, n_samples)
        alpha_eff = self.survival_alpha + n
        samples = beta_dist.rvs(alpha_eff, self.survival_beta)
        return np.clip(samples, 0.0, 1.0)

    def sample_electrode_offset(self, n_samples=1):
        k, theta = 4.0, 0.15
        samples = np.random.gamma(k, theta, n_samples)
        return np.clip(samples, 0.1, 2.0)

    def generate_patient_cohort(self, n_patients=100):
        return {
            'conductivity': self.sample_conductivity(n_patients),
            'survival_rate': self.sample_neural_survival_rate(n_patients),
            'offset': self.sample_electrode_offset(n_patients),
        }

    def probability_threshold_hearing(self, survival_rate, threshold=0.5):
        cdf, ifault = noncentral_beta_cdf(
            survival_rate, self.survival_alpha, self.survival_beta,
            self.survival_lambda
        )
        if ifault != 0:
            return 0.0
        return 1.0 - cdf


def clinical_outcome_probability(survival_rate, stimulation_level,
                                  alpha=5.0, beta=2.0, lambda_nc=2.0):

    effective_survival = survival_rate * (0.5 + 0.5 * stimulation_level)
    effective_survival = min(effective_survival, 0.999)

    cdf, ifault = noncentral_beta_cdf(
        effective_survival, alpha, beta, lambda_nc
    )
    if ifault != 0:
        return 0.0
    return cdf
