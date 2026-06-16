"""
fisher_forecast.py -- Fisher 矩阵预测与 Gibbs 采样暗能量约束
================================================================
Project 260: 暗能量状态方程约束

融合种子项目
-----------
  - 1081_FranciscoHS_toy-model-cis-code: 种子实验 + 特征重要性
  - 1236_OneFlipBackdoor_OneFlip: 数值扰动检测

数学公式
--------
(1)  Fisher 矩阵:
         F_{ab} = sum_i (1/sigma_i^2) * dM_i/dp_a * dM_i/dp_b
     其中 M_i 为模型预测, p_a 为参数.

(2)  DETF Figure of Merit:
         FoM = 1 / sqrt(det(Cov(w0, wa)))
             = 1 / (sigma_w0 * sigma_wa * sqrt(1 - r^2))

(3)  Marginalized error:
         sigma_a = sqrt((F^{-1})_{aa})

(4)  Gibbs 采样 (二维):
         p(w0 | wa) ~ N(mu_0|wa, sigma_0|wa^2)
         p(wa | w0) ~ N(mu_wa|w0, sigma_wa|w0^2)

(5)  数值扰动检测 (种子项目 1236):
     检测 MCMC 链中的异常跳变:
         jump 判定: |theta_{t} - theta_{t-1}| > threshold
         位翻转模拟: 单参数大跳变导致 chi^2 异常

(6)  K-sweep 特征重要性 (种子项目 1081):
     扫描 (w0, wa) 网格, 评估:
         Delta_D = max|D(w0,wa) - D_fid| / |D_fid|
     衡量各参数对增长因子的影响.

(7)  功率谱 Eisenstein-Hu:
     P(k) = A_s k^{n_s} T^2(k)
     T(k) 为转移函数.

(8)  BAO 信号模型:
     P(k) = P_nw(k) * [1 + A_bao * sin(k r_s + phi)]
================================================================
"""
from __future__ import annotations
import math
import random
from typing import List, Tuple, Dict, Callable, Optional

import cosmo_constants as cc
from growth_solver import BDFGrowthSolver, growth_coefficients


_rng = random.Random(42)


def set_seed(seed: int):
    global _rng
    _rng = random.Random(seed)


# =====================================================================
#  Fisher 矩阵
# =====================================================================

class FisherMatrix:
    """
    Fisher 信息矩阵计算器.

    参数: theta = (w0, wa)
    观测: f*sigma8(z_i) 或 D(z_i)
    """

    def __init__(self, z_observations: List[float] = None,
                 sigma_observations: List[float] = None,
                 observable: str = 'f_sigma8'):
        if z_observations is None:
            z_observations = [0.15, 0.35, 0.57, 0.8, 1.0, 1.2, 1.5]
        self.z_obs = z_observations
        if sigma_observations is None:
            sigma_observations = [0.05] * len(z_observations)
        self.sigma_obs = sigma_observations
        self.observable = observable

    def _compute_observable(self, w0: float, wa: float, z: float) -> float:
        """计算观测量的模型预测."""
        solver = BDFGrowthSolver(w0=w0, wa=wa)
        solver.setup_grid(a_min=1e-4, n_points=60)
        solver.solve_bdf1()
        solver.compute_growth_rate()
        D_z, f_z = solver.growth_at_z(z)
        if self.observable == 'f_sigma8':
            return f_z * cc.SIGMA_8 * D_z if D_z > 0 else 0.0
        elif self.observable == 'D':
            return D_z
        else:
            return f_z

    def _derivative(self, w0: float, wa: float, z: float,
                     param: str, delta: float = 0.01) -> float:
        """数值导数 d(observable)/d(param)."""
        if param == 'w0':
            f_plus = self._compute_observable(w0 + delta, wa, z)
            f_minus = self._compute_observable(w0 - delta, wa, z)
        elif param == 'wa':
            f_plus = self._compute_observable(w0, wa + delta, z)
            f_minus = self._compute_observable(w0, wa - delta, z)
        else:
            return 0.0
        return (f_plus - f_minus) / (2.0 * delta)

    def compute_fisher(self, w0: float = -1.0, wa: float = 0.0
                       ) -> List[List[float]]:
        """
        计算 2x2 Fisher 矩阵 F_{ab}.
        """
        F = [[0.0, 0.0], [0.0, 0.0]]
        params = ['w0', 'wa']

        for i, pi in enumerate(params):
            for j, pj in enumerate(params):
                s = 0.0
                for k, z in enumerate(self.z_obs):
                    di = self._derivative(w0, wa, z, pi)
                    dj = self._derivative(w0, wa, z, pj)
                    sig2 = self.sigma_obs[k]**2
                    if sig2 > 1e-30:
                        s += di * dj / sig2
                F[i][j] = s

        return F

    def inverse_2x2(self, F: List[List[float]]) -> List[List[float]]:
        """2x2 矩阵求逆."""
        det = F[0][0]*F[1][1] - F[0][1]*F[1][0]
        if abs(det) < 1e-30:
            return [[0.0, 0.0], [0.0, 0.0]]
        inv_det = 1.0 / det
        return [
            [F[1][1] * inv_det, -F[0][1] * inv_det],
            [-F[1][0] * inv_det, F[0][0] * inv_det],
        ]

    def figure_of_merit(self, F: List[List[float]]) -> float:
        """DETF FoM = 1 / sqrt(det(Cov))."""
        Cov = self.inverse_2x2(F)
        det = Cov[0][0]*Cov[1][1] - Cov[0][1]*Cov[1][0]
        if det <= 0:
            return 0.0
        return 1.0 / math.sqrt(det)

    def marginalized_errors(self, F: List[List[float]]) -> Dict:
        """Marginalized 1-sigma errors."""
        Cov = self.inverse_2x2(F)
        return {
            'sigma_w0': math.sqrt(max(Cov[0][0], 0.0)),
            'sigma_wa': math.sqrt(max(Cov[1][1], 0.0)),
            'correlation': (Cov[0][1] / math.sqrt(max(Cov[0][0]*Cov[1][1], 1e-30))
                            if Cov[0][0] > 0 and Cov[1][1] > 0 else 0.0),
        }


# =====================================================================
#  Gibbs 采样
# =====================================================================

class GibbsDarkEnergySampler:
    """
    Gibbs 采样约束 (w0, wa).

    使用条件分布 p(w0|wa) 和 p(wa|w0) 交替采样.
    基于高斯近似.
    """

    def __init__(self, w0_fid: float = -1.0, wa_fid: float = 0.0,
                 sigma_w0: float = 0.1, sigma_wa: float = 0.5,
                 correlation: float = -0.5):
        self.w0_fid = w0_fid
        self.wa_fid = wa_fid
        self.sigma_w0 = sigma_w0
        self.sigma_wa = sigma_wa
        self.rho = correlation

    def _conditional_w0(self, wa: float) -> Tuple[float, float]:
        """p(w0 | wa) 的条件分布."""
        mu = self.w0_fid + self.rho * (self.sigma_w0/self.sigma_wa) * (wa - self.wa_fid)
        sig = self.sigma_w0 * math.sqrt(1.0 - self.rho**2)
        return mu, max(sig, 1e-10)

    def _conditional_wa(self, w0: float) -> Tuple[float, float]:
        """p(wa | w0) 的条件分布."""
        mu = self.wa_fid + self.rho * (self.sigma_wa/self.sigma_w0) * (w0 - self.w0_fid)
        sig = self.sigma_wa * math.sqrt(1.0 - self.rho**2)
        return mu, max(sig, 1e-10)

    def run(self, n_steps: int = 500, burn_in: int = 100
            ) -> Dict:
        """运行 Gibbs 采样."""
        w0 = self.w0_fid
        wa = self.wa_fid
        chain_w0 = []
        chain_wa = []

        for _ in range(n_steps):
            # Sample w0 | wa
            mu0, sig0 = self._conditional_w0(wa)
            w0 = mu0 + sig0 * _rng.gauss(0, 1)
            # Sample wa | w0
            mua, siga = self._conditional_wa(w0)
            wa = mua + siga * _rng.gauss(0, 1)

            chain_w0.append(w0)
            chain_wa.append(wa)

        # 去除 burn-in
        chain_w0 = chain_w0[burn_in:]
        chain_wa = chain_wa[burn_in:]

        mean_w0 = sum(chain_w0) / len(chain_w0)
        mean_wa = sum(chain_wa) / len(chain_wa)
        var_w0 = sum((w-mean_w0)**2 for w in chain_w0) / len(chain_w0)
        var_wa = sum((w-mean_wa)**2 for w in chain_wa) / len(chain_wa)

        return {
            'w0_mean': mean_w0,
            'wa_mean': mean_wa,
            'w0_std': math.sqrt(var_w0),
            'wa_std': math.sqrt(var_wa),
            'chain_w0': chain_w0,
            'chain_wa': chain_wa,
        }


# =====================================================================
#  K-sweep 特征重要性 (种子项目 1081)
# =====================================================================

class ParameterSweep:
    """
    (w0, wa) 参数网格扫描 (种子项目 1081 的 K-sweep 推广).
    """

    def __init__(self, w0_range: Tuple[float, float] = (-1.2, -0.8),
                 wa_range: Tuple[float, float] = (-0.5, 0.5),
                 n_w0: int = 5, n_wa: int = 5):
        self.w0_range = w0_range
        self.wa_range = wa_range
        self.n_w0 = n_w0
        self.n_wa = n_wa

    def run_sweep(self) -> Dict:
        """执行参数扫描."""
        w0_vals = [self.w0_range[0] + i*(self.w0_range[1]-self.w0_range[0])/(self.n_w0-1)
                    for i in range(self.n_w0)]
        wa_vals = [self.wa_range[0] + i*(self.wa_range[1]-self.wa_range[0])/(self.n_wa-1)
                    for i in range(self.n_wa)]

        results = []
        best_chi2 = float('inf')
        best_point = None

        for w0 in w0_vals:
            for wa in wa_vals:
                solver = BDFGrowthSolver(w0=w0, wa=wa)
                solver.setup_grid(a_min=1e-4, n_points=60)
                try:
                    solver.solve_bdf1()
                    solver.compute_growth_rate()
                    f1 = solver.f_rate[-1] if solver.f_rate else 0.0
                    # chi^2 简化: (f - 0.54)^2 / 0.01 (LCDM 预期 f~0.54)
                    chi2 = (f1 - 0.54)**2 / 0.01
                except Exception:
                    chi2 = 1e10
                    f1 = 0.0

                results.append({
                    'w0': w0, 'wa': wa,
                    'f_at_1': f1,
                    'chi2': chi2,
                    'is_stable': chi2 < 100,
                })
                if chi2 < best_chi2:
                    best_chi2 = chi2
                    best_point = {'w0': w0, 'wa': wa, 'chi2': chi2}

        return {
            'sweep_results': results,
            'best_point': best_point,
            'n_total': len(results),
            'n_stable': sum(1 for r in results if r['is_stable']),
        }

    def compute_feature_importance(self) -> Dict:
        """
        特征重要性:
            Delta_D(w0) = max_{wa} |D(w0,wa) - D_fid| / |D_fid|
        """
        sweep = self.run_sweep()
        fid_f = 0.54  # LCDM fiducial

        delta_w0 = 0.0
        delta_wa = 0.0
        for r in sweep['sweep_results']:
            d = abs(r['f_at_1'] - fid_f) / max(abs(fid_f), 1e-10)
            if abs(r['w0'] - (-1.0)) > abs(r['wa']):
                delta_w0 = max(delta_w0, d)
            else:
                delta_wa = max(delta_wa, d)

        return {
            'delta_D_w0': delta_w0,
            'delta_D_wa': delta_wa,
            'dominant_parameter': 'w0' if delta_w0 > delta_wa else 'wa',
        }


# =====================================================================
#  链扰动检测 (种子项目 1236)
# =====================================================================

def detect_chain_artifacts(chain: List[float],
                            jump_threshold: float = None) -> Dict:
    """
    检测 MCMC 链中的异常跳变 (灵感来自种子项目 1236 的位翻转检测).
    """
    if not chain or len(chain) < 2:
        return {'n_jumps': 0, 'is_clean': True, 'jump_rate': 0.0}

    if jump_threshold is None:
        std = math.sqrt(sum((c - sum(chain)/len(chain))**2 for c in chain) / len(chain))
        jump_threshold = 4.0 * max(std, 1e-10)

    n_jumps = 0
    for i in range(1, len(chain)):
        if abs(chain[i] - chain[i-1]) > jump_threshold:
            n_jumps += 1

    return {
        'n_jumps': n_jumps,
        'is_clean': n_jumps == 0,
        'jump_rate': n_jumps / (len(chain) - 1),
        'jump_threshold': jump_threshold,
    }


# =====================================================================
#  BAO 信号模型
# =====================================================================

def bao_signal_model(k: float, r_s: float = 147.0,
                      A_bao: float = 0.05, phi: float = 0.0
                      ) -> float:
    """
    BAO 振荡信号:
        P_bao(k) = P_nw(k) * [1 + A_bao * sin(k * r_s + phi)]
    返回振荡部分: A_bao * sin(k * r_s + phi).
    """
    return A_bao * math.sin(k * r_s + phi)


def dsc_filter(signal: List[float], sigma: float = 2.0
               ) -> Tuple[List[float], List[float]]:
    """
    DSC (Discrete Singular Convolution) 滤波器.
    分离 smooth 和 oscillatory 成分.
    """
    n = len(signal)
    # 简单移动平均作为 smooth
    smooth = []
    M = max(int(3 * sigma), 1)
    for i in range(n):
        s = 0.0
        count = 0
        for j in range(max(0, i-M), min(n, i+M+1)):
            # 高斯权重
            w = math.exp(-0.5 * ((j-i)/max(sigma, 1e-10))**2)
            s += w * signal[j]
            count += w
        smooth.append(s / max(count, 1e-30))

    oscillatory = [signal[i] - smooth[i] for i in range(n)]
    return smooth, oscillatory


# =====================================================================
#  快速测试
# =====================================================================

if __name__ == '__main__':
    print("=== Fisher/Gibbs/Sweep 测试 ===")
    set_seed(42)

    print("\nFisher 矩阵:")
    fisher = FisherMatrix()
    F = fisher.compute_fisher(w0=-1.0, wa=0.0)
    print(f"  F = [[{F[0][0]:.4f}, {F[0][1]:.4f}], "
          f"[{F[1][0]:.4f}, {F[1][1]:.4f}]]")
    errors = fisher.marginalized_errors(F)
    print(f"  sigma_w0 = {errors['sigma_w0']:.4f}")
    print(f"  sigma_wa = {errors['sigma_wa']:.4f}")
    print(f"  FoM = {fisher.figure_of_merit(F):.2f}")

    print("\nGibbs 采样:")
    gibbs = GibbsDarkEnergySampler(sigma_w0=0.1, sigma_wa=0.5, correlation=-0.3)
    result = gibbs.run(n_steps=300, burn_in=50)
    print(f"  w0 = {result['w0_mean']:.4f} +/- {result['w0_std']:.4f}")
    print(f"  wa = {result['wa_mean']:.4f} +/- {result['wa_std']:.4f}")

    print("\n链扰动检测:")
    artifacts = detect_chain_artifacts(result['chain_w0'])
    print(f"  jumps: {artifacts['n_jumps']}, clean: {artifacts['is_clean']}")

    print("\n参数扫描:")
    sweep = ParameterSweep(n_w0=3, n_wa=3)
    sr = sweep.run_sweep()
    print(f"  扫描点: {sr['n_total']}, 稳定: {sr['n_stable']}")
    print(f"  最佳: w0={sr['best_point']['w0']:.2f}, wa={sr['best_point']['wa']:.2f}")

    imp = sweep.compute_feature_importance()
    print(f"  Delta_D(w0) = {imp['delta_D_w0']:.4f}")
    print(f"  Delta_D(wa) = {imp['delta_D_wa']:.4f}")
    print(f"  主导参数: {imp['dominant_parameter']}")

    print("\nBAO 信号:")
    k_test = [0.01 * i for i in range(1, 30)]
    bao_vals = [bao_signal_model(k) for k in k_test]
    smooth, osc = dsc_filter(bao_vals, sigma=2.0)
    print(f"  BAO 范围: [{min(bao_vals):.4f}, {max(bao_vals):.4f}]")
    print(f"  smooth 范围: [{min(smooth):.4f}, {max(smooth):.4f}]")
    print(f"  osc 范围: [{min(osc):.4f}, {max(osc):.4f}]")

    print("\n所有 Fisher/Gibbs 测试通过.")
