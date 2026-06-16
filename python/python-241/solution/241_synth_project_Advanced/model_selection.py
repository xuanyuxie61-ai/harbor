"""
model_selection.py
===================================================================
光学势模型选择与顺序检验模块

映射种子项目:
  - 1141_sshekhar17_nonparametric-testing-by-betting: 赌徒序贯检验
    → 光学势模型 (Woods-Saxon vs 方势阱 vs 高斯势) 的序贯假设检验

核心方法:
  序贯赌徒检验 (Martingale betting test):
    零假设 H0: 两个模型产生相同的截面数据
    备择假设 H1: 两个模型产生不同的截面数据

    财富过程: W_t = prod_{s=1}^t (1 + lambda_s * f_s(x_s))
    其中 f_s(x_s) 为第 s 个数据点的支付函数

    ONS (Online Newton Step) 下注策略:
      lambda_{t+1} = Pi([-c,c])(lambda_t - z_t * grad / A_t)

    Ville不等式停时:
      tau = inf{t : W_t >= 1/alpha}

    若 tau < N: 拒绝 H0 (模型显著不同)
    否则: 不能拒绝 H0
===================================================================
"""

import numpy as np
from typing import Tuple, Dict, List, Optional


# ---------- 物理常数 ----------
HBAR_C = 197.3269804


class SequentialModelTester:
    """
    基于赌徒财富过程的序贯模型检验器
    (映射自 nonparametric-testing-by-betting 的 martingale 框架)

    用途: 比较不同光学势模型 (Woods-Saxon vs 方势阱 vs 折叠高斯)
    对实验截面数据的拟合优度差异是否统计显著.

    优势:
    - 无需预设样本量
    - 控制第一类错误 ( Ville 不等式保证)
    - 自适应下注策略提高检验功效
    """

    def __init__(
        self,
        alpha: float = 0.05,
        lambda_max: float = 10.0,
        bet_strategy: str = 'ons'
    ):
        """
        参数:
            alpha: 显著性水平
            lambda_max: 最大下注额度
            bet_strategy: 'ons' | 'kelly' | 'fixed'
        """
        self.alpha = alpha
        self.lambda_max = lambda_max
        self.strategy = bet_strategy
        self.threshold = 1.0 / alpha

    def compute_payoffs(
        self,
        data: np.ndarray,
        model_a: np.ndarray,
        model_b: np.ndarray,
        sigma_noise: float = 0.1
    ) -> np.ndarray:
        """
        计算每个数据点的支付函数 (payoff):
            f_t = (data_t - model_b_t)^2 - (data_t - model_a_t)^2) / (2*sigma^2)

        正支付 => 数据更支持模型 A
        负支付 => 数据更支持模型 B

        参数:
            data: 实验/参考截面数据
            model_a: 模型 A 预测
            model_b: 模型 B 预测
            sigma_noise: 噪声标准差

        返回:
            支付序列 f_t
        """
        resid_a = (data - model_a) ** 2
        resid_b = (data - model_b) ** 2

        return (resid_b - resid_a) / (2.0 * sigma_noise ** 2 + 1e-30)

    def wealth_process(
        self,
        payoffs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        计算赌徒财富过程
        (核心算法映射自 SeqLC 的 ONS/Kelly 下注)

        财富更新:
            W_t = W_{t-1} * (1 + lambda_t * f_t)

        ONS 下注策略:
            grad_t = f_t
            A_t = sum_{s<=t} f_s^2 + epsilon
            lambda_{t+1} = clip(lambda_t - eta * f_t / A_t, -lambda_max, lambda_max)

        Kelly 下注策略:
            lambda = sum(f) / (sum(f^2) + eps)

        参数:
            payoffs: 支付序列

        返回:
            (wealth, lambdas, stopping_time)
        """
        n = len(payoffs)
        wealth = np.ones(n + 1)
        lambdas = np.zeros(n)
        stopping_time = n  # 默认不停止

        lam = 0.0  # 初始下注
        A = 1e-8   # 梯度平方累积
        eta = 1.0  # 学习率

        for t in range(n):
            f_t = payoffs[t]

            # 下注策略
            if self.strategy == 'ons':
                # ONS 更新
                grad = f_t
                A += grad ** 2
                lam = lam - eta * grad / A
                lam = np.clip(lam, -self.lambda_max, self.lambda_max)
            elif self.strategy == 'kelly':
                # Kelly 最优下注
                lam = np.mean(payoffs[:t + 1]) / (
                    np.mean(payoffs[:t + 1] ** 2) + 1e-8)
                lam = np.clip(lam, -self.lambda_max, self.lambda_max)
            else:
                # 固定下注
                lam = 0.1

            lambdas[t] = lam

            # 财富更新
            growth = 1.0 + lam * f_t
            if growth <= 0:
                # 破产保护
                wealth[t + 1] = wealth[t] * 1e-10
            else:
                wealth[t + 1] = wealth[t] * growth

            # 停时检查
            if wealth[t + 1] >= self.threshold and stopping_time == n:
                stopping_time = t + 1

        return wealth, lambdas, stopping_time

    def test_models(
        self,
        data: np.ndarray,
        model_a: np.ndarray,
        model_b: np.ndarray,
        sigma_noise: float = 0.1
    ) -> Dict:
        """
        执行完整的序贯模型比较检验

        H0: model_a 和 model_b 同等拟合数据
        H1: model_a 优于 model_b (或反之)

        返回:
            检验结果字典
        """
        payoffs = self.compute_payoffs(data, model_a, model_b, sigma_noise)
        wealth, lambdas, tau = self.wealth_process(payoffs)

        # 最终判定
        if tau < len(payoffs):
            reject_h0 = True
            winner = 'model_A' if payoffs[tau - 1] > 0 else 'model_B'
        else:
            reject_h0 = False
            winner = 'inconclusive'

        return {
            'reject_null': reject_h0,
            'winner': winner,
            'stopping_time': tau,
            'final_wealth': float(wealth[-1]),
            'threshold': self.threshold,
            'max_wealth': float(np.max(wealth)),
            'final_payoff_mean': float(np.mean(payoffs)),
            'strategy': self.strategy,
            'alpha': self.alpha,
            'n_data': len(data),
        }

    def power_analysis(
        self,
        effect_size: float = 0.1,
        n_samples: int = 200,
        n_trials: int = 50,
        seed: int = 42
    ) -> Dict:
        """
        检验功效分析 (蒙特卡洛)

        在 H1 为真 (模型差异 = effect_size) 下,
        估计检验拒绝 H0 的概率

        返回:
            功效估计
        """
        rng = np.random.RandomState(seed)
        rejections = 0

        for trial in range(n_trials):
            # 生成数据 (H1: 模型 A 为真)
            x = np.linspace(0, 1, n_samples)
            data = np.sin(2 * np.pi * x) + effect_size + rng.randn(n_samples) * 0.1
            model_a = np.sin(2 * np.pi * x)  # 真模型
            model_b = np.sin(2 * np.pi * x) + effect_size * 2  # 错误模型

            payoffs = self.compute_payoffs(data, model_a, model_b, 0.1)
            _, _, tau = self.wealth_process(payoffs)

            if tau < n_samples:
                rejections += 1

        power = rejections / n_trials

        return {
            'power': power,
            'effect_size': effect_size,
            'n_samples': n_samples,
            'n_trials': n_trials,
            'rejection_rate': power,
        }


class OpticalPotentialModelFactory:
    """
    光学势模型工厂 — 生成不同参数化形式的势

    支持:
    1. Woods-Saxon (标准)
    2. 方势阱 (Square Well)
    3. 折叠高斯 (Folded Gaussian)
    4. 二参数费米 (2pF)
    """

    @staticmethod
    def woods_saxon(r: np.ndarray, V0: float, R: float, a: float) -> np.ndarray:
        """标准 Woods-Saxon 势: V(r) = -V0 / (1 + exp((r-R)/a))"""
        x = (r - R) / a
        x = np.clip(x, -500, 500)
        return -V0 / (1.0 + np.exp(x))

    @staticmethod
    def square_well(r: np.ndarray, V0: float, R: float, diff: float = 0.5) -> np.ndarray:
        """
        扩散方势阱:
            V(r) = -V0                              r < R
            V(r) = -V0 * exp(-(r-R)/diff)           r >= R
        """
        V = np.zeros_like(r)
        mask = r < R
        V[mask] = -V0
        V[~mask] = -V0 * np.exp(-(r[~mask] - R) / diff)
        return V

    @staticmethod
    def folded_gaussian(r: np.ndarray, V0: float, R: float, sigma: float) -> np.ndarray:
        """
        折叠高斯势:
            V(r) = -V0 * integral rho(r') * exp(-(r-r')^2/sigma^2) dr'
            近似为: V(r) = -V0 * [exp(-(r-R)^2/sigma^2) + exp(-(r+R)^2/sigma^2)]
        """
        return -V0 * (np.exp(-(r - R) ** 2 / sigma ** 2) +
                       np.exp(-(r + R) ** 2 / sigma ** 2))

    @staticmethod
    def two_parameter_fermi(r: np.ndarray, V0: float, c: float, z_param: float) -> np.ndarray:
        """
        二参数费米分布:
            rho(r) = 1 / (1 + exp((r-c)/z))
            V(r) = -V0 * rho(r)
        (与 Woods-Saxon 等价, 参数命名不同)
        """
        x = (r - c) / z_param
        x = np.clip(x, -500, 500)
        return -V0 / (1.0 + np.exp(x))

    @classmethod
    def generate_model_prediction(
        cls,
        model_type: str,
        r: np.ndarray,
        params: Dict
    ) -> np.ndarray:
        """根据模型类型生成势的预测"""
        if model_type == 'woods_saxon':
            return cls.woods_saxon(r, params['V0'], params['R'], params['a'])
        elif model_type == 'square_well':
            return cls.square_well(r, params['V0'], params['R'], params.get('diff', 0.5))
        elif model_type == 'folded_gaussian':
            return cls.folded_gaussian(r, params['V0'], params['R'], params['sigma'])
        elif model_type == 'two_parameter_fermi':
            return cls.two_parameter_fermi(r, params['V0'], params['c'], params['z'])
        else:
            raise ValueError(f"未知模型类型: {model_type}")
