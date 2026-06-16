"""
multifidelity.py — 多保真度不确定性量化模块
===============================================
实现多保真度控制变量方法, 结合不同精度的模型
进行高效的不确定性传播。

核心公式:
  多保真度估计器:
    μ_MF = μ_LF + (1/N_H) Σ_{i=1}^{N_H} [f_H(ξ_i) - f_L(ξ_i)]

  方差缩减:
    Var[μ_MF] = (1/N_H) Var[f_H - f_L] + (1/N_L) Var[f_L]

  最优样本分配:
    N_L / N_H = √(C_H / C_L) * √(Var[f_L] / Var[f_H - f_L])

  控制变量系数:
    β* = Cov[f_H, f_L] / Var[f_L]

  非线性多保真度:
    f_MF(ξ) = f_L(ξ) + NN(f_H(ξ) - f_L(ξ); θ)

映射种子项目:
  - 1246_Gaulios: 有限-无穷horizon桥接 → 多保真度桥接
  - 1180_Yuva12345: MCMC + 贝叶斯证据 → 多模型融合
  - 1050_LaM-SLidE: 编码器-解码器 → 跨保真度映射
"""

import numpy as np
from typing import Tuple, Optional, Callable, Dict, List
from config import GlobalConfig, MultiFidelityConfig
from neural_closure import SimpleMLP


class MultiFidelityController:
    """
    多保真度控制器。

    管理不同精度模型之间的信息融合,
    以最低计算成本达到目标精度。

    映射 1246_Gaulios: 有限-无穷horizon的耦合求解
    推广为高低保真度模型的耦合。

    映射 1180_Yuva12345: 多模型贝叶斯证据计算
    推广为多保真度模型选择。
    """

    def __init__(self, config: GlobalConfig):
        self.config = config
        self.mf_config = config.multifidelity

        self.high_fidelity_results: List[float] = []
        self.low_fidelity_results: List[float] = []
        self.correction_model: Optional[SimpleMLP] = None

    def estimate_control_variate(self,
                                 f_high: np.ndarray,
                                 f_low: np.ndarray
                                 ) -> Tuple[float, float, float]:
        """
        控制变量估计器。

        μ_CV = E[f_L] + β* (E[f_H] - E[f_L_matched])

        β* = Cov[f_H, f_L] / Var[f_L]

        参数:
            f_high: 高保真度结果, shape (N_H,)
            f_low:  低保真度结果, shape (N_L,)

        返回:
            mu_cv:   控制变量估计的均值
            var_cv:  估计方差
            beta_star: 最优控制变量系数
        """
        N_H = len(f_high)
        N_L = len(f_low)

        if N_H == 0 or N_L == 0:
            return 0.0, 1.0, 0.0

        # 均值
        mu_H = np.mean(f_high)
        mu_L = np.mean(f_low)

        # 控制变量系数
        # 需要配对样本; 简化: 使用各自统计量
        var_L = np.var(f_low)
        if var_L < 1e-30:
            return mu_H, np.var(f_high) / N_H, 0.0

        # 估计 Cov[f_H, f_L] (假设正相关)
        # 使用 min(N_H, N_L) 配对
        n_pair = min(N_H, N_L)
        cov_HL = np.mean(
            (f_high[:n_pair] - mu_H) * (f_low[:n_pair] - mu_L))
        beta_star = cov_HL / var_L

        # 控制变量估计
        mu_cv = mu_L + beta_star * (mu_H - mu_L)

        # 方差估计
        if n_pair > 1:
            diff = f_high[:n_pair] - f_low[:n_pair]
            var_diff = np.var(diff)
            var_cv = var_diff / N_H + var_L / N_L
        else:
            var_cv = np.var(f_high) / max(N_H, 1)

        return float(mu_cv), float(max(var_cv, 0.0)), float(beta_star)

    def optimal_sample_allocation(self,
                                  cost_high: float,
                                  cost_low: float,
                                  var_high: float,
                                  var_diff: float,
                                  total_budget: float
                                  ) -> Tuple[int, int]:
        """
        最优样本分配。

        在总计算预算 C 下, 最小化估计方差:
          min Var[μ_MF]
          s.t. N_H * C_H + N_L * C_L ≤ C

        最优解:
          N_H = √(C * Var_diff / (C_H * (C_H + C_L * r)))
          N_L = r * N_H
          r = √(C_H * Var_L / (C_L * Var_diff))

        参数:
            cost_high:  单次高保真成本
            cost_low:   单次低保真成本
            var_high:   高保真方差
            var_diff:   差值方差 Var[f_H - f_L]
            total_budget: 总预算

        返回:
            N_H, N_L: 各保真度的样本数
        """
        if var_diff < 1e-30 or cost_high < 1e-30:
            return self.mf_config.n_high_fidelity, 0

        # 最优比率
        r = np.sqrt(cost_high * var_high /
                     max(cost_low * var_diff, 1e-30))
        r = max(r, 1.0)

        # 总成本: N_H * C_H + N_L * C_L = N_H * (C_H + r * C_L)
        effective_cost = cost_high + r * cost_low
        if effective_cost < 1e-30:
            return self.mf_config.n_high_fidelity, 0

        N_H = int(np.sqrt(total_budget * var_diff /
                           (cost_high * effective_cost)))
        N_H = max(N_H, 2)

        N_L = int(r * N_H)
        N_L = max(N_L, N_H)

        return N_H, N_L

    def compute_correlation(self,
                            f_high: np.ndarray,
                            f_low: np.ndarray) -> float:
        """
        计算高低保真度之间的相关系数。

        ρ = Cov[f_H, f_L] / √(Var[f_H] * Var[f_L])

        高相关性 (> 0.9) 意味着多保真度方法有效。
        """
        n = min(len(f_high), len(f_low))
        if n < 2:
            return 0.0

        fh = f_high[:n]
        fl = f_low[:n]

        cov = np.mean((fh - np.mean(fh)) * (fl - np.mean(fl)))
        std_h = np.std(fh)
        std_l = np.std(fl)

        if std_h < 1e-15 or std_l < 1e-15:
            return 0.0

        return float(cov / (std_h * std_l))

    def train_correction_model(self,
                               xi_samples: np.ndarray,
                               f_high: np.ndarray,
                               f_low: np.ndarray
                               ) -> List[float]:
        """
        训练非线性修正模型 (映射 1050_LaM-SLidE 的编码器思想)。

        f_corrected(ξ) = f_L(ξ) + NN(f_L(ξ); θ)

        目标: min ||f_H - f_corrected||^2

        参数:
            xi_samples: shape (N, d), 样本点
            f_high:     shape (N,), 高保真结果
            f_low:      shape (N,), 低保真结果

        返回:
            losses: 训练损失历史
        """
        n = len(f_high)
        diff = f_high - f_low  # 需要学习的修正量

        # 构建网络: 输入 = PCE 系数或随机变量
        input_dim = xi_samples.shape[1] if xi_samples.ndim > 1 else 1
        hidden = self.config.neural_closure.hidden_dim

        self.correction_model = SimpleMLP(
            [input_dim, hidden, hidden, 1],
            activation="tanh",
            seed=42,
        )

        lr = self.config.neural_closure.learning_rate
        losses = []

        for epoch in range(self.config.neural_closure.n_epochs):
            # 随机小批量
            batch_idx = np.random.choice(
                n, min(32, n), replace=False)
            batch_x = xi_samples[batch_idx]
            if batch_x.ndim == 1:
                batch_x = batch_x.reshape(-1, 1)
            batch_y = diff[batch_idx].reshape(-1, 1)

            # 前向
            pred = self.correction_model.forward(batch_x)
            loss = float(np.mean((pred - batch_y) ** 2))
            losses.append(loss)

            # 反向
            grad = 2.0 * (pred - batch_y) / batch_y.size
            grad_W, grad_b = self.correction_model.backward(grad)
            self.correction_model.update(
                grad_W, grad_b, lr,
                self.config.neural_closure.regularization)

        return losses

    def predict_corrected(self, xi: np.ndarray,
                          f_low_value: float) -> float:
        """
        使用修正模型预测高保真值。

        f̂_H(ξ) = f_L(ξ) + NN(ξ; θ)
        """
        if self.correction_model is None:
            return f_low_value

        x = xi.reshape(1, -1) if xi.ndim > 0 else np.array([[xi]])
        correction = float(self.correction_model.forward(x)[0, 0])
        return f_low_value + correction


def run_multifidelity_analysis(config: GlobalConfig,
                               high_fidelity_func: Callable,
                               low_fidelity_func: Callable
                               ) -> dict:
    """
    运行多保真度分析的完整流程。

    参数:
        config:           全局配置
        high_fidelity_func: 高保真度函数
        low_fidelity_func:  低保真度函数

    返回:
        results: 结果字典
    """
    controller = MultiFidelityController(config)
    d = len(config.measures)

    rng = np.random.RandomState(42)

    # 生成样本点
    N_H = config.multifidelity.n_high_fidelity
    N_L = config.multifidelity.n_low_fidelity

    # 从测度的支撑集采样
    xi_H = np.zeros((N_H, d))
    xi_L = np.zeros((N_L, d))
    for dim_idx, meas in enumerate(config.measures):
        support = meas.support
        xi_H[:, dim_idx] = rng.uniform(
            support[0], support[1], N_H)
        xi_L[:, dim_idx] = rng.uniform(
            support[0], support[1], N_L)

    # 评估
    f_H = np.array([high_fidelity_func(xi_H[i]) for i in range(N_H)])
    f_L_all = np.array([low_fidelity_func(xi_L[i]) for i in range(N_L)])

    # 在高保真样本点也评估低保真模型 (用于修正模型训练)
    f_L_at_H = np.array([low_fidelity_func(xi_H[i]) for i in range(N_H)])

    # 控制变量估计
    mu_cv, var_cv, beta = controller.estimate_control_variate(f_H, f_L_all)

    # 相关系数
    rho = controller.compute_correlation(f_H, f_L_all)

    # 训练修正模型 (使用配对的样本点)
    losses = controller.train_correction_model(xi_H, f_H, f_L_at_H)

    # 标准 MC 估计对比
    mu_mc = np.mean(f_H)
    var_mc = np.var(f_H) / N_H

    return {
        "control_variate_mean": mu_cv,
        "control_variate_var": var_cv,
        "control_variate_beta": beta,
        "mc_mean": mu_mc,
        "mc_var": var_mc,
        "correlation": rho,
        "variance_reduction": var_mc / max(var_cv, 1e-30),
        "n_high": N_H,
        "n_low": N_L,
        "correction_losses": losses,
    }
