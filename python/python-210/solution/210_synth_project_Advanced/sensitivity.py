"""
sensitivity.py - 灵敏度分析与稀疏识别：Sobol指数与压缩感知

本模块实现全局灵敏度分析的多种方法，用于识别对失效概率
贡献最大的随机输入变量。

1. Sobol 指数 (ANOVA 分解):
   一阶: S_i = Var_{u_i}[E_{u_{~i}}[Y|u_i]] / Var[Y]
   全阶: S_Ti = E_{u_{~i}}[Var_{u_i}[Y|u_{~i}]] / Var[Y]
   S_i 衡量第 i 个变量的独立贡献,
   S_Ti 衡量包括交互效应在内的总贡献。

2. 压缩感知 (L1 稀疏正则化):
   当灵敏度向量稀疏时 (只有少数变量重要),
   可用 L1 正则化高效识别重要变量:
   min ||s||_1  s.t.  ||A*s - y||_2 <= epsilon

3. Pearson 相关系数:
   rho_i = Cov(X_i, I_fail) / (sigma_Xi * sigma_fail)

种子项目映射:
  223_counterfeit_detection → 压缩感知 L1 最小化
  345_exm (pagerank) → 转移矩阵灵敏度
  1165_bioinfo_GEMS → 数据驱动灵敏度
"""

import numpy as np


class SobolSensitivity:
    """Sobol 全局灵敏度指数

    基于 Saltelli 采样方案的高效估计:
    生成两组独立样本 A, B (各 N 个),
    然后构造 AB_i (B 的第 i 列替换 A 的第 i 列),
    通过函数评估计算灵敏度指数。

    Jansen 估计量 (一阶):
      V_i = (1/(2N)) sum (f(B) - f(AB_i))^2
      S_i = V_i / Var[Y]

    Jansen 估计量 (全阶):
      VT_i = (1/(2N)) sum (f(A) - f(AB_i))^2
      S_Ti = VT_i / Var[Y]
    """

    def __init__(self, func, n_dim, n_samples=2000, rng_seed=42):
        self.func = func
        self.n_dim = n_dim
        self.n_samples = n_samples
        self.rng = np.random.default_rng(rng_seed)

    def compute(self):
        """计算一阶和全阶 Sobol 指数

        Returns
        -------
        S1 : np.ndarray (n_dim,), 一阶指数
        ST : np.ndarray (n_dim,), 全阶指数
        """
        N = self.n_samples
        d = self.n_dim

        # 生成两组独立样本 (标准正态空间)
        A = self.rng.standard_normal((N, d))
        B = self.rng.standard_normal((N, d))

        # 评估
        fA = np.array([self.func(A[i]) for i in range(N)])
        fB = np.array([self.func(B[i]) for i in range(N)])

        # 处理 NaN
        fA = np.nan_to_num(fA, nan=0.0)
        fB = np.nan_to_num(fB, nan=0.0)

        var_total = np.var(np.concatenate([fA, fB]))
        if var_total < 1e-30:
            return np.zeros(d), np.zeros(d)

        S1 = np.zeros(d)
        ST = np.zeros(d)

        for i in range(d):
            # AB_i: B 的所有列, 但第 i 列来自 A
            AB_i = B.copy()
            AB_i[:, i] = A[:, i]
            fAB_i = np.array([self.func(AB_i[j]) for j in range(N)])
            fAB_i = np.nan_to_num(fAB_i, nan=0.0)

            # Jansen 估计量
            V_i = np.mean((fB - fAB_i) ** 2)
            VT_i = np.mean((fA - fAB_i) ** 2)

            S1[i] = V_i / var_total
            ST[i] = VT_i / var_total

        return np.clip(S1, 0.0, 1.0), np.clip(ST, 0.0, 1.0)


class CompressedSensingSensitivity:
    """基于压缩感知的稀疏灵敏度识别

    种子项目 223_counterfeit_detection 的扩展:
    使用随机投影 ( sensing matrix ) 和 L1 最小化
    从少量观测中恢复稀疏的灵敏度向量。

    数学模型:
      y = Phi * s + noise
      min ||s||_1  s.t.  ||Phi*s - y||_2 <= epsilon

    其中:
      s = 灵敏度向量 (大部分分量为零)
      Phi =  sensing 矩阵 (随机 0/1)
      y = 观测向量 (通过扰动实验获得)

    理论基础:
      若 s 是 K-稀疏的, 则 m >= C*K*log(n/K) 次观测
      即可通过 L1 最小化精确恢复 s。
    """

    def __init__(self, n_dim, n_measurements=None, rng_seed=42):
        self.n_dim = n_dim
        self.n_meas = n_measurements or max(2 * n_dim, 10)
        self.rng = np.random.default_rng(rng_seed)
        self.sensing_matrix = None

    def build_sensing_matrix(self):
        """构建随机 sensing 矩阵 Phi (稀疏 0/1 矩阵)

        种子项目 223 中的 phi = randi([0,1], k, n)
        """
        self.sensing_matrix = self.rng.integers(0, 2,
                                                  size=(self.n_meas, self.n_dim)).astype(float)
        return self.sensing_matrix

    def l1_sensitivity_recovery(self, observations, epsilon=0.01):
        """L1 最小化恢复稀疏灵敏度

        使用迭代软阈值算法 (ISTA):
          s^{k+1} = soft_threshold(s^k - mu * Phi^T * (Phi*s^k - y), mu*lambda)

        其中 soft_threshold(z, t) = sign(z) * max(|z| - t, 0)
        """
        if self.sensing_matrix is None:
            self.build_sensing_matrix()

        Phi = self.sensing_matrix
        y = np.asarray(observations, dtype=float)
        n = self.n_dim
        m = len(y)

        # ISTA 参数
        L = np.linalg.norm(Phi.T @ Phi, 2)  # Lipschitz 常数
        mu = 0.9 / L if L > 1e-15 else 1.0
        lam = epsilon * 0.1  # L1 正则化参数

        s = np.zeros(n)
        for _ in range(500):
            residual = Phi @ s - y
            grad = Phi.T @ residual
            s_half = s - mu * grad
            # 软阈值
            s_new = np.sign(s_half) * np.maximum(np.abs(s_half) - mu * lam, 0.0)
            if np.linalg.norm(s_new - s) < 1e-10:
                break
            s = s_new

        return s

    def finite_difference_sensitivity(self, lsf, u_center=None, delta=0.1):
        """通过有限差分获取灵敏度观测

        在 u_center 附近扰动每个维度, 记录 g 的变化。
        """
        if u_center is None:
            u_center = np.zeros(self.n_dim)

        g0 = float(np.atleast_1d(lsf.evaluate(u_center.reshape(1, -1)))[0])
        observations = []

        for _ in range(self.n_meas):
            # 随机扰动方向
            direction = self.rng.standard_normal(self.n_dim)
            direction /= np.linalg.norm(direction) + 1e-30

            u_pert = u_center + delta * direction
            g_pert = float(np.atleast_1d(lsf.evaluate(u_pert.reshape(1, -1)))[0])
            observations.append((g_pert - g0) / delta)

        return np.array(observations)


class PearsonCorrelation:
    """Pearson 相关系数灵敏度

    rho_i = Cov(X_i, Y) / (sigma_Xi * sigma_Y)

    对失效指标 Y = I(g(u)<=0):
    rho_i 衡量 u_i 与失效事件的相关性。
    """

    @staticmethod
    def compute(u_samples, g_values):
        """计算输入-输出相关系数

        Parameters
        ----------
        u_samples : np.ndarray (N, d)
        g_values : np.ndarray (N,)

        Returns
        -------
        rho : np.ndarray (d,)
        """
        u = np.atleast_2d(u_samples)
        g = np.asarray(g_values, dtype=float).ravel()
        d = u.shape[1]
        rho = np.zeros(d)

        g_centered = g - np.mean(g)
        g_std = np.std(g)
        if g_std < 1e-15:
            return rho

        for i in range(d):
            u_centered = u[:, i] - np.mean(u[:, i])
            u_std = np.std(u[:, i])
            if u_std < 1e-15:
                rho[i] = 0.0
            else:
                rho[i] = np.mean(u_centered * g_centered) / (u_std * g_std)

        return rho


class TornadoSensitivity:
    """龙卷风图灵敏度 (OAT - One At a Time)

    逐个扰动每个输入变量 (固定其他变量在标称值),
    观察输出的变化范围, 按变化幅度排序绘制龙卷风图。

    Delta_i = |g(u_0 + delta*e_i) - g(u_0 - delta*e_i)| / 2
    """

    @staticmethod
    def compute(lsf, n_dim, delta=1.0, u0=None):
        if u0 is None:
            u0 = np.zeros(n_dim)

        g0 = float(np.atleast_1d(lsf.evaluate(u0.reshape(1, -1)))[0])
        sensitivities = np.zeros(n_dim)

        for i in range(n_dim):
            u_plus = u0.copy(); u_plus[i] += delta
            u_minus = u0.copy(); u_minus[i] -= delta
            g_plus = float(np.atleast_1d(lsf.evaluate(u_plus.reshape(1, -1)))[0])
            g_minus = float(np.atleast_1d(lsf.evaluate(u_minus.reshape(1, -1)))[0])
            sensitivities[i] = abs(g_plus - g_minus) / 2.0

        return sensitivities
