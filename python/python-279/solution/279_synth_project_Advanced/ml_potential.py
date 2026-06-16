"""
ml_potential.py
===============
机器学习代理势模块 (源自 1002_sampk1203 ML-accelerated MD of Li-ion in LLZO)。

科学背景:
  分子动力学模拟需要精确的原子间势, 但第一性原理计算 (DFT) 昂贵。
  机器学习势 (MLP) 通过学习 DFT 数据构建快速代理模型。

  本模块实现:
    1. 描述符生成 (原子环境特征化)
    2. 简单线性/核岭回归势模型
    3. 离子电导率预测

  物理基础:
    离子电导率的 Arrhenius 形式:
      sigma(T) = (sigma_0 / T) * exp(-E_a / (kB*T))

    迁移势垒 E_a 与局部结构的关系 (经验):
      E_a ≈ alpha * (d_Li-O / d_0)^beta + gamma * vol_free

    其中 d_Li-O 为 Li-O 键长, vol_free 为自由体积

  核岭回归 (KRR):
    alpha = (K + lambda*I)^{-1} * y
    K_{ij} = k(x_i, x_j)  (核矩阵)
    核函数: k(x, y) = exp(-||x-y||^2 / (2*sigma^2))  (Gaussian/RBF)

  均方根误差 (RMSE):
    RMSE = sqrt(mean((y_pred - y_true)^2))

  决定系数 R^2:
    R^2 = 1 - sum((y_pred - y_true)^2) / sum((y_true - mean(y_true))^2)
"""

import numpy as np
from material_constants import (
    KB, ELEMENTARY_CHARGE, LLZO_IONIC_CONDUCTIVITY, LLZO_ACTIVATION_ENERGY,
    SMALL_NUMBER, DOPANT_CONCENTRATIONS, TEMPERATURE_RANGE,
)


class GaussianKernelRegressor:
    """
    高斯核岭回归模型。

    核函数:
      k(x, y) = sigma_f^2 * exp(-||x-y||^2 / (2*l^2))

    预测:
      f(x*) = k(x*, X)^T * (K + lambda*I)^{-1} * y

    训练:
      alpha = (K + lambda*I)^{-1} * y
    """
    def __init__(self, length_scale=1.0, sigma_f=1.0, lambda_reg=1e-3):
        self.length_scale = length_scale
        self.sigma_f = sigma_f
        self.lambda_reg = lambda_reg
        self.X_train = None
        self.alpha = None

    def _kernel(self, X1, X2):
        """RBF 核矩阵"""
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        sq_dist = np.sum(X1**2, axis=1, keepdims=True) + np.sum(X2**2, axis=1) - 2*X1@X2.T
        sq_dist = np.maximum(sq_dist, 0.0)
        return self.sigma_f**2 * np.exp(-sq_dist / (2*self.length_scale**2))

    def fit(self, X, y):
        """训练模型"""
        self.X_train = np.atleast_2d(X).copy()
        K = self._kernel(self.X_train, self.X_train)
        n = K.shape[0]
        self.alpha = np.linalg.solve(K + self.lambda_reg * np.eye(n), y)
        self.y_train_mean = np.mean(y)
        self.y_train_std = max(np.std(y), SMALL_NUMBER)

    def predict(self, X):
        """预测"""
        X = np.atleast_2d(X)
        K_star = self._kernel(X, self.X_train)
        y_pred = K_star @ self.alpha
        return y_pred

    def score(self, X, y):
        """R^2 评分"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        return 1.0 - ss_res / max(ss_tot, SMALL_NUMBER)


def generate_llzo_training_data(n_samples=100, seed=42):
    """
    生成 LLZO 离子电导率训练数据 (模拟 DFT/MD 数据)。

    基于物理模型:
      sigma = sigma_0 * exp(-E_a(x, eps) / (kB*T))
    其中:
      E_a(x, eps) = E_a0 + a1*x + a2*x^2 + b1*eps + b2*eps^2 + c*x*eps
      x: 掺杂浓度, eps: 应变

    添加高斯噪声模拟计算误差。
    """
    rng = np.random.RandomState(seed)

    # 随机采样 (掺杂浓度, 应变, 温度)
    x_dopant = rng.uniform(0.0, 0.3, n_samples)
    strain = rng.uniform(-0.02, 0.02, n_samples)
    temperature = rng.choice(TEMPERATURE_RANGE, n_samples)

    # 物理模型: 活化能与掺杂/应变的关系
    E_a0 = LLZO_ACTIVATION_ENERGY  # 0.37 eV
    # 掺杂效应: 最优掺杂 ~0.15
    a1, a2 = -0.5, 2.0
    # 应变效应: 压应变降低势垒
    b1, b2 = 3.0, 50.0
    # 耦合
    c = -10.0

    E_a = E_a0 + a1 * x_dopant + a2 * x_dopant**2 + b1 * strain + b2 * strain**2 + c * x_dopant * strain
    E_a = np.maximum(E_a, 0.1)  # 物理约束

    # 电导率 (Arrhenius)
    sigma_0 = LLZO_IONIC_CONDUCTIVITY * np.exp(E_a0 * ELEMENTARY_CHARGE / (KB * 300.0))
    kB_eV = KB / ELEMENTARY_CHARGE
    sigma = sigma_0 * np.exp(-E_a / (kB_eV * temperature))

    # 添加噪声 (10% 相对误差)
    noise = rng.normal(0, 0.1, n_samples)
    sigma_noisy = sigma * (1.0 + noise)
    sigma_noisy = np.maximum(sigma_noisy, SMALL_NUMBER)

    features = np.column_stack([x_dopant, strain, temperature])
    targets = np.log10(sigma_noisy)  # 对数空间建模

    return features, targets, {'x_dopant': x_dopant, 'strain': strain, 'temperature': temperature, 'E_a': E_a, 'sigma': sigma_noisy}


def generate_band_gap_data(n_samples=80, seed=123):
    """
    生成带隙预测训练数据 (材料基因组)。

    特征: 组成描述符 (8维)
    目标: 带隙 [eV]
    """
    rng = np.random.RandomState(seed)
    from crystal_descriptor import composition_descriptor, ATOMIC_NUMBERS

    compounds = [
        {'Li': 7, 'La': 3, 'Zr': 2, 'O': 12},
        {'Li': 5, 'La': 3, 'Zr': 2, 'O': 12, 'Al': 1},
        {'Li': 4, 'La': 3, 'Zr': 2, 'O': 12, 'Al': 1.5},
        {'Li': 7, 'La': 3, 'Zr': 1.5, 'Ta': 0.5, 'O': 12},
        {'Li': 7, 'La': 2.5, 'Y': 0.5, 'Zr': 2, 'O': 12},
    ]

    features = []
    band_gaps = []
    for comp in compounds:
        desc = composition_descriptor(comp)
        for _ in range(n_samples // len(compounds)):
            # 微小扰动
            desc_perturbed = desc + rng.normal(0, 0.01, len(desc))
            features.append(desc_perturbed)
            # 简化物理模型: 带隙 ~ f(电负性差)
            avg_en = desc_perturbed[2]
            var_en = desc_perturbed[3]
            Eg = 2.5 + 0.5 * avg_en - 0.2 * var_en + rng.normal(0, 0.1)
            band_gaps.append(max(Eg, 0.1))

    return np.array(features), np.array(band_gaps)


def arrhenius_fit(temperatures, conductivities):
    """
    Arrhenius 拟合求活化能:

    log(sigma*T) = log(sigma_0) - E_a / (kB * T)

    线性回归:
      y = A + B * (1/T)
      A = log(sigma_0), B = -E_a / kB

    返回: (E_a_eV, sigma_0, R_squared)
    """
    T = np.asarray(temperatures, dtype=np.float64)
    sigma = np.asarray(conductivities, dtype=np.float64)
    mask = (T > 0) & (sigma > 0)
    T = T[mask]
    sigma = sigma[mask]

    y = np.log(sigma * T)
    x = 1.0 / T
    # 线性回归
    A_mat = np.column_stack([np.ones_like(x), x])
    coeffs, residuals, _, _ = np.linalg.lstsq(A_mat, y, rcond=None)
    A, B = coeffs

    kB_eV = KB / ELEMENTARY_CHARGE
    E_a_eV = -B * kB_eV
    sigma_0 = np.exp(A)

    # R^2
    y_pred = A_mat @ coeffs
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    R2 = 1.0 - ss_res / max(ss_tot, SMALL_NUMBER)

    return E_a_eV, sigma_0, R2


def predict_optimal_doping(ml_model, strain=0.0, temperature=300.0, doping_range=None):
    """
    使用 ML 模型预测最优掺杂浓度。

    对 doping_range 中每个浓度预测 sigma, 找最大值。
    """
    if doping_range is None:
        doping_range = DOPANT_CONCENTRATIONS

    X_pred = np.column_stack([
        doping_range,
        np.full_like(doping_range, strain),
        np.full_like(doping_range, temperature),
    ])
    log_sigma = ml_model.predict(X_pred)
    sigma = 10.0 ** log_sigma
    opt_idx = np.argmax(sigma)
    return doping_range[opt_idx], sigma[opt_idx], sigma


def cross_validate(X, y, n_folds=5, seed=42):
    """
    K-fold 交叉验证。

    返回: (mean_RMSE, std_RMSE, mean_R2)
    """
    rng = np.random.RandomState(seed)
    n = len(y)
    indices = rng.permutation(n)
    fold_size = n // n_folds

    rmse_list = []
    r2_list = []
    for k in range(n_folds):
        val_idx = indices[k*fold_size:(k+1)*fold_size]
        train_idx = np.concatenate([indices[:k*fold_size], indices[(k+1)*fold_size:]])
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        model = GaussianKernelRegressor(length_scale=1.0, sigma_f=1.0, lambda_reg=1e-2)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        rmse = np.sqrt(np.mean((y_val - y_pred)**2))
        ss_res = np.sum((y_val - y_pred)**2)
        ss_tot = np.sum((y_val - np.mean(y_val))**2)
        r2 = 1.0 - ss_res / max(ss_tot, SMALL_NUMBER)
        rmse_list.append(rmse)
        r2_list.append(r2)

    return np.mean(rmse_list), np.std(rmse_list), np.mean(r2_list)
