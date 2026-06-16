#!/usr/bin/env python3
"""
ml_surrogate.py
===============
机器学习代理模型模块，融合种子项目:
  - [1215] riskychoice: SVM 分类器, 交叉验证, 数据管道
  - [1002] LLZO MD: 弹性张量计算, BFGS 优化

物理背景:
  ICF 内爆模拟的计算代价极高 (3D 辐射流体力学)。
  使用 ML 代理模型快速预测:
    1. 内爆对称性 → 给定驱动不对称, 预测 P₂/P₄
    2. 不稳定性阈值 → 给定参数空间, 判断是否 RT 不稳定
    3. 产额预测 → 给定内爆参数, 预测中子产额

  特征向量:
    x = [驱动不对称度, 靶丸粗糙度, 内爆速度, 燃料ρR, 激光功率失衡, ...]
  输出:
    y = [P₂ 不对称度, P₄ 不对称度, 是否失稳, log₁₀(产额)]
"""

import numpy as np


class FeatureScaler:
    """
    特征标准化 (融合 [1215] sklearn.preprocessing)。
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        """计算均值和标准差"""
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ < 1e-14] = 1.0  # 防止除零
        return self

    def transform(self, X):
        """标准化: (X - mean) / std"""
        return (np.asarray(X) - self.mean_) / self.std_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_scaled):
        return X_scaled * self.std_ + self.mean_


class SVMClassifier:
    """
    支持向量机分类器 (融合 [1215] sklearn.svm)。
    简化版线性 SVM (SGD 训练)。

    用于 ICF 不稳定性阈值分类:
      输入: 内爆参数
      输出: +1 (稳定) / -1 (不稳定)
    """

    def __init__(self, C=1.0, learning_rate=0.01, n_iter=1000, tol=1e-4):
        self.C = C
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.tol = tol
        self.w = None
        self.b = 0.0

    def fit(self, X, y):
        """
        训练 SVM (Pegasos SGD 算法)。
        min (1/2)||w||² + C Σ max(0, 1 - y_i(w·x_i + b))
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        N, D = X.shape

        self.w = np.zeros(D)
        self.b = 0.0

        for t in range(1, self.n_iter + 1):
            eta = self.learning_rate / (1.0 + 0.01 * t)

            for i in range(N):
                margin = y[i] * (np.dot(X[i], self.w) + self.b)
                if margin < 1.0:
                    self.w = (1 - eta) * self.w + eta * self.C * y[i] * X[i]
                    self.b += eta * self.C * y[i]
                else:
                    self.w = (1 - eta) * self.w

        return self

    def predict(self, X):
        """预测类别"""
        scores = np.dot(np.asarray(X), self.w) + self.b
        return np.sign(scores)

    def decision_function(self, X):
        """决策函数值"""
        return np.dot(np.asarray(X), self.w) + self.b

    def accuracy(self, X, y):
        """计算分类精度"""
        preds = self.predict(X)
        return np.mean(preds == np.sign(y))


class CrossValidator:
    """
    K 折交叉验证 (融合 [1215] sklearn.model_selection)。
    """

    @staticmethod
    def k_fold_split(N, k=5, seed=42):
        """
        生成 K 折交叉验证的索引划分。
        返回: list of (train_idx, val_idx)
        """
        rng = np.random.default_rng(seed)
        indices = rng.permutation(N)
        fold_size = N // k
        splits = []

        for fold in range(k):
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < k - 1 else N
            val_idx = indices[val_start:val_end]
            train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
            splits.append((train_idx, val_idx))

        return splits

    @staticmethod
    def cross_validate(classifier, X, y, k=5):
        """
        K 折交叉验证。
        返回: (mean_accuracy, std_accuracy)
        """
        splits = CrossValidator.k_fold_split(len(y), k)
        accuracies = []

        for train_idx, val_idx in splits:
            clf = SVMClassifier(C=classifier.C, n_iter=classifier.n_iter)
            clf.fit(X[train_idx], y[train_idx])
            acc = clf.accuracy(X[val_idx], y[val_idx])
            accuracies.append(acc)

        return np.mean(accuracies), np.std(accuracies)


class ElasticTensorAnalyzer:
    """
    弹性张量分析 (融合 [1002] get_elastic_moduli)。
    用于计算 ICF 靶丸材料的力学性质。

    弹性刚度矩阵 C_ij (Voigt 记号):
      σ_i = C_ij ε_j  (i,j = 1..6)
    对于立方晶系:
      C = [[C11, C12, C12, 0, 0, 0],
           [C12, C11, C12, 0, 0, 0],
           [C12, C12, C11, 0, 0, 0],
           [0, 0, 0, C44, 0, 0],
           [0, 0, 0, 0, C44, 0],
           [0, 0, 0, 0, 0, C44]]

    体弹模量: K = (C11 + 2 C12) / 3
    剪切模量: G = (C11 - C12 + 3 C44) / 5
    杨氏模量: E = 9KG / (3K + G)
    泊松比: ν = (3K - 2G) / (6K + 2G)
    """

    @staticmethod
    def cubic_moduli(C11, C12, C44):
        """
        由立方晶系弹性常数计算工程模量。
        """
        K = (C11 + 2 * C12) / 3.0  # 体弹模量 (Voigt)
        G = (C11 - C12 + 3 * C44) / 5.0  # 剪切模量 (Voigt)

        # Reuss 平均
        S11 = (C11 + C12) / ((C11 - C12) * (C11 + 2 * C12) + 1e-30)
        S12 = -C12 / ((C11 - C12) * (C11 + 2 * C12) + 1e-30)
        S44 = 1.0 / (C44 + 1e-30)
        K_R = 1.0 / (3 * (S11 + 2 * S12) + 1e-30)
        G_R = 5.0 / (4 * S11 - 4 * S12 + 3 * S44 + 1e-30)

        # Voigt-Reuss-Hill 平均
        K_VRH = (K + K_R) / 2.0
        G_VRH = (G + G_R) / 2.0

        E = 9 * K_VRH * G_VRH / (3 * K_VRH + G_VRH + 1e-30)
        nu = (3 * K_VRH - 2 * G_VRH) / (6 * K_VRH + 2 * G_VRH + 1e-30)

        return {
            'K_Voigt': K, 'G_Voigt': G,
            'K_Reuss': K_R, 'G_Reuss': G_R,
            'K_VRH': K_VRH, 'G_VRH': G_VRH,
            'Young_modulus': E,
            'Poisson_ratio': nu,
            'C11': C11, 'C12': C12, 'C44': C44
        }

    @staticmethod
    def check_mechanical_stability(C11, C12, C44):
        """
        Born 力学稳定性判据 (立方晶系):
          C11 > |C12|
          C11 + 2 C12 > 0
          C44 > 0
        """
        stable = True
        checks = {}
        checks['C11 > |C12|'] = C11 > abs(C12)
        checks['C11 + 2*C12 > 0'] = (C11 + 2 * C12) > 0
        checks['C44 > 0'] = C44 > 0
        stable = all(checks.values())
        return stable, checks


class ICFSurrogateModel:
    """
    ICF 内爆 ML 代理模型。
    使用 SVM 快速预测内爆对称性和不稳定性。
    """

    @staticmethod
    def generate_training_data(n_samples=200, seed=42):
        """
        生成训练数据 (模拟 ICF 内爆参数空间)。
        """
        rng = np.random.default_rng(seed)

        # 特征: [驱动不对称, 粗糙度, 内爆速度, ρR, 功率失衡]
        X = np.zeros((n_samples, 5))
        X[:, 0] = rng.uniform(0.001, 0.05, n_samples)   # 驱动不对称
        X[:, 1] = rng.uniform(1e-6, 1e-4, n_samples)    # 粗糙度 [cm]
        X[:, 2] = rng.uniform(2e7, 5e7, n_samples)      # 内爆速度 [cm/s]
        X[:, 3] = rng.uniform(0.1, 0.5, n_samples)      # ρR [g/cm²]
        X[:, 4] = rng.uniform(0.001, 0.03, n_samples)   # 功率失衡

        # 标签: 基于物理判据
        # 不稳定条件: 驱动不对称 > 3% 且 粗糙度 > 50 nm
        y = np.ones(n_samples)
        for i in range(n_samples):
            drive_asym = X[i, 0]
            roughness = X[i, 1]
            velocity = X[i, 2]
            rhoR = X[i, 3]

            # 简化的不稳定性判据
            rt_driving = drive_asym * velocity / (rhoR + 0.01)
            roughness_factor = roughness / 5e-5  # 归一化到 50 nm
            if rt_driving * roughness_factor > 0.5:
                y[i] = -1.0  # 不稳定

        return X, y

    @staticmethod
    def train_and_validate(n_samples=200, k_folds=5, seed=42):
        """训练并验证代理模型"""
        X, y = ICFSurrogateModel.generate_training_data(n_samples, seed)
        scaler = FeatureScaler()
        X_scaled = scaler.fit_transform(X)

        clf = SVMClassifier(C=1.0, n_iter=500)
        mean_acc, std_acc = CrossValidator.cross_validate(clf, X_scaled, y, k=k_folds)

        # 全量训练
        clf.fit(X_scaled, y)

        return {
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'classifier': clf,
            'scaler': scaler,
            'n_samples': n_samples,
            'n_features': X.shape[1]
        }
