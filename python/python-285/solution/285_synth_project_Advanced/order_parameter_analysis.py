"""
序参量分析模块
===============
对应种子项目: 1234_playingwithgithub24_HackBio-Single-Cell-RNA-Seq-Stage-3
  (scRNA-seq PCA降维 → 序参量空间降维分析)

物理背景:
    多铁性材料的序参量空间是高维的:
        - 极化: P = (P₁, P₂, P₃)
        - 磁化: M = (M₁, M₂, M₃)
        - 应变: ε = (ε₁₁, ε₂₂, ε₃₃, ε₂₃, ε₁₃, ε₁₂) (6个独立分量)
        - 反铁磁矢量: L = (L₁, L₂, L₃)
    共 15 维.

    通过 PCA 降维, 可以:
    1. 找到主导的序参量组合 (主成分)
    2. 识别相变路径 (序参量空间的轨迹)
    3. 检测隐藏序参量 (高阶主成分中的异常)

方法论 (对应种子项目 1234):
    1. 标准化: 对每个序参量维度做 z-score 标准化
    2. PCA: 计算协方差矩阵, 特征值分解
    3. 选择前 k 个主成分 (累积方差 > 95%)
    4. 在低维空间中分析序参量演化

核心公式:
    协方差矩阵:
        C = (1/N) Σᵢ (xᵢ - x̄)(xᵢ - x̄)ᵀ

    PCA 分解:
        C = U·Λ·Uᵀ
    其中 Λ 为特征值矩阵, U 为主成分方向.

    累积方差比:
        R(k) = Σᵢ₌₁ᵏ λᵢ / Σᵢ λᵢ

    序参量空间的自由能面:
        F(PC₁, PC₂) = -k_BT · ln ρ(PC₁, PC₂)
    其中 ρ 为主成分空间中的概率密度.
"""

import numpy as np


class OrderParameterAnalyzer:
    """
    多铁性序参量分析器.

    使用 PCA 和统计方法分析序参量空间的维数、
    主方向和相变特征.
    """

    def __init__(self):
        self.mean = None
        self.std = None
        self.components = None
        self.explained_variance = None
        self.explained_variance_ratio = None

    # ============================================================
    # PCA (对应种子项目 1234 的 PCA 降维)
    # ============================================================

    def fit_pca(self, data, n_components=None):
        """
        对序参量数据做 PCA 降维.

        对应种子项目 1234 中的 sc.tl.pca():
        从多帧模拟数据中提取主成分.

        参数:
            data: 序参量矩阵, shape (n_samples, n_features)
                每行为一个时间步的序参量向量
            n_components: 保留的主成分数 (默认自动选择)

        返回:
            transformed: 降维后的数据, shape (n_samples, k)
        """
        n_samples, n_features = data.shape

        # 标准化
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        self.std[self.std < 1e-30] = 1.0
        data_scaled = (data - self.mean) / self.std

        # 协方差矩阵
        C = np.cov(data_scaled, rowvar=False)

        # 特征值分解
        eigenvalues, eigenvectors = np.linalg.eigh(C)

        # 按特征值降序排列
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # 确保非负
        eigenvalues = np.maximum(eigenvalues, 0.0)

        # 自动选择主成分数
        if n_components is None:
            total_var = np.sum(eigenvalues)
            cumvar = np.cumsum(eigenvalues) / max(total_var, 1e-30)
            n_components = np.searchsorted(cumvar, 0.95) + 1
            n_components = min(n_components, n_features)

        self.components = eigenvectors[:, :n_components]
        self.explained_variance = eigenvalues[:n_components]
        self.explained_variance_ratio = (eigenvalues[:n_components] /
                                         max(np.sum(eigenvalues), 1e-30))

        # 变换
        transformed = data_scaled @ self.components

        return transformed

    def transform(self, data):
        """
        将新数据变换到 PCA 空间.

        参数:
            data: shape (n_samples, n_features)

        返回:
            transformed: shape (n_samples, n_components)
        """
        if self.mean is None:
            raise RuntimeError("需要先调用 fit_pca")

        data_scaled = (data - self.mean) / self.std
        return data_scaled @ self.components

    # ============================================================
    # 自由能面重构
    # ============================================================

    def free_energy_landscape(self, pc_data, n_bins=30, T=300.0):
        """
        从主成分数据重构自由能面.

        F(PC₁, PC₂) = -k_BT · ln ρ(PC₁, PC₂) + F₀

        其中 ρ 为二维概率密度 (通过直方图估计).

        参数:
            pc_data: PCA 变换后的数据, shape (n_samples, ≥2)
            n_bins: 直方图箱数
            T: 温度 (K)

        返回:
            F: 自由能面, shape (n_bins, n_bins)
            pc1_edges: PC₁ 边界
            pc2_edges: PC₂ 边界
        """
        from multiferroic_constants import K_BOLTZMANN, E_CHARGE

        kT = K_BOLTZMANN * T / E_CHARGE  # eV

        pc1 = pc_data[:, 0]
        pc2 = pc_data[:, 1] if pc_data.shape[1] > 1 else pc1

        hist, x_edges, y_edges = np.histogram2d(
            pc1, pc2, bins=n_bins
        )

        # 概率密度
        rho = hist / np.sum(hist)
        rho = np.maximum(rho, 1e-30)

        # 自由能
        F = -kT * np.log(rho)
        F -= np.min(F)  # 以最小值为参考

        return F, x_edges, y_edges

    # ============================================================
    # 序参量统计分析
    # ============================================================

    def order_parameter_statistics(self, P_history, M_history):
        """
        计算序参量的统计量.

        参数:
            P_history: 极化时间序列, shape (n_steps, 3)
            M_history: 磁化时间序列, shape (n_steps, 3)

        返回:
            stats: 统计字典
        """
        stats = {}

        # 极化统计
        P_mag = np.linalg.norm(P_history, axis=1)
        stats['P_mean'] = np.mean(P_history, axis=0)
        stats['P_magnitude_mean'] = np.mean(P_mag)
        stats['P_magnitude_std'] = np.std(P_mag)
        stats['P_direction'] = (np.mean(P_history, axis=0) /
                                max(np.mean(P_mag), 1e-30))

        # 磁化统计
        M_mag = np.linalg.norm(M_history, axis=1)
        stats['M_mean'] = np.mean(M_history, axis=0)
        stats['M_magnitude_mean'] = np.mean(M_mag)
        stats['M_magnitude_std'] = np.std(M_mag)
        stats['M_direction'] = (np.mean(M_history, axis=0) /
                                max(np.mean(M_mag), 1e-30))

        # 磁电耦合
        PM_dot = np.sum(P_history * M_history, axis=1)
        stats['PM_correlation'] = np.mean(PM_dot)
        stats['PM_correlation_std'] = np.std(PM_dot)

        # 各向异性参数
        if len(P_mag) > 1:
            P_autocorr = np.correlate(P_mag - np.mean(P_mag),
                                      P_mag - np.mean(P_mag),
                                      mode='full')
            P_autocorr = P_autocorr[len(P_autocorr) // 2:]
            P_autocorr /= max(P_autocorr[0], 1e-30)
            stats['P_autocorr_time'] = np.sum(
                P_autocorr[:min(100, len(P_autocorr))])
        else:
            stats['P_autocorr_time'] = 0.0

        return stats

    # ============================================================
    # 相变检测
    # ============================================================

    def detect_phase_transition(self, T_array, order_param_array):
        """
        通过序参量突变检测相变温度.

        方法:
            1. 计算序参量随温度的变化率 d|P|/dT
            2. 最大变化率处即为相变点

        参数:
            T_array: 温度数组
            order_param_array: 序参量大小数组

        返回:
            T_transition: 相变温度
            max_derivative: 最大变化率
        """
        if len(T_array) < 3:
            return None, 0.0

        # 数值微分
        dP_dT = np.gradient(order_param_array, T_array)

        # 最大变化率
        max_idx = np.argmax(np.abs(dP_dT))
        T_transition = T_array[max_idx]
        max_derivative = dP_dT[max_idx]

        return T_transition, max_derivative

    # ============================================================
    # Binder 累积量
    # ============================================================

    def binder_cumulant(self, order_param_samples):
        """
        计算 Binder 累积量 U₄.

        U₄ = 1 - <m⁴> / (3·<m²>²)

        其中 m 为序参量.

        物理含义:
            U₄ → 2/3: 有序相
            U₄ → 0: 无序相
            U₄ 交叉点: 相变温度

        参数:
            order_param_samples: 序参量样本

        返回:
            U4: Binder 累积量
        """
        m2 = np.mean(order_param_samples ** 2)
        m4 = np.mean(order_param_samples ** 4)

        if m2 < 1e-30:
            return 0.0

        U4 = 1.0 - m4 / (3.0 * m2 ** 2)
        return U4
