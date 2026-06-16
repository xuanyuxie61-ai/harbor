"""
相图分类与临界参数回归模块
============================
对应种子项目: 1186_GlacierWeilin_AAR_THAR_AABR-ELA
  (冰川 ELA 比率分类树 → 多铁性相图分类)

物理背景:
    多铁性材料的相结构由温度、应力、外场等参数决定.
    BiFeO3 的相图包含:
        - 反铁磁顺电相 (AFM-PE): 高温
        - 反铁磁铁电相 (AFM-FE): 室温稳定相 (菱形 R3c)
        - 顺磁顺电相 (PM-PE): 极高温
        - 外加应力可诱导: 正交 → 四方相变

    本模块实现:
    1. 从模拟数据中学习相边界 (分类树)
    2. 计算各相的序参量平均值 (类似 Loibl AAR 回归)
    3. 层级分类 (温度 → 应力 → 外场)

方法论 (对应种子项目 1186):
    1. 分类树: 基于参数空间 (T, σ, E, H) 将模拟结果分类为不同相
    2. 层级分组: 先按温度分, 再按应力分, 再按外场分
    3. 线性回归: 计算相边界斜率 (dT_c/dσ)

核心公式:
    相边界 (Clausius-Clapeyron 型):
        dT_C/dσ = T_C · Δε / ΔS
    其中 Δε 为应变差, ΔS 为熵差.

    Morpherpic 相界 (MPB):
        在 MPB 附近, 自由能对序参量的二阶导数趋于零:
        ∂²F/∂P² → 0 → 介电常数发散
"""

import numpy as np
from scipy import stats as scipy_stats


class MultiferroicPhaseClassifier:
    """
    多铁性材料相结构分类器.

    基于模拟参数 (T, σ, E, H) 和序参量 (P, M, ε)
    使用分类树识别相边界.
    """

    def __init__(self):
        # 相标签
        self.phase_labels = {
            0: 'AFM-PE (反铁磁顺电)',
            1: 'AFM-FE-R (反铁磁铁电-菱形)',
            2: 'AFM-FE-T (反铁磁铁电-四方)',
            3: 'AFM-FE-O (反铁磁铁电-正交)',
            4: 'PM-FE (顺磁铁电)',
            5: 'PM-PE (顺磁顺电)',
        }

        # 分类树节点
        self.tree = None

    # ============================================================
    # 分类树构建 (对应种子项目 1186 classification.py)
    # ============================================================

    def build_classification_tree(self, parameters, order_params,
                                  phase_labels, min_samples=3):
        """
        构建层级分类树.

        对应种子项目 1186 的 classification_tree 函数:
        基于参数空间将序参量数据分组.

        算法:
            1. 对每个参数维度, 找到最佳分裂点
            2. 递归分裂, 直到每个叶节点足够纯
            3. 返回层级分类结构

        参数:
            parameters: dict, 各参数维度的数组
                {'temperature': [...], 'stress': [...], 'E_field': [...]}
            order_params: 序参量数组, shape (n_samples, n_order_params)
            phase_labels: 每个样本的相标签, shape (n_samples,)
            min_samples: 叶节点最小样本数

        返回:
            tree: 分类树字典
        """
        # 组合参数矩阵
        param_names = list(parameters.keys())
        X = np.column_stack([parameters[k] for k in param_names])
        y = phase_labels

        self.tree = self._build_tree_recursive(
            X, y, param_names, depth=0, max_depth=4,
            min_samples=min_samples
        )

        return self.tree

    def _build_tree_recursive(self, X, y, param_names, depth, max_depth,
                              min_samples):
        """递归构建分类树."""
        node = {
            'n_samples': len(y),
            'phase_distribution': self._count_phases(y),
            'dominant_phase': np.bincount(y.astype(int),
                                          minlength=6).argmax(),
            'mean_order_params': np.mean(y),
        }

        # 终止条件
        if (depth >= max_depth or len(y) < min_samples or
                len(np.unique(y)) == 1):
            node['is_leaf'] = True
            return node

        # 寻找最佳分裂
        best_split = None
        best_impurity = np.inf

        for feat_idx in range(X.shape[1]):
            values = X[:, feat_idx]
            unique_vals = np.unique(values)

            if len(unique_vals) <= 1:
                continue

            # 尝试中位数分裂
            threshold = np.median(values)
            left_mask = values <= threshold
            right_mask = ~left_mask

            if (np.sum(left_mask) < min_samples or
                    np.sum(right_mask) < min_samples):
                continue

            # Gini 不纯度
            impurity = (np.sum(left_mask) * self._gini(y[left_mask]) +
                        np.sum(right_mask) * self._gini(y[right_mask])) / len(y)

            if impurity < best_impurity:
                best_impurity = impurity
                best_split = {
                    'feature': param_names[feat_idx],
                    'threshold': threshold,
                    'left_mask': left_mask,
                    'right_mask': right_mask,
                }

        if best_split is None:
            node['is_leaf'] = True
            return node

        node['is_leaf'] = False
        node['split_feature'] = best_split['feature']
        node['split_threshold'] = best_split['threshold']

        # 递归分裂
        node['left'] = self._build_tree_recursive(
            X[best_split['left_mask']], y[best_split['left_mask']],
            param_names, depth + 1, max_depth, min_samples
        )
        node['right'] = self._build_tree_recursive(
            X[best_split['right_mask']], y[best_split['right_mask']],
            param_names, depth + 1, max_depth, min_samples
        )

        return node

    def _gini(self, y):
        """计算 Gini 不纯度."""
        if len(y) == 0:
            return 0.0
        counts = np.bincount(y.astype(int))
        probs = counts / len(y)
        return 1.0 - np.sum(probs ** 2)

    def _count_phases(self, y):
        """统计各相的样本数."""
        counts = {}
        for label, name in self.phase_labels.items():
            counts[name] = int(np.sum(y == label))
        return counts

    # ============================================================
    # 相边界回归 (对应种子项目 1186 Loibl_AAR.py)
    # ============================================================

    def compute_phase_boundary(self, T_array, order_param_array,
                               threshold=0.5):
        """
        通过线性回归确定相变温度.

        对应种子项目 1186 的 cal_AAR 函数:
        使用线性回归 |P| vs T 确定 T_C (截距法).

        物理基础:
            在平均场近似下, |P| ∝ (T_C - T)^β, β ≈ 0.5
            对于小温度范围, 近似为线性: |P| ≈ a - bT
            令 |P| = 0 → T_C = a/b

        参数:
            T_array: 温度数组 (K)
            order_param_array: 序参量大小数组 (C/m² 或 A/m)
            threshold: 相变判定阈值

        返回:
            T_critical: 临界温度 (K)
            slope: 回归斜率
            intercept: 回归截距
            r_squared: 决定系数
        """
        valid = order_param_array > threshold
        if np.sum(valid) < 3:
            return None, None, None, 0.0

        T_valid = T_array[valid]
        P_valid = order_param_array[valid]

        # 线性回归
        result = scipy_stats.linregress(T_valid, P_valid)

        slope = result.slope
        intercept = result.intercept
        r_squared = result.rvalue ** 2

        # 临界温度: 令 P = threshold
        if abs(slope) > 1e-30:
            T_critical = (threshold - intercept) / slope
        else:
            T_critical = None

        return T_critical, slope, intercept, r_squared

    # ============================================================
    # 层级分类查询 (对应种子项目 1186 run_classification)
    # ============================================================

    def classify_parameters(self, temperature, stress=0.0,
                            E_field=0.0, H_field=0.0):
        """
        根据物理参数查询相结构.

        使用构建好的分类树进行查询.

        参数:
            temperature: 温度 (K)
            stress: 应力 (Pa)
            E_field: 电场 (V/m)
            H_field: 磁场 (A/m)

        返回:
            phase_id: 相标签
            phase_name: 相名称
            confidence: 分类置信度
        """
        if self.tree is None:
            # 默认分类 (基于物理知识)
            return self._default_classify(temperature, stress)

        params = {
            'temperature': temperature,
            'stress': stress,
            'E_field': E_field,
            'H_field': H_field,
        }

        return self._traverse_tree(self.tree, params)

    def _traverse_tree(self, node, params):
        """递归遍历分类树."""
        if node.get('is_leaf', True):
            phase_id = node['dominant_phase']
            total = max(node['n_samples'], 1)
            count = node['phase_distribution'].get(
                self.phase_labels.get(phase_id, ''), 0
            )
            return {
                'phase_id': phase_id,
                'phase_name': self.phase_labels.get(phase_id, 'Unknown'),
                'confidence': count / total,
            }

        feat = node['split_feature']
        thresh = node['split_threshold']

        if feat in params and params[feat] <= thresh:
            return self._traverse_tree(node['left'], params)
        else:
            return self._traverse_tree(node['right'], params)

    def _default_classify(self, T, sigma):
        """
        基于物理知识的默认分类.

        BiFeO3 相图近似:
            T < 643 K, σ ≈ 0: AFM-FE (菱形 R3c)
            T > 643 K: AFM-PE
            T > 1103 K: PM-PE
            σ > 2 GPa: 可能诱导正交或四方相
        """
        T_N = 643.0  # Néel 温度
        T_C = 1103.0  # Curie 温度
        sigma_MPB = 2.0e9  # MPB 应力

        if T > T_C:
            return {
                'phase_id': 5,
                'phase_name': self.phase_labels[5],
                'confidence': 0.9,
            }
        elif T > T_N:
            return {
                'phase_id': 4,
                'phase_name': self.phase_labels[4],
                'confidence': 0.85,
            }
        elif abs(sigma) > sigma_MPB:
            return {
                'phase_id': 2,
                'phase_name': self.phase_labels[2],
                'confidence': 0.8,
            }
        else:
            return {
                'phase_id': 1,
                'phase_name': self.phase_labels[1],
                'confidence': 0.95,
            }

    # ============================================================
    # Clausius-Clapeyron 分析
    # ============================================================

    def clausius_clapeyron_slope(self, T_array, sigma_array, P_array):
        """
        计算 Clausius-Clapeyron 斜率 dT_C/dσ.

        dT_C/dσ = T_C · Δε / ΔS

        其中:
            Δε = 应变差 (通过电致伸缩 Q·P² 计算)
            ΔS = 熵差 (通过 -∂F/∂T 计算)

        参数:
            T_array: 温度数组
            sigma_array: 应力数组
            P_array: 极化大小数组

        返回:
            dT_dsigma: 斜率 (K/Pa)
            delta_S: 熵差 (J/(m³·K))
            delta_eps: 应变差
        """
        # 简化: 直接回归 T_C vs σ
        if len(T_array) < 3:
            return 0.0, 0.0, 0.0

        result = scipy_stats.linregress(sigma_array, T_array)
        dT_dsigma = result.slope

        # 估算 Δε (使用电致伸缩)
        Q_eff = 0.055  # 有效电致伸缩系数
        P_max = np.max(P_array)
        delta_eps = Q_eff * P_max ** 2

        # ΔS ≈ ΔF/T_C
        T_C_mean = np.mean(T_array)
        delta_S = delta_eps / max(T_C_mean, 1.0)

        return dT_dsigma, delta_S, delta_eps
