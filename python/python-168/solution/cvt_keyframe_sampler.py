"""
cvt_keyframe_sampler.py
基于 Centroidal Voronoi Tessellation 的关键帧采样与最优停止策略

核心数学模型：
1. Centroidal Voronoi Tessellation (CVT)：
   对于区域 Ω ⊂ R^d 和生成元集 G = {g_i}_{i=1}^N，
   Voronoi 区域 V_i = {x ∈ Ω : ||x - g_i|| <= ||x - g_j||, ∀j≠i}
   CVT 满足：每个生成元 g_i 恰好是其 Voronoi 区域 V_i 的质心
   
   质心定义：C_i = (∫_{V_i} x ρ(x) dx) / (∫_{V_i} ρ(x) dx)
   其中 ρ(x) 为密度函数（在SLAM中可取为信息增益）

2. Lloyd 迭代算法：
   g_i^{(k+1)} = C_i( {g_j^{(k)}} )
   能量泛函单调递减：E(G) = Σ ∫_{V_i} ||x - g_i||^2 ρ(x) dx
   
   一维解析形式（融合 cvt_1d_lloyd 项目）：
   对于排序后的生成元 g_1 < g_2 < ... < g_N，
   Voronoi 边界在 b_i = (g_i + g_{i+1}) / 2
   质心（均匀密度下）: C_i = (b_{i-1} + b_i) / 2

3. 最优停止策略（融合 high_card_simulation 项目）：
   关键帧选择可建模为秘书问题变体：
   观测序列 {I_1, I_2, ..., I_T}，每个 I_t 为当前帧的信息增益
   策略：跳过前 r = floor(T/e) 帧作为参考，之后选择第一个超过参考最大值的帧
   最优 r/T → 1/e ≈ 0.3679 当 T → ∞
   
   成功概率 P(success) ≈ 1/e ≈ 0.3679
"""

import numpy as np


class CVTKeyframeSampler:
    """
    基于 CVT 的关键帧采样器
    """

    def __init__(self, num_generators=20, max_iter=50, domain_bounds=None):
        """
        Parameters
        ----------
        num_generators : int
            CVT 生成元数量（即目标关键帧数）
        max_iter : int
            Lloyd 最大迭代次数
        domain_bounds : tuple or None
            配置空间边界 (min, max)，若为None则自适应
        """
        self.num_generators = max(int(num_generators), 2)
        self.max_iter = max(int(max_iter), 1)
        self.domain_bounds = domain_bounds
        self.generators = None
        self.energies = []

    def fit(self, feature_points, weights=None):
        """
        对特征点集进行 CVT 采样
        
        Parameters
        ----------
        feature_points : ndarray, shape (N, d)
            特征空间中的点（如 [x, y, θ] 或降维后的特征）
        weights : ndarray or None
            每个点的权重（信息密度）
        
        Returns
        -------
        generators : ndarray, shape (num_generators, d)
            CVT 生成元（关键帧特征）
        labels : ndarray, shape (N,)
            每个特征点所属的 Voronoi 区域索引
        """
        points = np.asarray(feature_points, dtype=np.float64)
        N, d = points.shape
        if N == 0:
            self.generators = np.zeros((self.num_generators, d))
            return self.generators, np.array([])

        if weights is None:
            weights = np.ones(N, dtype=np.float64)
        else:
            weights = np.asarray(weights, dtype=np.float64)
            weights = np.maximum(weights, 1e-12)

        # 确定边界
        if self.domain_bounds is None:
            mins = np.min(points, axis=0)
            maxs = np.max(points, axis=0)
            padding = (maxs - mins) * 0.1
            mins -= padding
            maxs += padding
            # 避免零宽度
            for dim_idx in range(d):
                if maxs[dim_idx] - mins[dim_idx] < 1e-8:
                    maxs[dim_idx] = mins[dim_idx] + 1.0
        else:
            mins = np.full(d, self.domain_bounds[0])
            maxs = np.full(d, self.domain_bounds[1])

        # 初始化生成元（K-means++风格初始化以更好覆盖）
        generators = self._kmeans_plus_plus(points, self.num_generators)

        self.energies = []
        for it in range(self.max_iter):
            # 分配到最近的生成元
            labels = self._assign_voronoi(points, generators)

            # 计算新质心
            new_generators = np.zeros_like(generators)
            for i in range(self.num_generators):
                mask = (labels == i)
                if np.any(mask):
                    w = weights[mask]
                    p = points[mask]
                    new_generators[i] = np.sum(p * w[:, None], axis=0) / np.sum(w)
                else:
                    # 空区域：重新初始化到数据密集处
                    new_generators[i] = points[np.random.choice(N)]

            # 投影到边界内
            new_generators = np.clip(new_generators, mins, maxs)

            # 计算能量
            energy = self._compute_energy(points, generators, labels, weights)
            self.energies.append(energy)

            # 收敛判断
            motion = np.mean(np.sum((new_generators - generators) ** 2, axis=1))
            generators = new_generators
            if motion < 1e-12:
                break

        self.generators = generators
        labels = self._assign_voronoi(points, generators)
        return generators, labels

    @staticmethod
    def _kmeans_plus_plus(points, k):
        """K-means++ 初始化"""
        N, d = points.shape
        centers = np.zeros((k, d), dtype=np.float64)
        centers[0] = points[np.random.randint(N)]
        for i in range(1, k):
            dists = np.min(np.sum((points[:, None, :] - centers[None, :i, :]) ** 2, axis=2), axis=1)
            probs = dists / (np.sum(dists) + 1e-12)
            idx = np.random.choice(N, p=probs)
            centers[i] = points[idx]
        return centers

    @staticmethod
    def _assign_voronoi(points, generators):
        """将点分配到最近的生成元"""
        dists = np.sum((points[:, None, :] - generators[None, :, :]) ** 2, axis=2)
        return np.argmin(dists, axis=1)

    @staticmethod
    def _compute_energy(points, generators, labels, weights):
        """
        CVT 能量泛函：
        E = Σ_i ∫_{V_i} ||x - g_i||^2 ρ(x) dx
        离散近似：
        E ≈ Σ_i Σ_{x∈V_i} w(x) ||x - g_i||^2
        """
        energy = 0.0
        for i in range(generators.shape[0]):
            mask = (labels == i)
            if np.any(mask):
                diff = points[mask] - generators[i]
                energy += np.sum(weights[mask] * np.sum(diff ** 2, axis=1))
        return energy


class OptimalStoppingKeyframeSelector:
    """
    基于最优停止理论的关键帧选择器
    融合 high_card_simulation 项目思想
    """

    def __init__(self, total_frames=None):
        self.total_frames = total_frames

    @staticmethod
    def optimal_skip_ratio(total_frames):
        """
        计算最优跳过比例
        
        对于秘书问题，最优策略跳过前 r = floor(T/e) 个候选
        返回 r / T ≈ 1/e
        """
        if total_frames is None or total_frames <= 0:
            return 1.0 / np.e
        return min(int(total_frames / np.e) / max(total_frames, 1), 0.999)

    def select_keyframes(self, information_gains, total_frames=None):
        """
        使用最优停止策略选择关键帧
        
        Parameters
        ----------
        information_gains : ndarray
            每帧的信息增益序列
        total_frames : int or None
            总帧数估计（若未提供则使用 len(information_gains)）
        
        Returns
        -------
        selected_indices : list
            选中的关键帧索引
        """
        gains = np.asarray(information_gains, dtype=np.float64)
        T = len(gains)
        if T == 0:
            return []

        if total_frames is None:
            total_frames = T

        skip_num = max(1, int(total_frames / np.e))
        skip_num = min(skip_num, T - 1)

        selected = []
        if skip_num >= T:
            return [np.argmax(gains)]

        # 参考阈值：前 skip_num 帧的最大信息增益
        reference_max = np.max(gains[:skip_num]) if skip_num > 0 else -np.inf

        for i in range(skip_num, T):
            if gains[i] > reference_max:
                selected.append(i)
                # 更新参考阈值
                reference_max = gains[i]

        # 保证至少选择一个
        if not selected:
            selected.append(int(np.argmax(gains[skip_num:])) + skip_num)

        return selected

    def simulate_strategy(self, deck_size=100, trial_num=500):
        """
        模拟最优停止策略的成功率
        
        融合 high_card_simulation 的蒙特卡洛模拟思想
        
        Returns
        -------
        success_rate : float
            成功选择全局最优的比例
        theoretical : float
            理论极限 1/e
        """
        correct = 0
        for _ in range(trial_num):
            cards = np.random.permutation(deck_size) + 1
            skip = max(1, int(deck_size / np.e))
            skip_max = np.max(cards[:skip]) if skip > 0 else -np.inf

            choice = cards[-1]
            for i in range(skip, deck_size):
                if cards[i] > skip_max:
                    choice = cards[i]
                    break

            if choice == deck_size:
                correct += 1

        success_rate = correct / trial_num
        theoretical = 1.0 / np.e
        return success_rate, theoretical


class InformationGainEstimator:
    """
    信息增益估计器：用于评估每帧的观测信息量
    """

    @staticmethod
    def compute_fisher_information(pose, landmarks, sigma_obs=0.1):
        """
        计算给定姿态下对地标观测的 Fisher 信息矩阵
        
        对于激光观测，观测模型：
        z_i = h(x, m_i) + ε
        
        Fisher 信息：
        I(x) = Σ_i (∂h/∂x)^T Σ^{-1} (∂h/∂x)
        
        信息增益可用 det(I(x)) 或 trace(I(x)) 度量
        """
        sigma_obs = max(float(sigma_obs), 1e-12)
        x, y, theta = pose

        info_mat = np.zeros((3, 3), dtype=np.float64)
        for lm in landmarks:
            mx, my = lm
            dx = mx - x
            dy = my - y
            r2 = dx * dx + dy * dy
            if r2 < 1e-12:
                continue
            r = np.sqrt(r2)

            # 观测雅可比（距离和角度）
            # h = [r, atan2(dy, dx) - theta]^T
            H = np.array([[-dx / r, -dy / r, 0],
                          [dy / r2, -dx / r2, -1]], dtype=np.float64)
            Sigma_inv = np.eye(2) / (sigma_obs ** 2)
            info_mat += H.T @ Sigma_inv @ H

        # 信息增益指标
        det_info = np.linalg.det(info_mat)
        trace_info = np.trace(info_mat)
        # 组合指标
        gain = trace_info + np.log(max(det_info, 1e-12) + 1.0)
        return gain
