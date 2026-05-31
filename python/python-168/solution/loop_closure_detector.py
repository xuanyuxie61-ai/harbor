"""
loop_closure_detector.py
基于随机游走与置换分析的环路闭合检测

核心数学模型：
1. 随机游走相似度（融合 random_walk_1d_simulation 项目）：
   将位姿序列建模为随机游走，环路闭合处应满足：
   
   E[||x_{t+L} - x_t||^2] ≈ σ^2 * L    （无环路时）
   
   若检测到环路闭合，则实际位移应显著小于随机游走进展
   
   定义环路似然：
   P_loop(t, j) ∝ exp( - ||x_t - x_j||^2 / (2 σ^2 |t-j|) )

2. 随机搜索数据关联（融合 locker_simulation 项目）：
   在大量候选帧中搜索匹配，采用随机采样一致性：
   - 随机抽取 K 个候选
   - 计算扫描匹配分数
   - 若最高分超过阈值则接受为环路闭合
   
   成功概率分析：
   P(find | N candidates, K samples) = 1 - C(N-M, K) / C(N, K)
   其中 M 为真匹配数

3. 置换正交性检验（融合 permutation_puzzle 项目）：
   检验两组点云对应关系是否保持旋转正交性
   
   对于对应点集 {p_i} 和 {q_i}，中心化后：
   H = Σ p_i q_i^T = U S V^T
   
   若对应正确，则 R = V U^T 应为纯正交矩阵（det=+1）
   若存在异常值，则可通过分析 R 的奇异值分布检测
   
   置换群视角：
   设 σ 为点对应排列，若 σ 保持度量结构，则：
   ||p_i - p_j|| ≈ ||q_{σ(i)} - q_{σ(j)}||
"""

import numpy as np


class RandomWalkLoopDetector:
    """
    基于随机游走模型的环路闭合检测器
    融合 random_walk_1d_simulation 项目
    """

    def __init__(self, sigma_odometry=0.1, window_size=10, threshold_ratio=2.0):
        self.sigma_odometry = max(float(sigma_odometry), 1e-12)
        self.window_size = max(int(window_size), 1)
        self.threshold_ratio = max(float(threshold_ratio), 0.1)

    def compute_loop_likelihood(self, trajectory):
        """
        计算每对时刻之间的环路闭合似然
        
        对于位姿序列 x_1, ..., x_T，定义距离矩阵：
        D_{ij} = ||trans(x_i) - trans(x_j)||
        
        随机游走进展预期（一维）：
        E[D_{ij}] ≈ σ * sqrt(|i-j|)
        
        环路闭合指标：
        S_{ij} = E[D_{ij}] / (D_{ij} + ε)
        若 S_{ij} >> 1，则可能存在环路闭合
        
        Parameters
        ----------
        trajectory : list of ndarray(3,)
            位姿序列 [x, y, theta]
        
        Returns
        -------
        likelihood_matrix : ndarray, shape (T, T)
        candidates : list of tuple (i, j, score)
        """
        T = len(trajectory)
        if T < 2:
            return np.zeros((T, T)), []

        positions = np.array([p[0:2] for p in trajectory], dtype=np.float64)
        likelihood = np.zeros((T, T), dtype=np.float64)

        for i in range(T):
            for j in range(i + self.window_size, T):
                dt = j - i
                actual_dist = np.linalg.norm(positions[i] - positions[j])
                expected_dist = self.sigma_odometry * np.sqrt(dt)

                if actual_dist < 1e-12:
                    score = 10.0
                else:
                    score = expected_dist / actual_dist

                likelihood[i, j] = score
                likelihood[j, i] = score

        # 提取候选对
        candidates = []
        for i in range(T):
            for j in range(i + self.window_size, T):
                if likelihood[i, j] > self.threshold_ratio:
                    candidates.append((i, j, likelihood[i, j]))

        # 按分数排序
        candidates.sort(key=lambda x: x[2], reverse=True)
        return likelihood, candidates


class RandomSearchAssociator:
    """
    随机搜索数据关联器
    融合 locker_simulation 的随机搜索策略
    """

    def __init__(self, max_candidates=50, max_trials=20, match_threshold=0.8):
        self.max_candidates = max(int(max_candidates), 1)
        self.max_trials = max(int(max_trials), 1)
        self.match_threshold = max(float(match_threshold), 0.0)

    def search_matches(self, current_scan, candidate_scans, similarity_func):
        """
        在候选扫描中随机搜索最佳匹配
        
        Parameters
        ----------
        current_scan : ndarray
        candidate_scans : list of ndarray
        similarity_func : callable
            (scan1, scan2) -> float score
        
        Returns
        -------
        best_idx : int or None
        best_score : float
        """
        n = len(candidate_scans)
        if n == 0:
            return None, 0.0

        # 随机采样候选索引
        num_samples = min(self.max_trials, n)
        indices = np.random.choice(n, size=num_samples, replace=False)

        best_idx = None
        best_score = -np.inf
        for idx in indices:
            score = similarity_func(current_scan, candidate_scans[idx])
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_score < self.match_threshold:
            return None, best_score
        return best_idx, best_score

    def simulate_find_probability(self, total_candidates, true_matches, num_trials, simulation_count=500):
        """
        蒙特卡洛模拟搜索成功概率
        
        融合 locker_simulation / high_card_simulation 思想
        
        P(success) = 1 - C(N-M, K) / C(N, K)
        其中 N=total_candidates, M=true_matches, K=num_trials
        """
        found_count = 0
        for _ in range(simulation_count):
            # 标记真匹配位置
            match_positions = set(np.random.choice(total_candidates, size=true_matches, replace=False))
            # 随机采样
            sampled = set(np.random.choice(total_candidates, size=min(num_trials, total_candidates), replace=False))
            if sampled & match_positions:
                found_count += 1

        empirical_prob = found_count / simulation_count
        # 理论概率
        from math import comb
        try:
            theoretical = 1.0 - comb(total_candidates - true_matches, num_trials) / comb(total_candidates, num_trials)
        except (ValueError, ZeroDivisionError):
            theoretical = empirical_prob

        return empirical_prob, theoretical


class PermutationOrthogonalityChecker:
    """
    置换正交性检验器
    融合 permutation_puzzle 项目
    """

    @staticmethod
    def check_rotation_orthogonality(points_a, points_b, correspondence):
        """
        检验对应关系是否保持旋转正交性
        
        给定两组点云和对应关系，计算最优刚体变换后检验正交性：
        
        1. 中心化
           p'_i = p_i - μ_p,  q'_i = q_i - μ_q
        2. SVD: H = Σ p'_i q'_i^T = U Σ V^T
        3. R = V U^T
        4. 检验 det(R) = +1（纯旋转，非反射）
        
        置换视角：
        若对应为正确置换 σ，则存在正交矩阵 R 使得：
        q_{σ(i)} ≈ R p_i + t
        
        不正交的程度可用 ||R^T R - I||_F 度量
        
        Parameters
        ----------
        points_a : ndarray, shape (N, 2)
        points_b : ndarray, shape (M, 2)
        correspondence : list of tuple (i, j)
        
        Returns
        -------
        is_valid : bool
        rotation_matrix : ndarray(2,2)
        translation : ndarray(2,)
        orthogonality_error : float
        """
        if not correspondence:
            return False, np.eye(2), np.zeros(2), np.inf

        pts_a = np.array([points_a[i] for i, _ in correspondence], dtype=np.float64)
        pts_b = np.array([points_b[j] for _, j in correspondence], dtype=np.float64)

        if pts_a.shape[0] < 2:
            return False, np.eye(2), np.zeros(2), np.inf

        mu_a = np.mean(pts_a, axis=0)
        mu_b = np.mean(pts_b, axis=0)
        a_centered = pts_a - mu_a
        b_centered = pts_b - mu_b

        H = a_centered.T @ b_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # 保证行列式为+1
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t = mu_b - R @ mu_a

        # 正交性误差
        ortho_err = np.linalg.norm(R.T @ R - np.eye(2), ord='fro')

        # 重投影误差
        reproj = b_centered - a_centered @ R.T
        reproj_err = np.mean(np.sum(reproj ** 2, axis=1))

        is_valid = (ortho_err < 0.1) and (reproj_err < 1.0)

        return is_valid, R, t, ortho_err

    @staticmethod
    def demonstrate_permutation_property(n=10):
        """
        展示置换正交性性质（融合 permutation_puzzle）
        
        对于任意置换 P1, P2 ∈ S_n：
        定义点集 XY = [P1; P2] - n/2
        旋转 45 度后，XY(1,:) 与 XY(2,:) 在 n 维空间中正交
        
        数学证明：
        旋转矩阵 A = [[cosθ, -sinθ], [sinθ, cosθ]]，θ=π/4
        X' = cosθ P1 - sinθ P2 - (n/2)(cosθ - sinθ)
        Y' = sinθ P1 + cosθ P2 - (n/2)(sinθ + cosθ)
        
        X'·Y' = (cosθ sinθ)(||P1||^2 - ||P2||^2) + (cos^2θ - sin^2θ) P1·P2 + const
        
        由于 P1, P2 为同一集合的置换，||P1||^2 = ||P2||^2
        且当 θ=π/4 时 cos^2θ - sin^2θ = 0
        故 X'·Y' = 0
        """
        p1 = np.random.permutation(n) + 1
        p2 = np.random.permutation(n) + 1

        xy = np.vstack([p1, p2]).astype(np.float64)
        xy = xy - n / 2.0

        angle = np.pi / 4.0
        A = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]], dtype=np.float64)
        xy_rot = A @ xy

        dot_product = np.dot(xy_rot[0, :], xy_rot[1, :])
        return dot_product


class IntegratedLoopClosureDetector:
    """
    集成环路闭合检测器
    组合多种检测策略
    """

    def __init__(self, rw_sigma=0.1, rw_window=10, rw_threshold=2.0,
                 rs_max_candidates=50, rs_trials=20, rs_threshold=0.8):
        self.rw_detector = RandomWalkLoopDetector(rw_sigma, rw_window, rw_threshold)
        self.rs_associator = RandomSearchAssociator(rs_max_candidates, rs_trials, rs_threshold)
        self.ortho_checker = PermutationOrthogonalityChecker()

    def detect(self, trajectory, scans, scan_similarity_func):
        """
        综合检测环路闭合
        
        Parameters
        ----------
        trajectory : list of ndarray(3,)
        scans : list of ndarray
        scan_similarity_func : callable
        
        Returns
        -------
        closures : list of dict
            每个元素 {'from': int, 'to': int, 'transform': ndarray(3,), 'score': float}
        """
        # 1. 随机游走检测候选对
        _, candidates = self.rw_detector.compute_loop_likelihood(trajectory)

        closures = []
        checked_pairs = set()

        for i, j, rw_score in candidates[:20]:  # 限制候选数量
            pair = tuple(sorted((i, j)))
            if pair in checked_pairs:
                continue
            checked_pairs.add(pair)

            # 2. 随机搜索扫描匹配
            # 构建候选扫描列表（排除时间邻近帧）
            candidate_scans = []
            candidate_indices = []
            for idx in range(len(scans)):
                if abs(idx - i) > self.rw_detector.window_size:
                    candidate_scans.append(scans[idx])
                    candidate_indices.append(idx)

            match_rel_idx, match_score = self.rs_associator.search_matches(
                scans[i], candidate_scans, scan_similarity_func
            )

            if match_rel_idx is None or candidate_indices[match_rel_idx] != j:
                continue

            # 3. 正交性检验
            # 使用最近邻获取粗略对应
            pts_i = scans[i]
            pts_j = scans[j]
            if pts_i.shape[0] == 0 or pts_j.shape[0] == 0:
                continue

            corr = []
            for pi, p in enumerate(pts_i[:min(20, len(pts_i))]):
                dists = np.sum((pts_j - p) ** 2, axis=1)
                corr.append((pi, np.argmin(dists)))

            is_valid, R, t, ortho_err = self.ortho_checker.check_rotation_orthogonality(
                pts_i, pts_j, corr
            )

            if is_valid:
                # 计算相对位姿变换
                theta = np.arctan2(R[1, 0], R[0, 0])
                transform = np.array([t[0], t[1], theta], dtype=np.float64)
                closures.append({
                    'from': i,
                    'to': j,
                    'transform': transform,
                    'score': rw_score * match_score
                })

        return closures
