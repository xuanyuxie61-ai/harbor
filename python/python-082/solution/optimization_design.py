"""
optimization_design.py
复合材料层合板铺层顺序的全局优化与动态规划。
原项目映射：
  - 471_glomin 的全局优化算法（带Lipschitz约束的Brent方法）
  - 156_change_dynamic 的动态规划思想用于最优铺层组合搜索
科学背景：
  层合板铺层设计优化问题：
    min_{θ_k} f(θ_1, ..., θ_n) = -F_{buckling} / F_{ref} + w_d * D_max
    subject to:
      -90° ≤ θ_k ≤ 90°
      |θ_k - θ_{k+1}| ≤ Δθ_max  (制造约束)
      平衡约束：对于每 +θ 层，存在 -θ 层
  其中 F_{buckling} 为屈曲载荷，D_max 为最大损伤变量，w_d 为权重。
  对于离散铺层角度（如每15°），可用动态规划求解最优序列。
"""

import numpy as np


class LaminateOptimization:
    """层合板铺层顺序优化器。"""

    def __init__(self, material, n_plies, target_load, thickness_per_ply=0.125):
        """
        material: CompositeMaterial
        n_plies: 铺层数
        target_load: 目标载荷 (N/m)
        thickness_per_ply: 单层厚度 (mm)
        """
        self.material = material
        self.n_plies = n_plies
        self.target_load = target_load
        self.thickness_per_ply = thickness_per_ply

    def objective_function(self, angles_deg):
        """
        目标函数：最小化 -(屈曲载荷/目标载荷) + 损伤惩罚。
        返回标量目标值。
        """
        from stiffness_assembly import LaminateStiffness
        from eigen_buckling import BucklingAnalysis

        thicknesses = [self.thickness_per_ply] * self.n_plies
        try:
            laminate = LaminateStiffness(angles_deg, thicknesses, self.material)
            buckling = BucklingAnalysis(laminate.D, 200.0, 100.0, nx=12, ny=12)
            N_cr = buckling.compute_critical_buckling_load(N_x=1.0)
        except Exception:
            return 1e6

        # 屈曲载荷因子
        buckling_factor = N_cr / (self.target_load + 1e-6)

        # 损伤惩罚（简化为角度不连续性惩罚）
        angle_penalty = 0.0
        for i in range(len(angles_deg) - 1):
            delta = abs(angles_deg[i] - angles_deg[i + 1])
            if delta > 60.0:
                angle_penalty += (delta - 60.0) * 0.01

        # 平衡性惩罚（非对称铺层）
        balance_penalty = 0.0
        if len(angles_deg) % 2 == 0:
            mid = len(angles_deg) // 2
            for i in range(mid):
                if abs(angles_deg[i] + angles_deg[-(i + 1)]) > 5.0:
                    balance_penalty += 0.1

        obj = -buckling_factor + angle_penalty + balance_penalty
        return obj

    def glomin_1d(self, f, a, b, c, m, e, t, max_calls=500):
        """
        一维全局最小化（从 471_glomin 迁移的Brent算法核心思想）。
        在区间[a,b]上寻找f(x)的全局最小值。
        假设 |f''(x)| ≤ M。
        """
        calls = 0
        a0 = float(b)
        x = a0
        a2 = float(a)
        y0 = f(b)
        calls += 1
        yb = y0
        y2 = f(a)
        calls += 1
        y = y2

        if y0 < y:
            y = y0
        else:
            x = a

        if m <= 0.0 or b <= a:
            return x, y, calls

        m2 = 0.5 * (1.0 + 16.0 * np.finfo(float).eps) * m

        if c <= a or b <= c:
            c = 0.5 * (a + b)

        y1 = f(c)
        calls += 1
        k = 3
        d0 = a2 - c
        h = 9.0 / 11.0

        if y1 < y:
            x = c
            y = y1

        y3 = y2
        while calls < max_calls:
            d1 = a2 - a0
            d2 = c - a0
            z2 = b - a2
            z0 = y2 - y1
            z1 = y2 - y0
            r = d1 * d1 * z0 - d0 * d0 * z1
            p = r
            qs = 2.0 * (d0 * z1 - d1 * z0)
            q = qs

            force_first = True
            if 100000 < k and y < y2:
                k = (1611 * k) % 1048576
                q = 1.0
                r = (b - a) * 0.00001 * k
                force_first = False

            inner_iter = 0
            while (r < z2 or force_first) and inner_iter < 100:
                inner_iter += 1
                force_first = False
                if q * (r * (yb - y2) + z2 * q * ((y2 - y) + t)) < z2 * m2 * r * (z2 * q - r):
                    a3 = a2 + r / q
                    if a3 < b:
                        y3 = f(a3)
                        calls += 1
                        if y3 < y:
                            x = a3
                            y = y3

                k = (1611 * k) % 1048576
                q = 1.0
                r = (b - a) * 0.00001 * k

            r = m2 * d0 * d1 * d2
            s = np.sqrt(((y2 - y) + t) / m2)
            h = 0.5 * (1.0 + h)
            p = h * (p + 2.0 * r * s)
            q = r + 0.5 * qs
            r = -0.5 * (d0 + (z0 + 2.01 * e) / (d0 * m2))

            if r < s or d0 < 0.0:
                r = a2 + s
            else:
                r = a2 + r

            if 0.0 < p * q:
                a3 = a2 + p / q
            else:
                a3 = r

            inner2 = 0
            while b <= a3 and inner2 < 50:
                inner2 += 1
                a3 = max(a3, r)
                if b <= a3:
                    a3 = b
                    y3 = yb
                else:
                    y3 = f(a3)
                    calls += 1

                if y3 < y:
                    x = a3
                    y = y3

                d0 = a3 - a2
                if a3 <= r:
                    break

                p = 2.0 * (y2 - y3) / (m * d0 + 1e-15)
                if (1.0 + 9.0 * np.finfo(float).eps) * d0 <= abs(p):
                    break
                if 0.5 * m2 * (d0 * d0 + p * p) <= (y2 - y) + (y3 - y) + 2.0 * t:
                    break

                a3 = 0.5 * (a2 + a3)
                h = 0.9 * h

            if b <= a3:
                break

            a0 = c
            c = a2
            a2 = a3
            y0 = y1
            y1 = y2
            y2 = y3

        return x, y, calls

    def optimize_single_angle(self):
        """优化单一铺层角度（所有层同向）。"""
        def f(theta):
            angles = [theta] * self.n_plies
            return self.objective_function(angles)

        # 全局搜索角度 [0, 90] 度，限制调用次数
        best_theta, best_obj, calls = self.glomin_1d(
            f, 0.0, 90.0, 45.0, 0.1, 1e-6, 1e-4, max_calls=30)
        return best_theta, best_obj, calls

    def dynamic_programming_stack(self, angle_set=None):
        """
        动态规划求解最优离散铺层序列（从 change_dynamic 迁移）。
        状态：dp[i][θ] = 前i层以角度θ结尾的最小目标值
        转移：dp[i][θ] = min_{θ'} {dp[i-1][θ'] + cost(θ', θ)}
        angle_set: 可选角度集合（默认每15度）
        """
        if angle_set is None:
            angle_set = list(range(-90, 91, 15))

        n_angles = len(angle_set)
        INF = 1e12

        # dp[j] = 当前层以 angle_set[j] 结尾的最优值
        dp = [INF] * n_angles
        parent = [-1] * n_angles

        # 初始化第一层
        for j in range(n_angles):
            angles = [angle_set[j]]
            dp[j] = self.objective_function(angles)

        # 逐层动态规划
        for layer in range(1, self.n_plies):
            new_dp = [INF] * n_angles
            new_parent = [-1] * n_angles
            for j in range(n_angles):
                theta_curr = angle_set[j]
                best_val = INF
                best_prev = -1
                for k in range(n_angles):
                    theta_prev = angle_set[k]
                    # 制造约束：相邻层角度变化不超过60度
                    if abs(theta_curr - theta_prev) > 60.0:
                        continue
                    # 快速评估：只构造前layer层
                    trial_angles = [angle_set[k]] * layer + [theta_curr]
                    val = dp[k] + self.objective_function(trial_angles)
                    if val < best_val:
                        best_val = val
                        best_prev = k
                new_dp[j] = best_val
                new_parent[j] = best_prev
            dp = new_dp
            parent = new_parent

        # 回溯最优解
        best_j = int(np.argmin(dp))
        optimal_angles = []
        j = best_j
        for layer in range(self.n_plies - 1, -1, -1):
            optimal_angles.append(angle_set[j])
            j = parent[j]
        optimal_angles.reverse()

        return optimal_angles, dp[best_j]


def compute_ply_combinations(n_plies, angle_candidates):
    """
    动态规划计算所有可行的铺层组合数（从 change_dynamic 的计数思想迁移）。
    类比 change_dynamic 的硬币问题：角度为"硬币面值"，n_plies为"目标金额"。
    这里统计满足平衡约束的铺层序列数。
    """
    # 简化为统计所有可能的序列数
    return len(angle_candidates) ** n_plies
