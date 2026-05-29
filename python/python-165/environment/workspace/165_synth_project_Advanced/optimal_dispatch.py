"""
optimal_dispatch.py
最优经济调度与机组组合
融合种子项目：change_dynamic（动态规划）, change_polynomial（生成函数/卷积）
"""

import numpy as np
from typing import List, Tuple, Optional
from utils import polynomial_multiply


class EconomicDispatch:
    """
    电力系统经济调度（Economic Dispatch, ED）。

    目标：在满足负荷需求和机组出力约束的前提下，最小化总发电成本。

    数学模型：
        minimize   C_total = Σ_{i=1}^{N_g} C_i(P_{G,i})
        subject to Σ_{i=1}^{N_g} P_{G,i} = P_D
                   P_{G,i}^{min} ≤ P_{G,i} ≤ P_{G,i}^{max}

    其中发电成本函数通常采用二次模型：
        C_i(P_{G,i}) = a_i·P_{G,i}^2 + b_i·P_{G,i} + c_i   ($/h)

    最优性条件（等微增率准则，Equal Incremental Cost Criterion）：
        dC_i/dP_{G,i} = λ   (对所有未达边界的机组)
        即 2·a_i·P_{G,i} + b_i = λ

    解析解：
        P_{G,i} = (λ - b_i) / (2·a_i)

    λ 通过二分搜索确定，使得 Σ P_{G,i} = P_D。
    """

    def __init__(self, a: np.ndarray, b: np.ndarray, c: np.ndarray,
                 p_min: np.ndarray, p_max: np.ndarray):
        self.a = np.array(a, dtype=np.float64)
        self.b = np.array(b, dtype=np.float64)
        self.c = np.array(c, dtype=np.float64)
        self.p_min = np.array(p_min, dtype=np.float64)
        self.p_max = np.array(p_max, dtype=np.float64)
        self.n_gen = len(a)

    def incremental_cost(self, p: np.ndarray) -> np.ndarray:
        """
        微增成本：
            IC_i = dC_i/dP_{G,i} = 2·a_i·P_{G,i} + b_i
        """
        return 2.0 * self.a * p + self.b

    def solve_lambda(self, P_demand: float, lambda_bounds: Tuple[float, float] = (0.0, 200.0),
                     tol: float = 1e-6, max_iter: int = 100) -> dict:
        """
        基于等微增率准则的经济调度求解（λ 迭代法）。

        算法：
            1) 给定 λ，计算各机组出力 P_i(λ) = (λ - b_i)/(2a_i)。
            2) 将 P_i 裁剪到 [P_min, P_max]。
            3) 若 ΣP_i > P_D，降低 λ；否则升高 λ。
            4) 二分搜索直到收敛。
        """
        if P_demand < 0:
            raise ValueError("P_demand must be non-negative")
        lam_lo, lam_hi = lambda_bounds

        for _ in range(max_iter):
            lam = (lam_lo + lam_hi) * 0.5
            p = (lam - self.b) / (2.0 * self.a)
            p = np.clip(p, self.p_min, self.p_max)
            total = float(np.sum(p))
            if abs(total - P_demand) < tol:
                break
            if total > P_demand:
                lam_hi = lam
            else:
                lam_lo = lam

        cost = np.sum(self.a * p**2 + self.b * p + self.c)
        return {
            "pg": p,
            "lambda": lam,
            "total_cost": float(cost),
            "total_generation": float(total)
        }


class UnitCommitmentDP:
    """
    机组组合问题的动态规划求解（融合 change_dynamic 的动态规划思想）。

    问题描述：在 T 个时段内，决定每台机组的启停状态 u_{i,t}∈{0,1}，
    使得总成本（发电成本+启停成本）最小，且满足各时段负荷需求。

    单时段子问题（给定机组组合下的经济调度）可用 EconomicDispatch 求解。
    多时段状态转移考虑最小启停时间约束。

    为简化且保证可解性，采用单台机组多时段的 DP 模型演示核心思想，
    再扩展到多台机组的聚合 DP（生成函数卷积法）。
    """

    def __init__(self, n_gen: int, T: int,
                 startup_cost: np.ndarray, shutdown_cost: np.ndarray,
                 min_up: np.ndarray, min_down: np.ndarray):
        self.n_gen = n_gen
        self.T = T
        self.startup_cost = np.array(startup_cost, dtype=np.float64)
        self.shutdown_cost = np.array(shutdown_cost, dtype=np.float64)
        self.min_up = np.array(min_up, dtype=np.int32)
        self.min_down = np.array(min_down, dtype=np.int32)

    def solve_single_unit_dp(self, gen_idx: int,
                             ed_cost_on: np.ndarray,
                             ed_cost_off: float = 0.0) -> dict:
        """
        对单台机组进行 T 时段动态规划。

        状态定义：s_t = (u_t, τ_t)
            u_t ∈ {0,1} 为当前启停状态
            τ_t 为当前状态已持续时段数

        状态转移：
            若 u_{t-1}=1 且 τ_{t-1} < min_up：  u_t 必须为 1
            若 u_{t-1}=0 且 τ_{t-1} < min_down：u_t 必须为 0
            否则可自由切换，产生启停成本。

        DP 方程：
            V_t(u, τ) = C_t(u) + min_{u'} { V_{t-1}(u', τ') + C_switch(u'→u) }
        """
        T = self.T
        mu = int(self.min_up[gen_idx])
        md = int(self.min_down[gen_idx])
        INF = 1e18

        # 状态空间扁平化：状态 (u, tau) 映射到索引
        # u=0: tau=1..md; u=1: tau=1..mu
        n_states = mu + md
        # 映射：idx 0..md-1 对应 (0, tau=idx+1)
        #       idx md..md+mu-1 对应 (1, tau=idx-md+1)
        def idx_map(u, tau):
            if u == 0:
                return tau - 1
            return md + tau - 1

        # 初始化：假设 t=0 时机组已运行 1 时段
        V_prev = np.full(n_states, INF, dtype=np.float64)
        V_prev[idx_map(1, 1)] = ed_cost_on[0] if len(ed_cost_on) > 0 else 0.0
        V_prev[idx_map(0, 1)] = ed_cost_off

        policy = []

        for t in range(1, T):
            V_curr = np.full(n_states, INF, dtype=np.float64)
            best_prev = np.full(n_states, -1, dtype=np.int32)
            for u in [0, 1]:
                max_tau = mu if u == 1 else md
                for tau in range(1, max_tau + 1):
                    idx = idx_map(u, tau)
                    best_cost = INF
                    best_state = -1
                    for u_prev in [0, 1]:
                        max_tau_prev = mu if u_prev == 1 else md
                        for tau_prev in range(1, max_tau_prev + 1):
                            idx_prev = idx_map(u_prev, tau_prev)
                            if V_prev[idx_prev] >= INF:
                                continue
                            # 状态转移可行性
                            if u_prev == 1 and u == 1 and tau != tau_prev + 1:
                                continue
                            if u_prev == 0 and u == 0 and tau != tau_prev + 1:
                                continue
                            if u_prev == 1 and u == 0 and tau_prev < mu:
                                continue
                            if u_prev == 0 and u == 1 and tau_prev < md:
                                continue
                            if u_prev == 1 and u == 0 and tau != 1:
                                continue
                            if u_prev == 0 and u == 1 and tau != 1:
                                continue

                            switch_cost = 0.0
                            if u_prev == 0 and u == 1:
                                switch_cost = self.startup_cost[gen_idx]
                            if u_prev == 1 and u == 0:
                                switch_cost = self.shutdown_cost[gen_idx]

                            stage_cost = ed_cost_on[t] if u == 1 else ed_cost_off
                            total = V_prev[idx_prev] + stage_cost + switch_cost
                            if total < best_cost:
                                best_cost = total
                                best_state = idx_prev
                    V_curr[idx] = best_cost
                    best_prev[idx] = best_state
            V_prev = V_curr.copy()
            policy.append(best_prev)

        # 回溯最优路径
        final_idx = int(np.argmin(V_prev))
        path = [final_idx]
        for t in range(T - 2, -1, -1):
            final_idx = int(policy[t][final_idx])
            if final_idx < 0:
                break
            path.append(final_idx)
        path.reverse()

        schedule = []
        for idx in path:
            if idx < md:
                schedule.append(0)
            else:
                schedule.append(1)
        return {
            "schedule": np.array(schedule, dtype=np.int32),
            "total_cost": float(np.min(V_prev))
        }

    def solve_aggregated_dp(self, demand_series: np.ndarray,
                            ed_solver: EconomicDispatch) -> dict:
        """
        采用生成函数卷积（polynomial multiply）进行多机组容量聚合的动态规划。

        核心思想：将每台机组的可用出力状态视为多项式
            P_i(x) = Σ_{k} x^{p_{i,k}}
        其中 p_{i,k} 为机组 i 的第 k 个离散出力等级。

        n 台机组的总可用出力分布为卷积：
            P_total(x) = P_1(x) * P_2(x) * ... * P_n(x)

        这对应 change_polynomial 的卷积算法，用于快速判断某负荷需求
        是否存在可行机组组合，并枚举近似最优方案。
        """
        # 每台机组离散化为 5 个出力等级
        n_levels = 5
        max_demand = int(np.ceil(np.max(demand_series)))
        # 聚合容量分布（初始为多项式 [1] 表示只有 0 出力）
        agg = np.array([1.0])
        gen_polys = []
        for i in range(ed_solver.n_gen):
            p_levels = np.linspace(ed_solver.p_min[i], ed_solver.p_max[i], n_levels)
            poly = np.zeros(int(np.ceil(ed_solver.p_max[i])) + 1)
            for pl in p_levels:
                idx = int(round(pl))
                if idx < len(poly):
                    poly[idx] += 1.0
            gen_polys.append(poly)
            agg = polynomial_multiply(agg, poly)

        # 检查各时段负荷是否可被满足
        feasible = []
        for d in demand_series:
            idx = int(round(d))
            if idx < len(agg) and agg[idx] > 0:
                feasible.append(True)
            else:
                feasible.append(False)

        return {
            "capacity_distribution": agg,
            "feasible_per_period": np.array(feasible),
            "all_feasible": all(feasible)
        }
