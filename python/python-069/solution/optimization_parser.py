"""
优化解解析模块：基于 cplex_solution_read 思想，
解析碳分配线性规划问题的最优解。

核心模型（线性规划）：
  最大化：
      Z = c_leaf * x_leaf + c_stem * x_stem + c_root * x_root + c_storage * x_storage

  约束：
      x_leaf + x_stem + x_root + x_storage <= C_total
      x_leaf >= R_maint_leaf
      x_stem >= R_maint_stem
      x_root >= R_maint_root
      x_storage >= 0

  其中 C_total 为当日总光合碳收入。
"""
import numpy as np


def solve_carbon_lp(c_total, coeffs, maint_req):
    """
    用贪心法求解碳分配 LP（cplex 解的思想简化）。
    c_total: 总碳
    coeffs: dict {'leaf': c_leaf, 'stem': c_stem, 'root': c_root, 'storage': c_storage}
    maint_req: dict 各器官维持呼吸需求
    返回: 最优解 dict, 目标值
    """
    organs = list(coeffs.keys())
    remaining = float(c_total)

    # 首先满足维持呼吸
    x = {}
    for org in organs:
        req = maint_req.get(org, 0.0)
        alloc = min(req, remaining)
        x[org] = alloc
        remaining -= alloc

    if remaining <= 0:
        return x, sum(x[o] * coeffs[o] for o in organs)

    # 剩余碳按边际收益排序分配（贪心最优，因为 LP 无耦合约束）
    sorted_orgs = sorted(organs, key=lambda o: coeffs[o], reverse=True)
    for org in sorted_orgs:
        x[org] += remaining
        remaining = 0.0
        break

    obj = sum(x[o] * coeffs[o] for o in organs)
    return x, obj


def parse_solution_vector(sol_vec, organ_names):
    """
    解析解向量，确保非负并四舍五入。
    模拟 cplex_solution_read 对解的清洗。
    """
    sol = np.asarray(sol_vec, dtype=float)
    sol = np.abs(sol)
    sol = np.round(sol, decimals=6)
    sol = np.maximum(sol, 0.0)
    return dict(zip(organ_names, sol))


def shadow_prices(c_total, coeffs, maint_req):
    """
    计算各约束的影子价格（敏感性分析）。
    """
    organs = list(coeffs.keys())
    _, obj_base = solve_carbon_lp(c_total, coeffs, maint_req)
    eps = 0.01
    shadows = {}
    for org in organs:
        req_new = maint_req.copy()
        req_new[org] += eps
        _, obj_new = solve_carbon_lp(c_total, coeffs, req_new)
        shadows[org] = (obj_new - obj_base) / eps
    return shadows
