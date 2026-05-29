"""
碳分配模块：基于 change_greedy 贪心算法，
模拟光合作用产物在叶、干、根之间的资源分配。

核心公式：
  总碳收入：
      C_total = integral A_n(z) * LAI(z) dz  [gC/m^2/day]

  分配优先级（贪心策略，按边际收益递减）：
      1. 维持呼吸（最高优先级）
      2. 叶生长
      3. 干生长
      4. 根生长
      5. 储存

  边际收益（基于 Allometric 生长模型）：
      MR_leaf = dW_leaf / dC_leaf = 1 / (c_leaf * W_leaf^eta)
      MR_stem = dW_stem / dC_stem = 1 / (c_stem * W_stem^eta)
      MR_root = dW_root / dC_root = 1 / (c_root * W_root^eta)
"""
import numpy as np


def greedy_carbon_allocation(c_total, demands, current_biomass,
                             c_costs, eta=0.75):
    """
    贪心碳分配算法。
    c_total: 总可用碳 (gC/m^2)
    demands: dict {'leaf': demand, 'stem': demand, 'root': demand, 'storage': demand}
    current_biomass: dict {'leaf': w, 'stem': w, 'root': w}
    c_costs: dict 各器官的碳转化效率 (g biomass / g C)
    eta: 异速生长指数
    返回: allocated dict
    """
    organs = ['leaf', 'stem', 'root', 'storage']
    allocated = {k: 0.0 for k in organs}
    remaining = float(c_total)

    # 计算边际收益并排序（贪心）
    mr = []
    for org in organs:
        if org == 'storage':
            mr_val = 0.5  # 储存的边际收益最低
        else:
            w = max(current_biomass.get(org, 1.0), 1.0)
            cost = c_costs.get(org, 2.0)
            mr_val = 1.0 / (cost * (w ** eta))
        mr.append((mr_val, org))

    mr.sort(reverse=True)

    for mr_val, org in mr:
        demand = demands.get(org, 0.0)
        alloc = min(demand, remaining)
        allocated[org] = alloc
        remaining -= alloc
        if remaining <= 1e-6:
            break

    return allocated


def allometric_biomass_update(allocated, current_biomass, c_costs, dt_days=1.0):
    """
    更新生物量。
    返回: 新的生物量 dict
    """
    new_biomass = {}
    for org, w in current_biomass.items():
        cost = c_costs.get(org, 2.0)
        delta_w = allocated.get(org, 0.0) / max(cost, 1e-6) * dt_days
        new_biomass[org] = w + delta_w
    return new_biomass


def compute_allocation_efficiency(allocated, demands):
    """计算分配效率（满足需求的比例）。"""
    total_demand = sum(demands.values())
    total_alloc = sum(allocated.values())
    if total_demand < 1e-9:
        return 1.0
    return min(total_alloc / total_demand, 1.0)
