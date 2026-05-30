import numpy as np


def greedy_carbon_allocation(c_total, demands, current_biomass,
                             c_costs, eta=0.75):
    organs = ['leaf', 'stem', 'root', 'storage']
    allocated = {k: 0.0 for k in organs}
    remaining = float(c_total)


    mr = []
    for org in organs:
        if org == 'storage':
            mr_val = 0.5
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
    new_biomass = {}
    for org, w in current_biomass.items():
        cost = c_costs.get(org, 2.0)
        delta_w = allocated.get(org, 0.0) / max(cost, 1e-6) * dt_days
        new_biomass[org] = w + delta_w
    return new_biomass


def compute_allocation_efficiency(allocated, demands):
    total_demand = sum(demands.values())
    total_alloc = sum(allocated.values())
    if total_demand < 1e-9:
        return 1.0
    return min(total_alloc / total_demand, 1.0)
