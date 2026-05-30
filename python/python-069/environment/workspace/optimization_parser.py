import numpy as np


def solve_carbon_lp(c_total, coeffs, maint_req):
    organs = list(coeffs.keys())
    remaining = float(c_total)


    x = {}
    for org in organs:
        req = maint_req.get(org, 0.0)
        alloc = min(req, remaining)
        x[org] = alloc
        remaining -= alloc

    if remaining <= 0:
        return x, sum(x[o] * coeffs[o] for o in organs)


    sorted_orgs = sorted(organs, key=lambda o: coeffs[o], reverse=True)
    for org in sorted_orgs:
        x[org] += remaining
        remaining = 0.0
        break

    obj = sum(x[o] * coeffs[o] for o in organs)
    return x, obj


def parse_solution_vector(sol_vec, organ_names):
    sol = np.asarray(sol_vec, dtype=float)
    sol = np.abs(sol)
    sol = np.round(sol, decimals=6)
    sol = np.maximum(sol, 0.0)
    return dict(zip(organ_names, sol))


def shadow_prices(c_total, coeffs, maint_req):
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
