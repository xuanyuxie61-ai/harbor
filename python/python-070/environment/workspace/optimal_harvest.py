
import numpy as np
from utils import NumericalConfig, safe_divide


class BrentOptimizer:

    def __init__(self, a, b, tol=None):
        if b <= a:
            raise ValueError("Require a < b for Brent optimizer")
        self.a = float(a)
        self.b = float(b)
        self.c = 0.5 * (3.0 - np.sqrt(5.0))
        self.eps_sqrt = NumericalConfig.EPS_SQRT
        self.tol = tol if tol is not None else NumericalConfig.TOL

        self.v = self.a + self.c * (self.b - self.a)
        self.w = self.v
        self.x = self.v
        self.e = 0.0
        self.fx = None
        self.fv = None
        self.fw = None
        self.status = 1
        self.arg = self.x
        self._first_call = True

    def step(self, value):
        if self._first_call:
            self.fx = value
            self.fv = self.fx
            self.fw = self.fx
            self._first_call = False
            self.status = 2
            self.arg = self.x
            return self.arg, self.status

        fu = value

        if fu <= self.fx:
            if self.x <= self.arg:
                self.a = self.x
            else:
                self.b = self.x
            self.v = self.w
            self.fv = self.fw
            self.w = self.x
            self.fw = self.fx
            self.x = self.arg
            self.fx = fu
        else:
            if self.arg < self.x:
                self.a = self.arg
            else:
                self.b = self.arg
            if fu <= self.fw or self.w == self.x:
                self.v = self.w
                self.fv = self.fw
                self.w = self.arg
                self.fw = fu
            elif fu <= self.fv or self.v == self.x or self.v == self.w:
                self.v = self.arg
                self.fv = fu

        midpoint = 0.5 * (self.a + self.b)
        tol1 = self.eps_sqrt * abs(self.x) + self.tol / 3.0
        tol2 = 2.0 * tol1

        if abs(self.x - midpoint) <= (tol2 - 0.5 * (self.b - self.a)):
            self.status = 0
            self.arg = self.x
            return self.arg, self.status

        if abs(self.e) <= tol1:
            if midpoint <= self.x:
                self.e = self.a - self.x
            else:
                self.e = self.b - self.x
            d = self.c * self.e
        else:
            r_val = (self.x - self.w) * (self.fx - self.fv)
            q_val = (self.x - self.v) * (self.fx - self.fw)
            p_val = (self.x - self.v) * q_val - (self.x - self.w) * r_val
            q_val = 2.0 * (q_val - r_val)
            if 0.0 < q_val:
                p_val = -p_val
            q_val = abs(q_val)
            r_val = self.e
            self.e = d if hasattr(self, 'd') else 0.0

            if (abs(0.5 * q_val * r_val) <= abs(p_val)) and (p_val > q_val * (self.a - self.x)) and (p_val < q_val * (self.b - self.x)):
                d = p_val / q_val
                u = self.x + d
                if (u - self.a) < tol2:
                    d = tol1 * np.sign(midpoint - self.x)
                if (self.b - u) < tol2:
                    d = tol1 * np.sign(midpoint - self.x)
            else:
                if midpoint <= self.x:
                    self.e = self.a - self.x
                else:
                    self.e = self.b - self.x
                d = self.c * self.e

        if abs(d) >= tol1:
            u = self.x + d
        else:
            u = self.x + tol1 * np.sign(d)

        self.arg = u
        self.status += 1
        return self.arg, self.status

    def optimize(self, func):
        arg, status = self.arg, self.status
        while status > 0:
            val = func(arg)
            arg, status = self.step(val)
        return arg, func(arg)


def schaefer_gordon_steady_state(E, r, K, q):
    E = np.asarray(E, dtype=float)
    B = K * (1.0 - q * E / r)
    B = np.where(q * E <= r, B, 0.0)
    return np.maximum(B, 0.0)


def discounted_profit_objective(E, r, K, q, p, c, delta, T=50.0):






    pass


def find_optimal_effort(r, K, q, p, c, delta, T=50.0, E_max=None):
    if E_max is None:
        E_max = r / q

    def obj(E):
        return discounted_profit_objective(E, r, K, q, p, c, delta, T)

    optimizer = BrentOptimizer(0.0, E_max)
    E_opt, neg_profit = optimizer.optimize(obj)
    return E_opt, -neg_profit


def bellman_ford_shortest_paths(v_num, e_list, e_weight, source):
    e_weight = np.asarray(e_weight, dtype=float)
    dist = np.full(v_num, NumericalConfig.R8_BIG, dtype=float)
    dist[source] = 0.0
    predecessor = np.full(v_num, -1, dtype=int)


    for _ in range(v_num - 1):
        updated = False
        for j, (u, v) in enumerate(e_list):
            if dist[u] + e_weight[j] < dist[v] - NumericalConfig.TOL:
                dist[v] = dist[u] + e_weight[j]
                predecessor[v] = u
                updated = True
        if not updated:
            break


    for j, (u, v) in enumerate(e_list):
        if dist[u] + e_weight[j] < dist[v] - NumericalConfig.TOL:
            raise RuntimeError("Graph contains a negative-weight cycle")

    return dist, predecessor


def mpa_network_optimize(n_patches, connectivity_matrix, source_patch=0):
    e_list = []
    e_weight = []
    for i in range(n_patches):
        for j in range(n_patches):
            if i != j and not np.isinf(connectivity_matrix[i, j]):
                e_list.append((i, j))
                e_weight.append(connectivity_matrix[i, j])

    dist, predecessor = bellman_ford_shortest_paths(n_patches, e_list, e_weight, source_patch)
    return dist, predecessor


def reconstruct_path(predecessor, target):
    path = []
    v = target
    while v != -1:
        path.append(int(v))
        v = predecessor[v]
    path.reverse()
    return path
