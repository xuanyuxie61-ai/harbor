
import numpy as np
from typing import List, Tuple


class LocalState:

    def __init__(self,
                 peclet: float,
                 cfl: float,
                 grad: float,
                 curvature: float):
        self.peclet = peclet
        self.cfl = cfl
        self.grad = grad
        self.curvature = curvature


class NumericalStrategy:

    STRATEGIES = ["centered", "upwind", "lax_wendroff", "shock_capturing"]

    def __init__(self, name: str):
        if name not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy {name}")
        self.name = name

    def estimate_error(self, state: LocalState) -> float:
        if self.name == "centered":

            if state.peclet > 2.0:
                return 1e3
            return state.cfl ** 2 + state.curvature * 1e-3
        elif self.name == "upwind":

            return state.cfl + state.grad * 1e-2
        elif self.name == "lax_wendroff":

            return state.cfl ** 2 + state.grad * 1e-3 + abs(state.cfl - 1.0) * 1e-2
        elif self.name == "shock_capturing":

            if state.curvature < 10.0:
                return 1.0
            return 0.1 * state.curvature
        return 1.0


class StrategySelector:

    def __init__(self):
        self.strategies = [NumericalStrategy(s) for s in NumericalStrategy.STRATEGIES]

    def evaluate_state(self,
                       u: np.ndarray,
                       x: np.ndarray,
                       v: float,
                       epsilon: float,
                       dt: float) -> List[LocalState]:
        nx = len(u)
        states = []
        for i in range(nx):
            hp = x[min(i + 1, nx - 1)] - x[i] if i < nx - 1 else x[i] - x[max(i - 1, 0)]
            hm = x[i] - x[max(i - 1, 0)] if i > 0 else hp
            h = 0.5 * (hp + hm)
            pe = abs(v) * h / epsilon if epsilon > 0 else 1e6
            cfl = abs(v) * dt / h if h > 0 else 0.0

            if 0 < i < nx - 1:
                grad = (u[i + 1] - u[i - 1]) / (hp + hm)
                curvature = abs((u[i + 1] - 2 * u[i] + u[i - 1]) / (0.5 * (hp + hm)) ** 2)
            else:
                grad = 0.0
                curvature = 0.0

            states.append(LocalState(pe, cfl, abs(grad), curvature))
        return states

    def select_best_strategies(self,
                               u: np.ndarray,
                               x: np.ndarray,
                               v: float,
                               epsilon: float,
                               dt: float) -> List[str]:
        states = self.evaluate_state(u, x, v, epsilon, dt)
        best = []
        for state in states:
            scores = [(s.name, -np.log10(s.estimate_error(state) + 1e-16)) for s in self.strategies]
            scores.sort(key=lambda item: item[1], reverse=True)
            best.append(scores[0][0])
        return best

    def aggregate_recommendation(self,
                                 u: np.ndarray,
                                 x: np.ndarray,
                                 v: float,
                                 epsilon: float,
                                 dt: float) -> str:
        best_per_node = self.select_best_strategies(u, x, v, epsilon, dt)
        from collections import Counter
        c = Counter(best_per_node)
        return c.most_common(1)[0][0]
