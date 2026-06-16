"""
markov_transitions.py
=====================

Markov and semi-Markov transition models for parton shower evolution
and partonic state transitions:
- Markov chain state machine (from 1295_jackdeadman_turn-taking)
- Distribution fitting and sampling (from 1295_jackdeadman_turn-taking)

Scientific context:
-------------------
The parton shower in QCD is a Markov process in the space of partonic
states. At each branching step, a parton of type i (quark or gluon)
transitions to a pair (j, k) with probability given by the splitting
function P_{i->jk}(z).

The Sudakov form factor gives the no-branching probability:
    Delta_i(t1, t2) = exp(-int_{t1}^{t2} dt/t * int dz * alpha_s/(2*pi) * P_{i->jk}(z))

The Markov chain has states:
    {q, qbar, g, qg, qqbar, gg, qgg, ...}

We model this as a discrete-time Markov chain with transition matrix
T_{ij} = probability of going from state i to state j in one step.

The semi-Markov extension allows the waiting time between transitions
to follow a non-exponential distribution, which is more physical for
the evolution in ln(Q^2).
"""

import math
import random
from typing import Dict, List, Optional, Tuple


# ===========================================================================
# Section 1: Markov state machine (from 1295_jackdeadman_turn-taking)
# ===========================================================================

class MarkovState:
    """
    A single state in a Markov chain.

    Attributes
    ----------
    name : str
        State identifier (e.g., 'q', 'g', 'qg').
    transitions : dict
        {next_state_name: transition_probability}.
    """

    def __init__(self, name: str):
        self.name = name
        self.transitions: Dict[str, float] = {}

    def add_transition(self, target: str, prob: float):
        if prob < 0.0:
            raise ValueError(f"MarkovState: negative prob {prob} for {self.name}->{target}")
        self.transitions[target] = self.transitions.get(target, 0.0) + prob

    def normalize(self):
        total = sum(self.transitions.values())
        if total > 0.0:
            for k in self.transitions:
                self.transitions[k] /= total

    def sample_next(self, rng: random.Random) -> str:
        if not self.transitions:
            return self.name
        u = rng.random()
        cum = 0.0
        for target, prob in self.transitions.items():
            cum += prob
            if u <= cum:
                return target
        return list(self.transitions.keys())[-1]


class MarkovChain:
    """
    Discrete-time Markov chain with named states.

    Used to model the parton shower branching sequence:
        g -> q qbar (with prob proportional to alpha_s * P_{g->qqbar})
        g -> g g (with prob proportional to alpha_s * P_{g->gg})
        q -> q g (with prob proportional to alpha_s * P_{q->qg})
    """

    def __init__(self):
        self.states: Dict[str, MarkovState] = {}

    def add_state(self, name: str) -> MarkovState:
        if name not in self.states:
            self.states[name] = MarkovState(name)
        return self.states[name]

    def add_transition(self, src: str, dst: str, prob: float):
        s = self.add_state(src)
        self.add_state(dst)
        s.add_transition(dst, prob)

    def normalize_all(self):
        for s in self.states.values():
            s.normalize()

    def build_parton_shower_chain(self, alpha_s: float):
        """
        Build a simplified parton shower Markov chain with states:
            q (quark), g (gluon), qg (quark+gluon), gg (2 gluons), qqbar (q-qbar pair)

        Transition probabilities are proportional to the LO splitting
        functions integrated over z in [0.1, 0.9] (soft/collinear cutoff).
        """
        cf = 4.0 / 3.0
        ca = 3.0
        tf = 0.5
        # Integrated splitting functions (approximate)
        int_pqq = cf * 1.2  # ~ integral of P_qq(z) dz
        int_pgg = ca * 2.5  # ~ integral of P_gg(z) dz
        int_pgq = tf * 0.8  # ~ integral of P_gq(z) dz
        # Scale by alpha_s / (2*pi)
        prefactor = alpha_s / (2.0 * math.pi)
        # Quark can emit gluon
        self.add_transition('q', 'qg', prefactor * int_pqq)
        self.add_transition('q', 'q', 1.0 - prefactor * int_pqq)
        # Gluon can split to gg or qqbar
        self.add_transition('g', 'gg', prefactor * int_pgg)
        self.add_transition('g', 'qqbar', prefactor * int_pgq)
        self.add_transition('g', 'g', max(0.0, 1.0 - prefactor * (int_pgg + int_pgq)))
        # Terminal states
        self.add_transition('qg', 'qg', 1.0)
        self.add_transition('gg', 'gg', 1.0)
        self.add_transition('qqbar', 'qqbar', 1.0)
        self.normalize_all()

    def sample(self, start: str, n_steps: int,
               rng: random.Random) -> List[str]:
        """Generate a trajectory of length n_steps from start state."""
        if start not in self.states:
            raise ValueError(f"MarkovChain.sample: unknown state '{start}'")
        trajectory = [start]
        current = start
        for _ in range(n_steps):
            current = self.states[current].sample_next(rng)
            trajectory.append(current)
        return trajectory

    def transition_matrix(self, state_order: List[str]
                          ) -> List[List[float]]:
        """Return the transition matrix as a list of lists."""
        n = len(state_order)
        idx = {s: i for i, s in enumerate(state_order)}
        mat = [[0.0] * n for _ in range(n)]
        for i, si in enumerate(state_order):
            if si in self.states:
                for dst, prob in self.states[si].transitions.items():
                    if dst in idx:
                        mat[i][idx[dst]] = prob
        return mat


# ===========================================================================
# Section 2: Distribution fitting (from 1295_jackdeadman_turn-taking)
# ===========================================================================

class BranchingTimeDistribution:
    """
    Distribution of branching times (in ln(Q^2)) for the semi-Markov
    parton shower model.

    We use a log-normal distribution:
        f(t) = 1/(t * sigma * sqrt(2*pi)) * exp(-(ln(t) - mu)^2 / (2*sigma^2))

    which captures the broad distribution of branching scales observed
    in parton showers.
    """

    def __init__(self):
        self.mu = 0.0
        self.sigma = 1.0
        self._fitted = False

    def fit(self, data: List[float]):
        """Fit log-normal to data using MLE:
            mu = mean(ln(data))
            sigma = std(ln(data))
        """
        if len(data) < 2:
            self.mu = 0.0
            self.sigma = 1.0
            self._fitted = True
            return
        log_data = [math.log(max(1e-10, d)) for d in data]
        n = len(log_data)
        self.mu = sum(log_data) / n
        var = sum((x - self.mu) ** 2 for x in log_data) / (n - 1)
        self.sigma = math.sqrt(max(var, 1e-10))
        self._fitted = True

    def sample(self, rng: random.Random) -> float:
        """Sample a branching time from the log-normal distribution."""
        if not self._fitted:
            raise RuntimeError("BranchingTimeDistribution: not fitted yet")
        z = rng.gauss(self.mu, self.sigma)
        return math.exp(z)

    def pdf(self, t: float) -> float:
        """Evaluate the log-normal PDF at t."""
        if t <= 0.0:
            return 0.0
        lt = math.log(t)
        z = (lt - self.mu) / self.sigma
        return (math.exp(-0.5 * z * z)
                / (t * self.sigma * math.sqrt(2.0 * math.pi)))


def generate_branching_times(n: int, seed: int = 123
                             ) -> Tuple[List[float], BranchingTimeDistribution]:
    """
    Generate n branching times from a reference distribution.

    Returns the data and the fitted distribution object.
    """
    rng = random.Random(seed)
    # Generate from a known log-normal
    true_mu = 1.0
    true_sigma = 0.5
    data = [math.exp(rng.gauss(true_mu, true_sigma)) for _ in range(n)]
    dist = BranchingTimeDistribution()
    dist.fit(data)
    return data, dist


def sudakov_form_factor(t1: float, t2: float,
                        gamma: float) -> float:
    """
    Compute the Sudakov no-branching probability:

        Delta(t1, t2) = exp(-gamma * (t2 - t1))

    where gamma is the integrated splitting function * alpha_s/(2*pi).
    """
    if t2 < t1:
        t1, t2 = t2, t1
    arg = -gamma * (t2 - t1)
    if arg < -50.0:
        return 0.0
    return math.exp(arg)
