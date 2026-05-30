
import numpy as np


def strategy_random_search(discriminant_func, param_bounds, n_trials=10000, threshold=1e-3, seed=42):
    rng = np.random.default_rng(seed)
    names = list(param_bounds.keys())
    bounds = [param_bounds[n] for n in names]
    candidates = []

    for _ in range(n_trials):
        pdict = {}
        for name, (lo, hi) in zip(names, bounds):
            pdict[name] = lo + rng.random() * (hi - lo)
        delta = discriminant_func(pdict)
        if abs(delta) < threshold:
            candidates.append({'params': pdict, 'delta': delta, 'abs_delta': abs(delta)})

    return candidates


def strategy_importance_sampling(discriminant_func, param_bounds, n_trials=10000, temperature=1e-2, seed=42):
    rng = np.random.default_rng(seed)
    names = list(param_bounds.keys())
    bounds = [param_bounds[n] for n in names]
    candidates = []

    for _ in range(n_trials):
        pdict = {}
        for name, (lo, hi) in zip(names, bounds):
            pdict[name] = lo + rng.random() * (hi - lo)
        delta = discriminant_func(pdict)
        abs_delta = abs(delta)
        weight = np.exp(-abs_delta / temperature)
        if rng.random() < weight:
            candidates.append({'params': pdict, 'delta': delta, 'abs_delta': abs_delta})

    return candidates


def strategy_adaptive_local_search(discriminant_func, param_bounds, initial_guess, n_iter=5000, step=0.01, seed=42):
    rng = np.random.default_rng(seed)
    names = list(param_bounds.keys())
    bounds = [param_bounds[n] for n in names]

    current = dict(initial_guess)
    current_delta = discriminant_func(current)
    current_abs = abs(current_delta)

    best = {'params': dict(current), 'delta': current_delta, 'abs_delta': current_abs}

    for _ in range(n_iter):
        proposal = {}
        for name, (lo, hi) in zip(names, bounds):
            val = current[name] + step * (rng.random() - 0.5) * (hi - lo)
            val = max(lo, min(hi, val))
            proposal[name] = val

        delta = discriminant_func(proposal)
        abs_delta = abs(delta)

        if abs_delta < current_abs:
            current = proposal
            current_delta = delta
            current_abs = abs_delta
            if abs_delta < best['abs_delta']:
                best = {'params': dict(current), 'delta': current_delta, 'abs_delta': current_abs}

    return best


def metropolis_hastings_ep_search(discriminant_func, param_bounds, n_steps=20000, beta=1e3, step=0.02, seed=42):
    rng = np.random.default_rng(seed)
    names = list(param_bounds.keys())
    bounds = [param_bounds[n] for n in names]

    current = {}
    for name, (lo, hi) in zip(names, bounds):
        current[name] = lo + rng.random() * (hi - lo)
    current_delta = discriminant_func(current)
    current_energy = abs(current_delta)

    chain = []
    accepted = 0

    for _ in range(n_steps):
        proposal = {}
        for name, (lo, hi) in zip(names, bounds):
            val = current[name] + step * (hi - lo) * (rng.random() - 0.5)
            val = max(lo, min(hi, val))
            proposal[name] = val

        delta = discriminant_func(proposal)
        energy = abs(delta)
        dE = energy - current_energy

        if dE < 0 or rng.random() < np.exp(-beta * dE):
            current = proposal
            current_delta = delta
            current_energy = energy
            accepted += 1

        chain.append({'params': dict(current), 'delta': current_delta, 'abs_delta': current_energy})

    accept_rate = accepted / n_steps
    return chain, accept_rate
