
import numpy as np


def cvt_energy(generators, samples):


    diff = samples[:, np.newaxis, :] - generators[np.newaxis, :, :]
    dists = np.sum(diff ** 2, axis=2)
    assignments = np.argmin(dists, axis=1)
    min_dists = dists[np.arange(len(samples)), assignments]
    energy = np.mean(min_dists)
    return energy, assignments


def cvt_iterate(generators, samples, density=None):
    n_gen = generators.shape[0]
    energy, assignments = cvt_energy(generators, samples)
    
    new_generators = np.zeros_like(generators)
    counts = np.zeros(n_gen)
    
    if density is None:
        density = np.ones(len(samples))
    
    for i in range(n_gen):
        mask = (assignments == i)
        if np.any(mask):
            weights = density[mask]
            new_generators[i] = np.sum(weights[:, np.newaxis] * samples[mask], axis=0) / np.sum(weights)
            counts[i] = np.sum(mask)
        else:

            new_generators[i] = generators[i]
    
    diff = np.linalg.norm(new_generators - generators)
    return new_generators, diff, energy


def generate_cvt_samples(dim, n_gen, n_samples=10000, it_max=50, 
                         bounds=None, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    if bounds is None:
        bounds = [(-1.0, 1.0)] * dim
    bounds = np.asarray(bounds)
    

    generators = np.random.rand(n_gen, dim)
    generators = bounds[:, 0] + generators * (bounds[:, 1] - bounds[:, 0])
    
    for it in range(it_max):

        samples = np.random.rand(n_samples, dim)
        samples = bounds[:, 0] + samples * (bounds[:, 1] - bounds[:, 0])
        
        generators, diff, energy = cvt_iterate(generators, samples)
        
        if diff < 1e-6:
            break
    
    return generators


def cvt_quadrature_weights(generators, samples=None, n_samples=50000):
    if samples is None:
        dim = generators.shape[1]
        samples = np.random.rand(n_samples, dim) * 2.0 - 1.0
    
    _, assignments = cvt_energy(generators, samples)
    n_gen = generators.shape[0]
    weights = np.zeros(n_gen)
    for i in range(n_gen):
        weights[i] = np.mean(assignments == i)
    return weights


def adaptive_cvt_for_reliability(dim, n_gen, beta_sphere_radius,
                                  n_samples=20000, it_max=40, seed=None):
    if seed is not None:
        np.random.seed(seed)
    

    generators = np.random.randn(n_gen, dim)
    norms = np.linalg.norm(generators, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    generators = generators / norms * beta_sphere_radius
    
    for it in range(it_max):

        samples = np.random.randn(n_samples, dim)
        sample_norms = np.linalg.norm(samples, axis=1)
        sample_norms = np.maximum(sample_norms, 1e-10)
        

        radial_factor = np.exp(-0.5 * (sample_norms - beta_sphere_radius) ** 2)
        
        generators, diff, energy = cvt_iterate(generators, samples, density=radial_factor)
        
        if diff < 1e-5:
            break
    
    weights = cvt_quadrature_weights(generators, n_samples=n_samples)
    return generators, weights
