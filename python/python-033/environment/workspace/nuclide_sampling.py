
import numpy as np


def cvt_1d_nonuniform(n_generators, density_type='power', n_samples=50000,
                       n_steps=50, init_type='grid'):

    if init_type == 'random':
        generators = np.sort(np.random.rand(n_generators))
    elif init_type == 'grid':
        generators = np.linspace(0.0, 1.0, n_generators)
    else:
        generators = np.zeros(n_generators)

    for step in range(n_steps):

        u = np.random.rand(n_samples)
        if density_type == 'uniform':
            samples = u
        elif density_type == 'sqrt':
            samples = u ** 2
        elif density_type == 'power':
            samples = u ** 0.5
        elif density_type == 'log':
            samples = np.log(1.0 + (np.e - 1.0) * u)
        elif density_type == 'arctan':
            samples = np.tan(np.pi / 4.0 * u)
            samples = np.clip(samples, 0.0, 1.0)
        elif density_type == 'chebyshev':
            samples = np.sin(np.pi / 2.0 * u) ** 2
        else:
            samples = u



        indices = np.argmin(np.abs(samples[:, None] - generators[None, :]), axis=1)


        new_generators = np.zeros(n_generators)
        for i in range(n_generators):
            cell_samples = samples[indices == i]
            if len(cell_samples) > 0:
                new_generators[i] = np.mean(cell_samples)
            else:

                new_generators[i] = generators[i]


        new_generators = np.sort(new_generators)
        new_generators[0] = max(new_generators[0], 0.0)
        new_generators[-1] = min(new_generators[-1], 1.0)

        generators = new_generators

    return generators


def sample_nuclide_mass_chain(a_min, a_max, n_nuclides,
                               density_profile='r_process_path'):
    n_generators = n_nuclides
    generators = cvt_1d_nonuniform(n_generators, density_type='power',
                                    n_samples=20000, n_steps=30, init_type='grid')

    a_values = a_min + generators * (a_max - a_min)
    a_values = np.round(a_values).astype(int)
    a_values = np.clip(a_values, a_min, a_max)
    a_values = np.unique(a_values)
    return a_values


def build_r_process_nuclide_set(a_values, beta_stability_offset=5):
    nuclides = []
    for A in a_values:
        if A <= 0:
            continue

        Z_stable = int(A / (1.98 + 0.0158 * (A ** (2.0 / 3.0))))
        Z_stable = max(1, min(Z_stable, A - 1))

        N_stable = A - Z_stable
        N_rp = N_stable + beta_stability_offset
        Z_rp = A - N_rp
        if Z_rp < 1:
            Z_rp = 1
            N_rp = A - 1
        nuclides.append((int(Z_rp), int(N_rp), int(A)))
    return nuclides


def test_nuclide_sampling():
    gens = cvt_1d_nonuniform(20, density_type='power', n_steps=20)
    print(f"[nuclide_sampling] CVT generators range: [{gens[0]:.4f}, {gens[-1]:.4f}]")

    a_vals = sample_nuclide_mass_chain(80, 240, 30)
    print(f"[nuclide_sampling] Sampled mass numbers: {a_vals[:10]} ...")

    nuclides = build_r_process_nuclide_set(a_vals, beta_stability_offset=8)
    print(f"[nuclide_sampling] Built {len(nuclides)} r-process nuclides")
    if nuclides:
        print(f"[nuclide_sampling] Example: Z={nuclides[0][0]}, N={nuclides[0][1]}, A={nuclides[0][2]}")


if __name__ == "__main__":
    test_nuclide_sampling()
