
import numpy as np


def atbash_mirror_map(nuclide_list):
    mirrored = []
    for z, n, a in nuclide_list:
        mirrored.append((n, z, a))
    return mirrored


def monoalphabetic_nuclide_encode(nuclide_list, key_seed=42):
    rng = np.random.default_rng(key_seed)
    indices = np.arange(len(nuclide_list))
    rng.shuffle(indices)
    encode_map = {}
    decode_map = {}
    for i, (z, n, a) in enumerate(nuclide_list):
        code = int(indices[i])
        encode_map[(z, n)] = code
        decode_map[code] = (z, n, a)
    return encode_map, decode_map


def is_gaussian_prime(a, b):
    def is_prime(n):
        if n < 2:
            return False
        if n % 2 == 0:
            return n == 2
        i = 3
        while i * i <= n:
            if n % i == 0:
                return False
            i += 2
        return True

    if a == 0:
        return is_prime(abs(b)) and (abs(b) % 4 == 3)
    if b == 0:
        return is_prime(abs(a)) and (abs(a) % 4 == 3)
    return is_prime(a * a + b * b)


def gaussian_prime_spiral_trajectory(c_start, d_start, max_steps=1000):
    c = complex(c_start)
    d = complex(d_start)
    trajectory = [c]
    visited = {c}
    for _ in range(max_steps):
        c = c + d
        a, b = int(round(c.real)), int(round(c.imag))
        c = complex(a, b)
        if is_gaussian_prime(a, b):
            d = d * 1j
        if c in visited:
            break
        trajectory.append(c)
        visited.add(c)
    return trajectory


def build_nuclide_grid_path(z_min, z_max, n_min, n_max):
    center_z = (z_min + z_max) // 2
    center_n = (n_min + n_max) // 2
    c_start = complex(center_n, center_z)
    trajectory = gaussian_prime_spiral_trajectory(c_start, 1, max_steps=2000)
    path = []
    for c in trajectory:
        n, z = int(round(c.real)), int(round(c.imag))
        if z_min <= z <= z_max and n_min <= n <= n_max:
            path.append((z, n))

    seen = set()
    unique_path = []
    for p in path:
        if p not in seen:
            unique_path.append(p)
            seen.add(p)
    return unique_path


def test_nuclide_encoding():
    nuclides = [(26, 30, 56), (82, 126, 208), (92, 146, 238)]
    mirrored = atbash_mirror_map(nuclides)
    print(f"[nuclide_encoding] Mirror mapping: {nuclides} -> {mirrored}")

    enc, dec = monoalphabetic_nuclide_encode(nuclides, key_seed=123)
    print(f"[nuclide_encoding] Encoding map: {enc}")


    traj = gaussian_prime_spiral_trajectory(0+0j, 1, max_steps=100)
    print(f"[nuclide_encoding] Gaussian prime spiral length: {len(traj)}")

    path = build_nuclide_grid_path(20, 40, 20, 50)
    print(f"[nuclide_encoding] Nuclide grid path length: {len(path)}")


if __name__ == "__main__":
    test_nuclide_encoding()
