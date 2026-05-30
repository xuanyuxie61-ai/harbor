
import numpy as np


def gf2_add(p, q):
    return p ^ q


def gf2_multiply(p, q):
    result = 0
    while q:
        if q & 1:
            result ^= p
        p <<= 1
        q >>= 1
    return result


def gf2_mod(p, mod_poly):
    deg_mod = mod_poly.bit_length() - 1
    if deg_mod < 0:
        raise ValueError("模多项式不能为 0")

    while p.bit_length() > deg_mod:
        shift = p.bit_length() - deg_mod - 1
        p ^= mod_poly << shift
    return p


def gf2_poly_degree(p):
    if p == 0:
        return -1
    return p.bit_length() - 1


def gf2_poly_string(p):
    if p == 0:
        return "0"
    terms = []
    for i in range(gf2_poly_degree(p) + 1):
        if (p >> i) & 1:
            if i == 0:
                terms.append("1")
            elif i == 1:
                terms.append("x")
            else:
                terms.append(f"x^{i}")
    return " + ".join(reversed(terms))


def parity_operator_state(l):
    return 1 if l % 2 == 0 else -1


def spin_orbit_coupling_gf2(l):

    j_plus = 0b10
    j_minus = 0b01




    transition = {
        'j_plus': j_plus,
        'j_minus': j_minus,
        'degeneracy': 2,
    }
    return transition


def nuclear_configuration_gf2(n_particles, n_states):
    if n_particles > n_states:
        return []
    configurations = []

    def generate(pos, ones_left, current):
        if ones_left == 0:
            configurations.append(current)
            return
        if pos < 0:
            return

        generate(pos - 1, ones_left - 1, current | (1 << pos))

        generate(pos - 1, ones_left, current)

    generate(n_states - 1, n_particles, 0)
    return configurations


def isospin_states(N, Z):
    Tz = (N - Z) / 2.0
    min_T = abs(Tz)

    max_T = (N + Z) / 2.0

    states = []
    T = min_T
    while T <= max_T:
        multiplicity = int(2 * T + 1)
        states.append({'T': T, 'Tz': Tz, 'multiplicity': multiplicity})
        T += 1.0

    return states


def time_reversal_symmetry_check(J, config_gf2):
    if abs(J - round(J)) < 0.25:

        return {'kramers_degeneracy': 1, 'time_reversal_even': True}
    else:

        return {'kramers_degeneracy': 2, 'time_reversal_even': False}


def shell_model_parity(configuration, shell_parity_list):
    parity = 1
    for i, p in enumerate(shell_parity_list):
        if (configuration >> i) & 1:
            parity *= p
    return parity


def young_tableau_dimension(partition):
    n = sum(partition)



    if len(partition) == 1:
        return 1

    if len(partition) == 2:
        a, b = partition
        from math import factorial
        return factorial(a + b) * (a - b + 1) // (factorial(a + 1) * factorial(b))
    return 1


if __name__ == "__main__":

    p = 0b1011
    q = 0b110
    print(f"p = {gf2_poly_string(p)}")
    print(f"q = {gf2_poly_string(q)}")
    print(f"p+q = {gf2_poly_string(gf2_add(p, q))}")
    print(f"p*q = {gf2_poly_string(gf2_multiply(p, q))}")

    print(f"l=2 宇称: {parity_operator_state(2)}")
    print(f"l=3 宇称: {parity_operator_state(3)}")

    configs = nuclear_configuration_gf2(2, 4)
    print(f"4 态中填 2 粒子: {len(configs)} 种组态")
    print([bin(c) for c in configs])

    iso = isospin_states(30, 26)
    print(f"56Fe 同位旋态: {iso}")

    tr = time_reversal_symmetry_check(2.5, configs[0])
    print(f"J=5/2 时间反演检验: {tr}")
