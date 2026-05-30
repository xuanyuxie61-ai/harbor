
from typing import List


def _sum_of_digits(val: int) -> int:
    s = 0
    while val > 0:
        s += val % 10
        val //= 10
    return s


def topology_checksum(topo_string: str) -> int:
    digits = [int(ch) for ch in topo_string if ch.isdigit()]
    if len(digits) == 0:
        return 1
    n = len(digits)
    total = 0

    for i in range(n - 1, -1, -2):
        total += digits[i]

    for i in range(n - 2, -1, -2):
        doubled = digits[i] * 2
        total += _sum_of_digits(doubled)
    return total % 10


def topology_check_digit(topo_string: str) -> int:
    tmp = topo_string + "0"
    cs = topology_checksum(tmp)
    return (10 - cs) % 10


def validate_topology(topo_string: str) -> bool:
    return topology_checksum(topo_string) == 0


def generate_topologies(bead_types: List[str], count: int) -> List[str]:
    results = []
    for bt in bead_types:
        for i in range(count):
            base = f"{bt}-{i:04d}"
            cd = topology_check_digit(base)
            results.append(f"{base}{cd}")
    return results
