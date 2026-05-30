
import numpy as np


def float_to_digits(value, scale=1e9, max_digits=15):
    if not np.isfinite(value):
        return [0]

    scaled = int(np.round(np.abs(value) * scale))
    if scaled == 0:
        return [0]
    digits = []
    while scaled > 0 and len(digits) < max_digits:
        digits.append(scaled % 10)
        scaled //= 10
    return digits[::-1]


def luhn_checksum_digits(digits):
    digits = np.asarray(digits, dtype=int)
    n = len(digits)
    if n == 0:
        return 0


    total = np.sum(digits[n % 2::2])
    for i in range(n % 2 == 0, n, 2):
        d2 = 2 * digits[i]
        total += d2 // 10 + d2 % 10

    return int(total % 10)


def compute_array_luhn(array, scale=1e9):
    array = np.asarray(array, dtype=float)
    flat = array.ravel()

    all_digits = []
    for val in flat:
        all_digits.extend(float_to_digits(val, scale))

    if len(all_digits) == 0:
        return 0, '0'

    checksum = luhn_checksum_digits(all_digits)

    check_digit = (10 - checksum) % 10
    return checksum, str(check_digit)


def verify_array_integrity(array, expected_check_digit, scale=1e9):
    checksum, check_digit = compute_array_luhn(array, scale)
    return str(check_digit) == str(expected_check_digit)


def mesh_topology_checksum(triangles, nodes):
    triangles = np.asarray(triangles, dtype=int)
    nodes = np.asarray(nodes, dtype=float)


    all_digits = []
    for tri in triangles.ravel():
        all_digits.extend(float_to_digits(float(tri), scale=1.0))
    for coord in nodes.ravel():
        all_digits.extend(float_to_digits(coord, scale=1e6))

    if len(all_digits) == 0:
        return 0, '0'

    checksum = luhn_checksum_digits(all_digits)
    check_digit = (10 - checksum) % 10
    return checksum, str(check_digit)


class NumericalIntegrityChecker:

    def __init__(self, scale=1e9):
        self.scale = float(scale)
        self._checkpoints = {}

    def checkpoint(self, name, array):
        checksum, check_digit = compute_array_luhn(array, self.scale)
        self._checkpoints[name] = {
            'shape': array.shape,
            'dtype': str(array.dtype),
            'checksum': checksum,
            'check_digit': check_digit,
            'mean': float(np.mean(array)) if array.size > 0 else 0.0,
            'std': float(np.std(array)) if array.size > 0 else 0.0
        }

    def verify(self, name, array):
        if name not in self._checkpoints:
            return False, {'error': 'Checkpoint not found'}

        cp = self._checkpoints[name]
        details = {
            'shape_match': array.shape == cp['shape'],
            'checksum_match': False,
            'statistical_match': False
        }

        _, check_digit = compute_array_luhn(array, self.scale)
        details['checksum_match'] = (check_digit == cp['check_digit'])

        if array.size > 0:
            mean_diff = abs(float(np.mean(array)) - cp['mean'])
            std_diff = abs(float(np.std(array)) - cp['std'])
            details['statistical_match'] = (mean_diff < 1e-6 * max(abs(cp['mean']), 1.0) and
                                            std_diff < 1e-6 * max(cp['std'], 1.0))

        passed = details['shape_match'] and details['checksum_match']
        return passed, details

    def summary(self):
        return self._checkpoints.copy()
