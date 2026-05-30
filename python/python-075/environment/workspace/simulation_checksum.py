
import numpy as np


def hamming74_generator_matrix():
    return np.array([
        [1, 1, 0, 1],
        [1, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 1, 1, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=int)


def hamming74_parity_check_matrix():
    return np.array([
        [1, 0, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1],
    ], dtype=int)


def float_to_nibbles(val, n_nibbles=8):
    max_val = 1e6
    if not np.isfinite(val):
        val = 0.0
    val = np.clip(float(val), -max_val, max_val)

    scale = (2**(4 * n_nibbles) - 1) / (2 * max_val)
    int_val = int((val + max_val) * scale)
    int_val = max(0, min(int_val, 2**(4 * n_nibbles) - 1))
    nibbles = []
    for _ in range(n_nibbles):
        nibbles.append(int_val & 0xF)
        int_val >>= 4
    return nibbles[::-1]


def nibbles_to_float(nibbles, n_nibbles=8):
    max_val = 1e6
    int_val = 0
    for n in nibbles:
        int_val = (int_val << 4) | n
    scale = (2**(4 * n_nibbles) - 1) / (2 * max_val)
    return int_val / scale - max_val


def encode_nibble_hamming74(nibble):
    m = np.array([(nibble >> 3) & 1,
                  (nibble >> 2) & 1,
                  (nibble >> 1) & 1,
                  nibble & 1], dtype=int)
    G = hamming74_generator_matrix()
    c = np.dot(G, m) % 2
    return c.tolist()


def decode_hamming74(codeword):
    c = np.array(codeword, dtype=int) % 2
    H = hamming74_parity_check_matrix()
    s = np.dot(H, c) % 2
    syndrome = s[0] + 2 * s[1] + 4 * s[2]

    error_detected = (syndrome != 0)
    error_position = syndrome - 1 if error_detected else -1

    if error_detected and 0 <= error_position < 7:
        c[error_position] = 1 - c[error_position]


    m = np.array([c[2], c[4], c[5], c[6]], dtype=int)
    nibble = (m[0] << 3) | (m[1] << 2) | (m[2] << 1) | m[3]
    return nibble, error_detected, error_position


def encode_simulation_state(state_dict):
    encoded = {}
    for key, val in state_dict.items():
        if isinstance(val, (int, float, np.floating)):
            nibbles = float_to_nibbles(float(val))
            codewords = [encode_nibble_hamming74(n) for n in nibbles]
            encoded[key] = {
                'value': float(val),
                'parity': codewords,
                'nibbles': nibbles,
            }
        elif isinstance(val, np.ndarray):

            checksum = np.sum(val) % 1000000.0
            encoded[key] = {
                'shape': val.shape,
                'checksum': float(checksum),
                'mean': float(np.mean(val)),
                'std': float(np.std(val)),
            }
        else:
            encoded[key] = val
    return encoded


def verify_simulation_state(encoded_state):
    errors = []
    for key, val in encoded_state.items():
        if isinstance(val, dict) and 'parity' in val:
            nibbles = val['nibbles']
            parity = val['parity']
            for i, (n, cw) in enumerate(zip(nibbles, parity)):
                decoded_n, err_detected, err_pos = decode_hamming74(cw)
                if decoded_n != n:
                    errors.append(f"{key}[{i}]: parity mismatch (decoded={decoded_n}, expected={n})")
                elif err_detected:
                    errors.append(f"{key}[{i}]: single error corrected at position {err_pos}")
        elif isinstance(val, dict) and 'checksum' in val:

            if not np.isfinite(val['checksum']):
                errors.append(f"{key}: invalid checksum")
    return len(errors) == 0, errors


def compute_fletcher_checksum(data):
    data = np.asarray(data).flatten()

    data_int = np.mod(np.round(data * 1e6).astype(np.int64), 65535)
    sum1 = 0
    sum2 = 0
    for val in data_int:
        sum1 = (sum1 + val) % 255
        sum2 = (sum2 + sum1) % 255
    return (sum2 << 8) | sum1
