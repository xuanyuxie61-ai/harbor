"""
simulation_checksum.py
======================
Data Integrity Verification for DNS Simulation States Using Hamming Codes.

Based on seed project 499 (hamming74_g, hamming74_h):
- Hamming (7,4) code generation and parity-check matrices
- Error detection for simulation checkpoint data

Scientific Context:
-------------------
Long-running DNS simulations generate massive datasets. Data corruption in
checkpoint files can invalidate months of computation. Hamming codes provide
single-error detection and correction for critical simulation state vectors.

Hamming (7,4) Code:
-------------------
For a 4-bit message m = [m1, m2, m3, m4], the 7-bit codeword is:
  c = G · m

where G is the 7×4 generator matrix:
      [1 1 0 1]
      [1 0 1 1]
  G = [1 0 0 0]
      [0 1 1 1]
      [0 1 0 0]
      [0 0 1 0]
      [0 0 0 1]

Parity-check matrix H (3×7):
      [1 0 1 0 1 0 1]
  H = [0 1 1 0 0 1 1]
      [0 0 0 1 1 1 1]

Syndrome computation: s = H · c (mod 2)
  - s = 0: valid codeword
  - s ≠ 0: error position given by binary value of syndrome

For simulation data, we pack floating-point state variables into 4-bit nibbles,
encode with Hamming (7,4), and store parity bits alongside data.
"""

import numpy as np


def hamming74_generator_matrix():
    """
    Return Hamming (7,4) generator matrix G.
    Based on seed 499 (hamming74_g.m).
    """
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
    """
    Return Hamming (7,4) parity-check matrix H.
    Based on seed 499 (hamming74_h.m).
    """
    return np.array([
        [1, 0, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1],
    ], dtype=int)


def float_to_nibbles(val, n_nibbles=8):
    """
    Convert a float to n_nibbles 4-bit unsigned integers via fixed-point encoding.
    Range: [-1e6, 1e6] mapped to [0, 16^n_nibbles - 1].
    """
    max_val = 1e6
    if not np.isfinite(val):
        val = 0.0
    val = np.clip(float(val), -max_val, max_val)
    # Map to unsigned integer
    scale = (2**(4 * n_nibbles) - 1) / (2 * max_val)
    int_val = int((val + max_val) * scale)
    int_val = max(0, min(int_val, 2**(4 * n_nibbles) - 1))
    nibbles = []
    for _ in range(n_nibbles):
        nibbles.append(int_val & 0xF)
        int_val >>= 4
    return nibbles[::-1]  # MSB first


def nibbles_to_float(nibbles, n_nibbles=8):
    """Inverse of float_to_nibbles."""
    max_val = 1e6
    int_val = 0
    for n in nibbles:
        int_val = (int_val << 4) | n
    scale = (2**(4 * n_nibbles) - 1) / (2 * max_val)
    return int_val / scale - max_val


def encode_nibble_hamming74(nibble):
    """
    Encode a 4-bit nibble into a 7-bit Hamming codeword.
    Returns list of 7 bits.
    """
    m = np.array([(nibble >> 3) & 1,
                  (nibble >> 2) & 1,
                  (nibble >> 1) & 1,
                  nibble & 1], dtype=int)
    G = hamming74_generator_matrix()
    c = np.dot(G, m) % 2
    return c.tolist()


def decode_hamming74(codeword):
    """
    Decode Hamming (7,4) codeword, correct single-bit errors.
    Returns (nibble, error_detected, error_position).
    """
    c = np.array(codeword, dtype=int) % 2
    H = hamming74_parity_check_matrix()
    s = np.dot(H, c) % 2
    syndrome = s[0] + 2 * s[1] + 4 * s[2]

    error_detected = (syndrome != 0)
    error_position = syndrome - 1 if error_detected else -1

    if error_detected and 0 <= error_position < 7:
        c[error_position] = 1 - c[error_position]

    # Extract message bits from systematic positions
    m = np.array([c[2], c[4], c[5], c[6]], dtype=int)
    nibble = (m[0] << 3) | (m[1] << 2) | (m[2] << 1) | m[3]
    return nibble, error_detected, error_position


def encode_simulation_state(state_dict):
    """
    Encode a dictionary of simulation state variables with Hamming parity.
    Returns encoded dict with parity fields.
    """
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
            # Store checksum for arrays
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
    """
    Verify encoded simulation state and report any errors.
    Returns (valid, errors).
    """
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
            # Just verify checksum is present (cannot recompute without original data)
            if not np.isfinite(val['checksum']):
                errors.append(f"{key}: invalid checksum")
    return len(errors) == 0, errors


def compute_fletcher_checksum(data):
    """
    Fletcher-16 checksum for binary data integrity.
    More robust than simple sum for large arrays.
    """
    data = np.asarray(data).flatten()
    # Quantize to 16-bit integers
    data_int = np.mod(np.round(data * 1e6).astype(np.int64), 65535)
    sum1 = 0
    sum2 = 0
    for val in data_int:
        sum1 = (sum1 + val) % 255
        sum2 = (sum2 + sum1) % 255
    return (sum2 << 8) | sum1
