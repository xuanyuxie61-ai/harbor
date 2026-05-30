# -*- coding: utf-8 -*-

import numpy as np
import zlib


def float_array_to_digit_sequence(arr, precision=6):
    arr = np.asarray(arr, dtype=float)

    strs = [f"{x:.{precision}e}" for x in arr.flatten()]

    digit_str = ''.join(ch for s in strs for ch in s if ch.isdigit())
    return digit_str


def luhn_checksum(digit_str):
    digits = [int(ch) for ch in digit_str if ch.isdigit()]
    if not digits:
        return 0

    total = 0
    n = len(digits)
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d2 = d * 2
            if d2 >= 10:
                d2 = d2 - 9
            total += d2
        else:
            total += d

    checksum = total % 10
    return checksum


def compute_data_fingerprint(arr, precision=6):

    arr_bytes = np.asarray(arr, dtype=float).tobytes()
    crc_val = zlib.crc32(arr_bytes) & 0xffffffff


    digit_str = float_array_to_digit_sequence(arr, precision)
    luhn_val = luhn_checksum(digit_str)

    fingerprint = f"crc:{crc_val:08x}/luhn:{luhn_val}"
    return fingerprint


def verify_data_fingerprint(arr, fingerprint, precision=6):
    computed = compute_data_fingerprint(arr, precision)
    return computed == fingerprint


def checksum_plasma_state(ne, Te, phi, precision=6):
    fp_ne = compute_data_fingerprint(ne, precision)
    fp_Te = compute_data_fingerprint(Te, precision)
    fp_phi = compute_data_fingerprint(phi, precision)
    combined = f"ne:{fp_ne}|Te:{fp_Te}|phi:{fp_phi}"
    return combined
