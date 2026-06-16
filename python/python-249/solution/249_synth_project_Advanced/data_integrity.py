# -*- coding: utf-8 -*-
"""
data_integrity.py
=================
Hamming-style error detection and data I/O for the stellar evolution
simulation state.

Background
----------
Long-running astrophysical simulations must guard against silent data
corruption (memory flips, disk corruption, numerical NaN propagation).
We use a Hamming (7,4) code adapted from seed 499_hamming to compute
checksums over the simulation state, and we provide a robust text
file I/O layer modelled on seed 1419_xy_display (XY data read/write
with comments and header parsing).

Key elements
------------
  - hamming74_encode / hamming74_decode : classic (7,4) error correction
  - hamming_syndrome   : detect and locate single-bit errors
  - state_checksum     : compute a Hamming-style checksum of a float array
  - write_state_file   : save simulation state to text file
  - read_state_file    : restore simulation state from text file
  - check_integrity    : verify checksum consistency
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Optional
import struct
import math
import os


# =====================================================================
# Hamming (7,4) code (from seed 499_hamming)
# =====================================================================

def hamming74_G() -> List[List[int]]:
    """Return the (7,4) Hamming code generator matrix G (7x4).

    A 4-bit message m is encoded as  c = G m  (mod 2),
    producing a 7-bit codeword c.
    """
    return [
        [1, 1, 0, 1],
        [1, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 1, 1, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]


def hamming74_H() -> List[List[int]]:
    """Return the (7,4) Hamming parity-check matrix H (3x7).

    For a received codeword r, the syndrome  s = H r (mod 2)
    identifies the error position (0 = no error, 1..7 = bit flipped).
    """
    return [
        [1, 0, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1],
    ]


def hamming74_R() -> List[List[int]]:
    """Return the (7,4) decoding matrix R (4x7) that extracts the
    message bits from a received codeword."""
    return [
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1],
    ]


def hamming74_encode(msg: List[int]) -> List[int]:
    """Encode a 4-bit message into a 7-bit codeword."""
    if len(msg) != 4:
        raise ValueError("hamming74_encode: message must have 4 bits")
    G = hamming74_G()
    code = [0] * 7
    for i in range(7):
        s = 0
        for j in range(4):
            s += G[i][j] * msg[j]
        code[i] = s % 2
    return code


def hamming74_syndrome(r: List[int]) -> int:
    """Compute the syndrome of a received 7-bit word.

    Returns an integer in 0..7:
        0 => no error
        1..7 => bit position that was flipped
    """
    if len(r) != 7:
        raise ValueError("hamming74_syndrome: word must have 7 bits")
    H = hamming74_H()
    s = 0
    for i in range(3):
        bit = 0
        for j in range(7):
            bit ^= H[i][j] & r[j]
        s |= (bit << i)
    return s


def hamming74_decode(r: List[int]) -> Tuple[List[int], int]:
    """Decode a received 7-bit word, correcting a single-bit error.

    Returns (decoded_4_bit_message, syndrome).
    """
    syn = hamming74_syndrome(r)
    if syn != 0 and 1 <= syn <= 7:
        # Correct the error
        r = r[:]
        r[syn - 1] ^= 1
    R = hamming74_R()
    msg = [0] * 4
    for i in range(4):
        s = 0
        for j in range(7):
            s ^= R[i][j] & r[j]
        msg[i] = s
    return msg, syn


# =====================================================================
# State checksum using Hamming code
# =====================================================================

def float_to_bits(v: float) -> List[int]:
    """Convert a 64-bit IEEE-754 float to a list of 64 bits."""
    packed = struct.pack("d", v)
    bits = []
    for byte in packed:
        for k in range(8):
            bits.append((byte >> k) & 1)
    return bits


def bits_to_float(bits: List[int]) -> float:
    """Inverse of float_to_bits."""
    if len(bits) != 64:
        raise ValueError("bits_to_float: need 64 bits")
    packed = bytearray(8)
    for i in range(8):
        b = 0
        for k in range(8):
            b |= (bits[i*8 + k] << k)
        packed[i] = b
    return struct.unpack("d", bytes(packed))[0]


def state_checksum(values: List[float]) -> int:
    """Compute a Hamming-style checksum over a list of floats.

    For each float we take the 64-bit representation, group into
    16 non-overlapping 4-bit nibbles, encode each nibble to a 7-bit
    Hamming codeword, and fold all codewords into a single 32-bit
    integer using XOR.

    The result can be stored alongside the data and re-computed later
    to detect silent corruption.
    """
    checksum = 0
    for v in values:
        bits = float_to_bits(v)
        for nibble_start in range(0, 64, 4):
            nibble = bits[nibble_start:nibble_start + 4]
            code = hamming74_encode(nibble)
            # Fold into 32 bits
            folded = 0
            for k in range(7):
                folded ^= (code[k] << (k % 32))
            checksum ^= folded
    return checksum & 0xFFFFFFFF


def check_data_integrity(values: List[float],
                         expected_checksum: int) -> bool:
    """Verify that the checksum of `values` matches `expected_checksum`.
    Returns True if the data passes the integrity check."""
    return state_checksum(values) == expected_checksum


# =====================================================================
# Text file I/O (modelled on seed 1419_xy_display)
# =====================================================================

def write_state_file(filename: str, state: Dict[str, object],
                     header_comment: str = "") -> None:
    """Write the simulation state to a text file.

    The file format is
        # comment lines beginning with '#'
        KEY VALUE
    for scalar values, and
        # ARRAY name length
        v0 v1 v2 ...
    for arrays.
    """
    with open(filename, "w") as f:
        f.write(f"# Stellar evolution state\n")
        if header_comment:
            f.write(f"# {header_comment}\n")
        for k, v in state.items():
            if isinstance(v, list) and all(isinstance(x, (int, float)) for x in v):
                f.write(f"# ARRAY {k} {len(v)}\n")
                f.write(" ".join(f"{x:.15e}" for x in v) + "\n")
            else:
                f.write(f"{k} {v}\n")


def read_state_file(filename: str) -> Dict[str, object]:
    """Read a simulation state from a text file written by
    write_state_file."""
    state: Dict[str, object] = {}
    if not os.path.exists(filename):
        return state
    with open(filename, "r") as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            if line.startswith("# ARRAY"):
                parts = line.split()
                if len(parts) >= 4:
                    name = parts[2]
                    try:
                        n = int(parts[3])
                    except ValueError:
                        continue
                    if i < len(lines):
                        data = lines[i].strip().split()
                        i += 1
                        try:
                            arr = [float(x) for x in data[:n]]
                        except ValueError:
                            arr = []
                        state[name] = arr
            continue
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            k, v = parts
            try:
                state[k] = int(v)
            except ValueError:
                try:
                    state[k] = float(v)
                except ValueError:
                    state[k] = v
    return state


# =====================================================================
# Diagnostic
# =====================================================================

def _self_test():
    print("data_integrity self-test:")
    # Hamming encode/decode
    for msg in [[1,0,1,0], [1,1,1,1], [0,0,0,0], [0,1,0,1]]:
        c = hamming74_encode(msg)
        msg_dec, syn = hamming74_decode(c)
        assert syn == 0
        assert msg_dec == msg
    # Error injection
    c = hamming74_encode([1,0,0,1])
    c[3] ^= 1  # flip bit 4
    msg_dec, syn = hamming74_decode(c)
    assert syn == 4
    assert msg_dec == [1,0,0,1]
    print("  Hamming(7,4) encode/decode OK")
    # State checksum
    vals = [1.0, 2.0, 3.14, -2.718281828, 0.0]
    cs = state_checksum(vals)
    assert check_data_integrity(vals, cs)
    # Corrupt one bit of one float
    bits = float_to_bits(vals[2])
    bits[5] ^= 1
    vals2 = vals[:]
    vals2[2] = bits_to_float(bits)
    assert not check_data_integrity(vals2, cs)
    print("  state checksum OK")
    print("data_integrity self-test OK")


if __name__ == "__main__":
    _self_test()
