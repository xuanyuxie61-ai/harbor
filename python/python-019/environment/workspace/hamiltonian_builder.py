
import numpy as np



PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)
IDENTITY2 = np.eye(2, dtype=complex)


def build_pt_symmetric_hamiltonian_1d(k, t=1.0, m=0.5, gamma=0.3):
    H = (m + t * np.cos(k)) * PAULI_Z + t * np.sin(k) * PAULI_Y + 1j * gamma * PAULI_X
    return H


def build_pt_symmetric_hamiltonian_2d(kx, ky, t=1.0, m=0.5, gamma=0.3, a=1.0):
    H = -2.0 * t * (np.cos(kx * a) * PAULI_X + np.cos(ky * a) * PAULI_Y)
    H += (m + 1j * gamma) * PAULI_Z
    return H


def build_nonhermitian_ssh_hamiltonian(k, t1=1.0, t2=0.5, gamma=0.2):


    raise NotImplementedError("SSH Hamiltonian construction is missing.")


def build_nonhermitian_hofstadter_hamiltonian(kx, ky, phi, t=1.0, gamma=0.1, q=4):
    if q <= 0:
        raise ValueError("q must be a positive integer.")
    H = np.zeros((q, q), dtype=complex)
    for n in range(q):

        H[n, n] = 1j * gamma * ((-1) ** n)

        H[n, (n + 1) % q] += -t * np.exp(1j * kx)
        H[(n + 1) % q, n] += -t * np.exp(-1j * kx)

        H[n, n] += -2.0 * t * np.cos(ky - 2.0 * np.pi * phi * n)
    return H


def characteristic_polynomial_2x2(H):
    if H.shape != (2, 2):
        raise ValueError("Only 2x2 matrices supported.")
    c2 = 1.0 + 0.0j
    c1 = -np.trace(H)
    c0 = np.linalg.det(H)
    return c2, c1, c0


def discriminant_2x2(H):
    c2, c1, c0 = characteristic_polynomial_2x2(H)
    delta = c1 ** 2 - 4.0 * c2 * c0
    return delta
