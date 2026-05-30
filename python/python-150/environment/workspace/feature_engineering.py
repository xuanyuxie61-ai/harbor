
import numpy as np
from typing import List






def threshold_binarize(features: np.ndarray, threshold: float) -> np.ndarray:
    return (features > threshold).astype(np.float64)


def double_threshold_encode(features: np.ndarray, low: float, high: float) -> np.ndarray:
    encoded = np.ones_like(features)
    encoded[features < low] = 0.0
    encoded[features > high] = 2.0
    return encoded


def molecular_fingerprint(atom_features: np.ndarray, thresholds: List[float]) -> np.ndarray:
    parts = []
    for t in thresholds:
        parts.append(threshold_binarize(atom_features, t))
    return np.concatenate(parts, axis=0)






def coulomb_matrix(atoms: np.ndarray, charges: np.ndarray,
                   max_size: int = 12, alpha: float = 1.0) -> np.ndarray:
    n = atoms.shape[0]
    C = np.zeros((max_size, max_size), dtype=np.float64)
    if n == 0:
        return C.flatten()


    C_full = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        C_full[i, i] = 0.5 * charges[i] ** 2.4
        for j in range(i + 1, n):
            r = np.linalg.norm(atoms[i] - atoms[j])
            r = max(r, 0.5)
            val = charges[i] * charges[j] / r
            C_full[i, j] = val
            C_full[j, i] = val


    norms = np.linalg.norm(C_full, axis=1)
    order = np.argsort(-norms)
    C_sorted = C_full[order][:, order]

    sz = min(n, max_size)
    C[:sz, :sz] = C_sorted[:sz, :sz]
    return C.flatten()






def radial_distribution_histogram(atoms: np.ndarray, dr: float = 0.1,
                                  r_max: float = 5.0) -> np.ndarray:
    n = atoms.shape[0]
    n_bins = int(r_max / dr)
    hist = np.zeros(n_bins, dtype=np.float64)
    if n < 2:
        return hist

    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(atoms[i] - atoms[j])
            idx = int(r / dr)
            if 0 <= idx < n_bins:

                shell_vol = 4.0 * np.pi * (r ** 2) * dr
                if shell_vol > 1e-12:
                    hist[idx] += 1.0 / shell_vol


    total = hist.sum()
    if total > 1e-12:
        hist = hist / total
    return hist






def compute_atom_features(atomic_numbers: np.ndarray) -> np.ndarray:
    Z = atomic_numbers.astype(np.float64)

    EN = 0.7 + 0.18 * Z - 0.0005 * Z ** 2
    EN = np.clip(EN, 0.7, 4.0)

    vdw = 1.2 + 0.01 * Z

    IE = 5.0 + 0.3 * Z - 0.001 * Z ** 2
    IE = np.clip(IE, 3.0, 25.0)

    feats = np.column_stack([Z, np.sqrt(Z), EN, vdw, IE])
    return feats


def encode_molecular_features(graph, atomic_numbers: np.ndarray,
                              thresholds: List[float] = None) -> np.ndarray:
    if thresholds is None:
        thresholds = [0.5, 1.0, 1.5, 2.0]

    from polynomial_basis import compute_polynomial_descriptors

    atoms = graph.atoms
    charges = atomic_numbers.astype(np.float64)


    cm = coulomb_matrix(atoms, charges, max_size=12)


    rdf = radial_distribution_histogram(atoms, dr=0.2, r_max=4.0)


    atom_feats = compute_atom_features(atomic_numbers)
    mean_feats = atom_feats.mean(axis=0)
    fp = molecular_fingerprint(mean_feats, thresholds)


    poly = compute_polynomial_descriptors(atoms, degree=3)


    combined = np.concatenate([cm, rdf, fp, poly])

    norm = np.linalg.norm(combined)
    if norm > 1e-12:
        combined = combined / norm
    return combined
