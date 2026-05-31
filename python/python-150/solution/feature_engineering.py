"""
feature_engineering.py
======================
分子特征工程与编码

融合种子项目:
  - 586_image_threshold : 阈值分割与二值化

科学背景:
  将连续的物理化学属性（电负性、电离能、原子半径等）通过阈值映射为离散编码，
  形成类似分子指纹（Morgan fingerprint）的二值特征，增强模型的非线性表达能力。

  同时实现 Coulomb 矩阵特征、RDF（径向分布函数）直方图等经典分子描述符。
"""

import numpy as np
from typing import List


# ------------------------------------------------------------------
# 1. 阈值编码 (源自 image_threshold)
# ------------------------------------------------------------------

def threshold_binarize(features: np.ndarray, threshold: float) -> np.ndarray:
    """
    单阈值二值化: value ≤ threshold → 0, value > threshold → 1。
    类似图像阈值分割，将连续原子特征映射为二值码。
    """
    return (features > threshold).astype(np.float64)


def double_threshold_encode(features: np.ndarray, low: float, high: float) -> np.ndarray:
    """
    双阈值编码: value < low → 0, value > high → 2, 否则 → 1。
    形成三态离散特征。
    """
    encoded = np.ones_like(features)
    encoded[features < low] = 0.0
    encoded[features > high] = 2.0
    return encoded


def molecular_fingerprint(atom_features: np.ndarray, thresholds: List[float]) -> np.ndarray:
    """
    多层阈值分子指纹：对每个阈值进行二值化并拼接，形成扩展指纹。
    """
    parts = []
    for t in thresholds:
        parts.append(threshold_binarize(atom_features, t))
    return np.concatenate(parts, axis=0)


# ------------------------------------------------------------------
# 2. Coulomb 矩阵描述符
# ------------------------------------------------------------------

def coulomb_matrix(atoms: np.ndarray, charges: np.ndarray,
                   max_size: int = 12, alpha: float = 1.0) -> np.ndarray:
    """
    构建 Coulomb 矩阵 (Rupp et al., PRL 2012):
        C_{ij} = 0.5 Z_i^{2.4}           (i = j)
        C_{ij} = Z_i Z_j / |R_i - R_j|   (i ≠ j)
    用于编码分子核-核库仑相互作用。

    为保持固定维度，对原子按范数排序并截断/填充至 max_size。
    """
    n = atoms.shape[0]
    C = np.zeros((max_size, max_size), dtype=np.float64)
    if n == 0:
        return C.flatten()

    # 计算完整矩阵
    C_full = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        C_full[i, i] = 0.5 * charges[i] ** 2.4
        for j in range(i + 1, n):
            r = np.linalg.norm(atoms[i] - atoms[j])
            r = max(r, 0.5)
            val = charges[i] * charges[j] / r
            C_full[i, j] = val
            C_full[j, i] = val

    # 按行范数降序排列，保证置换不变性
    norms = np.linalg.norm(C_full, axis=1)
    order = np.argsort(-norms)
    C_sorted = C_full[order][:, order]

    sz = min(n, max_size)
    C[:sz, :sz] = C_sorted[:sz, :sz]
    return C.flatten()


# ------------------------------------------------------------------
# 3. 径向分布函数 (RDF) 直方图
# ------------------------------------------------------------------

def radial_distribution_histogram(atoms: np.ndarray, dr: float = 0.1,
                                  r_max: float = 5.0) -> np.ndarray:
    """
    计算分子内原子对的径向分布函数 g(r) 直方图。
    g(r) = (V / N²) * (n(r) / (4π r² dr))
    这里简化为归一化的距离直方图。
    """
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
                # 4π r² 权重归一化
                shell_vol = 4.0 * np.pi * (r ** 2) * dr
                if shell_vol > 1e-12:
                    hist[idx] += 1.0 / shell_vol

    # 归一化
    total = hist.sum()
    if total > 1e-12:
        hist = hist / total
    return hist


# ------------------------------------------------------------------
# 4. 综合特征向量
# ------------------------------------------------------------------

def compute_atom_features(atomic_numbers: np.ndarray) -> np.ndarray:
    """
    从原子序数构造基础物理化学特征:
      [Z, sqrt(Z), EN (Pauling), vdw_radius, ionization_energy]
    使用经验公式近似。
    """
    Z = atomic_numbers.astype(np.float64)
    # 电负性近似 (Pauling 经验拟合)
    EN = 0.7 + 0.18 * Z - 0.0005 * Z ** 2
    EN = np.clip(EN, 0.7, 4.0)
    # 范德华半径近似 (Å)
    vdw = 1.2 + 0.01 * Z
    # 电离能近似 (eV)
    IE = 5.0 + 0.3 * Z - 0.001 * Z ** 2
    IE = np.clip(IE, 3.0, 25.0)

    feats = np.column_stack([Z, np.sqrt(Z), EN, vdw, IE])
    return feats


def encode_molecular_features(graph, atomic_numbers: np.ndarray,
                              thresholds: List[float] = None) -> np.ndarray:
    """
    将分子图编码为综合特征向量，包含:
      - Coulomb 矩阵特征
      - RDF 直方图
      - 多层阈值指纹
      - 多项式描述符 (从 polynomial_basis 调用)
    """
    if thresholds is None:
        thresholds = [0.5, 1.0, 1.5, 2.0]

    from polynomial_basis import compute_polynomial_descriptors

    atoms = graph.atoms
    charges = atomic_numbers.astype(np.float64)

    # Coulomb 矩阵
    cm = coulomb_matrix(atoms, charges, max_size=12)

    # RDF
    rdf = radial_distribution_histogram(atoms, dr=0.2, r_max=4.0)

    # 多层阈值指纹 (基于原子特征均值)
    atom_feats = compute_atom_features(atomic_numbers)
    mean_feats = atom_feats.mean(axis=0)
    fp = molecular_fingerprint(mean_feats, thresholds)

    # 多项式描述符
    poly = compute_polynomial_descriptors(atoms, degree=3)

    # 拼接
    combined = np.concatenate([cm, rdf, fp, poly])
    # 归一化
    norm = np.linalg.norm(combined)
    if norm > 1e-12:
        combined = combined / norm
    return combined
