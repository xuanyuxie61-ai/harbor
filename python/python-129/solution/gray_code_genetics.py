"""
gray_code_genetics.py
基于 485_gray_code_display 的 Gray 码与 Hamming 距离思想，
构建凝血因子基因多态性的编码与距离度量工具。

科学背景：
    凝血因子基因（如 F5 Leiden 突变、F2 G20210A、MTHFR C677T）
    的多态性可用二进制串表示。
    Gray 码保证相邻基因型只有一个位点差异，
    这与生物进化中的逐步突变一致。

    Hamming 距离度量两个基因型之间的差异位数，
    可用于聚类分析患者遗传风险。

数学公式：
    1. Gray 码转换：
       g_i = b_i ⊕ b_{i+1}  （二进制位异或）
       或递归：G(n) = n ⊕ (n >> 1)

    2. Hamming 距离：
       H(a,b) = Σ_i |a_i - b_i| = popcount(a ⊕ b)

    3. 遗传风险评分：
       Risk = Σ_j w_j * g_j
       其中 w_j 为位点 j 的效应权重，g_j ∈ {0,1} 为基因型。
"""

import numpy as np


class GrayCodeGenetics:
    """
    Gray码基因型编码与遗传距离分析。
    """

    @staticmethod
    def int_to_gray(n, n_bits=16):
        """
        将整数 n 转换为 n_bits 位 Gray 码。
        公式：G(n) = n ^ (n >> 1)
        """
        if n < 0:
            raise ValueError("n 必须非负")
        if n_bits < 1:
            raise ValueError("n_bits 必须 >= 1")
        gray = n ^ (n >> 1)
        # 返回二进制数组
        bits = np.zeros(n_bits, dtype=int)
        for i in range(n_bits):
            bits[n_bits - 1 - i] = (gray >> i) & 1
        return bits

    @staticmethod
    def gray_to_int(bits):
        """
        将 Gray 码二进制数组转换回整数。
        递推公式：b_i = g_i ⊕ b_{i+1}
        """
        bits = np.asarray(bits, dtype=int)
        n = 0
        for b in bits:
            n = n ^ b  # 因为 Gray->Binary: b = b ^ g
            n = (n << 1)
        return n >> 1

    @staticmethod
    def hamming_distance(a, b):
        """
        计算两个二进制串的 Hamming 距离。
        """
        a = np.asarray(a, dtype=int)
        b = np.asarray(b, dtype=int)
        if a.shape != b.shape:
            raise ValueError("输入数组必须同形状")
        return np.sum(np.abs(a - b))

    @staticmethod
    def binary_distance_matrix(genotypes):
        """
        计算一组基因型之间的 Hamming 距离矩阵。

        参数:
            genotypes : ndarray, shape (n_individuals, n_snps)

        返回:
            dist : ndarray, shape (n_individuals, n_individuals)
        """
        genotypes = np.asarray(genotypes, dtype=int)
        n = genotypes.shape[0]
        dist = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(i + 1, n):
                d = GrayCodeGenetics.hamming_distance(genotypes[i], genotypes[j])
                dist[i, j] = d
                dist[j, i] = d
        return dist

    @classmethod
    def genetic_risk_score(cls, genotype, weights):
        """
        计算加权遗传风险评分。

        参数:
            genotype : ndarray, 基因型位点 (0/1)
            weights  : ndarray, 各位点效应权重

        返回:
            risk : float, 风险评分
        """
        genotype = np.asarray(genotype, dtype=float)
        weights = np.asarray(weights, dtype=float)
        if genotype.shape != weights.shape:
            raise ValueError("genotype 与 weights 必须同形状")
        return float(np.dot(genotype, weights))

    @classmethod
    def encode_coagulation_genotype(cls, has_f5_leiden=False, has_f2_g20210a=False,
                                     mthfr_677_ct=False, mthfr_677_tt=False,
                                     prothrombin_level_high=False,
                                     antithrombin_deficiency=False,
                                     protein_c_deficiency=False,
                                     protein_s_deficiency=False,
                                     n_bits=16):
        """
        将血凝相关基因多态性编码为 Gray 码。

        位点定义（示例）：
            bit 0: F5 Leiden (R506Q)
            bit 1: F2 G20210A
            bit 2: MTHFR C677T (heterozygous)
            bit 3: MTHFR C677T (homozygous)
            bit 4: Prothrombin level high
            bit 5: Antithrombin deficiency
            bit 6: Protein C deficiency
            bit 7: Protein S deficiency
        """
        bits = np.zeros(n_bits, dtype=int)
        bits[0] = int(has_f5_leiden)
        bits[1] = int(has_f2_g20210a)
        bits[2] = int(mthfr_677_ct)
        bits[3] = int(mthfr_677_tt)
        bits[4] = int(prothrombin_level_high)
        bits[5] = int(antithrombin_deficiency)
        bits[6] = int(protein_c_deficiency)
        bits[7] = int(protein_s_deficiency)
        # 其余位点保留
        return bits


def demo_genetics():
    """
    演示凝血因子基因型的 Gray 码编码与距离分析。
    """
    gc = GrayCodeGenetics()

    # 定义5个虚拟患者的基因型
    patients = [
        ("正常对照", False, False, False, False, False, False, False, False),
        ("F5 Leiden 杂合", True, False, False, False, False, False, False, False),
        ("F2 G20210A", False, True, False, False, False, False, False, False),
        ("复合突变", True, True, True, False, False, False, False, False),
        ("抗凝蛋白缺陷", False, False, False, False, False, True, True, False),
    ]

    genotypes = []
    weights = np.array([3.0, 2.5, 1.0, 2.0, 1.5, 4.0, 3.5, 3.0] + [0.0] * 8)

    print("=" * 60)
    print("凝血因子基因多态性 Gray 码编码与风险评分")
    print("=" * 60)

    for name, *flags in patients:
        bits = gc.encode_coagulation_genotype(*flags)
        risk = gc.genetic_risk_score(bits, weights)
        genotypes.append(bits)
        print(f"{name:20s}: 风险评分 = {risk:.2f}")

    genotypes = np.array(genotypes)
    dist = gc.binary_distance_matrix(genotypes)
    print("\nHamming 距离矩阵:")
    print(dist)

    return gc, genotypes, dist


if __name__ == "__main__":
    demo_genetics()
