
import numpy as np


class GrayCodeGenetics:

    @staticmethod
    def int_to_gray(n, n_bits=16):
        if n < 0:
            raise ValueError("n 必须非负")
        if n_bits < 1:
            raise ValueError("n_bits 必须 >= 1")
        gray = n ^ (n >> 1)

        bits = np.zeros(n_bits, dtype=int)
        for i in range(n_bits):
            bits[n_bits - 1 - i] = (gray >> i) & 1
        return bits

    @staticmethod
    def gray_to_int(bits):
        bits = np.asarray(bits, dtype=int)
        n = 0
        for b in bits:
            n = n ^ b
            n = (n << 1)
        return n >> 1

    @staticmethod
    def hamming_distance(a, b):
        a = np.asarray(a, dtype=int)
        b = np.asarray(b, dtype=int)
        if a.shape != b.shape:
            raise ValueError("输入数组必须同形状")
        return np.sum(np.abs(a - b))

    @staticmethod
    def binary_distance_matrix(genotypes):
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
        bits = np.zeros(n_bits, dtype=int)
        bits[0] = int(has_f5_leiden)
        bits[1] = int(has_f2_g20210a)
        bits[2] = int(mthfr_677_ct)
        bits[3] = int(mthfr_677_tt)
        bits[4] = int(prothrombin_level_high)
        bits[5] = int(antithrombin_deficiency)
        bits[6] = int(protein_c_deficiency)
        bits[7] = int(protein_s_deficiency)

        return bits


def demo_genetics():
    gc = GrayCodeGenetics()


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
