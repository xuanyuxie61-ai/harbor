"""
phase_space_contrastive.py — 相空间结构对比诊断 (自监督学习思想)
================================================================

种子项目映射: 1016_omipan_camera_traps_self_supervised.

原项目:
  使用自监督对比学习 (SimCLR, SimSiam, Triplet) 从相机陷阱图像中
  学习动物特征的嵌入表示. 核心损失函数:
    NT-Xent: L = -log(exp(sim(z_i,z_j)/τ) / Σ exp(sim(z_i,z_k)/τ))
    Triplet: L = max(0, ||anchor-positive||² - ||anchor-negative||² + margin)

本项目映射:
  "图像" → 分布函数 f(v) 在不同速度区域的 "快照"
  "正样本对" → 相邻时间步的分布函数 (应相似)
  "负样本对" → 与 Maxwellian 差距大的分布函数 (应远离)
  "嵌入空间" → 分布函数的低维特征空间 (矩空间)

  对比损失衡量 f(v,t) 与平衡态 f_M 之间的结构差异,
  以及相邻时间步之间的演化一致性.

物理含义:
  对比诊断提供了传统 L2 范数之外的拓扑度量,
  能够捕捉分布函数的高阶结构变化 (如尾部非 Maxwellian 特征).
"""

import math
import numpy as np

from physical_constants import PI, FOUR_PI, maxwellian_1d


# ===========================================================================
#  §1  特征提取: 分布函数 → 矩向量  (多尺度, 源自 Reproduce-Algorithm)
# ===========================================================================
class MultiScaleFeatureExtractor:
    """多尺度特征提取器.

    映射自 1191_jones12138_Reproduce-Algorithm:
      原项目: 使用 CNN+LSTM, DFFN, STFTNet 等多尺度架构提取信号特征
      本项目: 使用多尺度速度空间滤波提取分布函数的层次化特征

    特征层次:
      Level 0 (raw): 分布函数值 f(v_i)
      Level 1 (local): 局部矩 (密度, 温度) 在滑动窗口内
      Level 2 (global): 全局矩 (n, T, 热流, 峰度)
      Level 3 (spectral): 速度空间的 "频率" 特征 (差分谱)

    类似 STFT (短时 Fourier 变换) 的思想:
      在不同速度尺度上分析分布函数的局部结构.
    """

    def __init__(self, v_grid, n_scales=4):
        self.v_grid = v_grid
        self.N = len(v_grid)
        self.n_scales = n_scales
        self.dv = np.diff(v_grid)

    def extract(self, f):
        """提取多尺度特征向量.

        Returns
        -------
        features : dict  各层特征
        """
        features = {}

        # Level 0: 原始分布函数值
        features["f_raw"] = f.copy()

        # Level 1: 局部统计 (滑动窗口)
        window_sizes = [max(3, self.N // (2**k)) for k in range(self.n_scales)]
        for k, ws in enumerate(window_sizes):
            local_mean = np.zeros(self.N)
            local_var = np.zeros(self.N)
            for i in range(self.N):
                start = max(0, i - ws // 2)
                end = min(self.N, i + ws // 2 + 1)
                local_mean[i] = np.mean(f[start:end])
                local_var[i] = np.var(f[start:end])
            features[f"local_mean_s{k}"] = local_mean
            features[f"local_var_s{k}"] = local_var

        # Level 2: 全局矩
        integrand_n = FOUR_PI * self.v_grid**2 * f
        integrand_T = FOUR_PI * self.v_grid**4 * f
        integrand_k = FOUR_PI * self.v_grid**6 * f
        integrand_q = FOUR_PI * self.v_grid**8 * f

        n_mom = np.trapz(integrand_n, self.v_grid)
        T_mom = np.trapz(integrand_T, self.v_grid)
        k_mom = np.trapz(integrand_k, self.v_grid)
        q_mom = np.trapz(integrand_q, self.v_grid)

        features["global_moments"] = {
            "n": n_mom,
            "mean_v2": T_mom / max(n_mom, 1e-30),
            "mean_v4": k_mom / max(n_mom, 1e-30),
            "mean_v6": q_mom / max(n_mom, 1e-30),
        }

        # Level 3: 差分谱 (类似 DFFN 的高频特征)
        df = np.diff(f)
        d2f = np.diff(df)
        features["gradient_spectrum"] = {
            "l1_norm_df": np.sum(np.abs(df)),
            "l2_norm_df": np.sqrt(np.sum(df**2)),
            "l1_norm_d2f": np.sum(np.abs(d2f)),
            "l2_norm_d2f": np.sqrt(np.sum(d2f**2)),
        }

        return features


# ===========================================================================
#  §2  对比损失函数  (源自 omipan NT-Xent / Triplet)
# ===========================================================================
def cosine_similarity(a, b):
    """余弦相似度: sim(a,b) = a·b / (||a|| · ||b||)."""
    a_flat = np.asarray(a).ravel()
    b_flat = np.asarray(b).ravel()
    dot = np.dot(a_flat, b_flat)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a < 1e-30 or norm_b < 1e-30:
        return 0.0
    return dot / (norm_a * norm_b)


def nt_xent_loss(z_anchor, z_positive, z_negatives, temperature=0.5):
    """NT-Xent 损失 (Normalized Temperature-scaled Cross Entropy).

    映射自 omipan 中的 SimCLR 损失:
      L = -log( exp(sim(a,p)/τ) / (exp(sim(a,p)/τ) + Σ_k exp(sim(a,n_k)/τ)) )

    Parameters
    ----------
    z_anchor : ndarray  锚样本特征
    z_positive : ndarray  正样本特征
    z_negatives : list of ndarray  负样本特征列表
    temperature : float  温度参数 τ
    """
    sim_ap = cosine_similarity(z_anchor, z_positive) / temperature
    exp_ap = math.exp(min(sim_ap, 50.0))  # 防止溢出

    exp_sum = exp_ap
    for z_neg in z_negatives:
        sim_an = cosine_similarity(z_anchor, z_neg) / temperature
        exp_sum += math.exp(min(sim_an, 50.0))

    loss = -math.log(max(exp_ap / max(exp_sum, 1e-30), 1e-30))
    return loss


def triplet_loss(anchor, positive, negative, margin=1.0):
    """Triplet 损失.

    映射自 omipan 中的 triplet_loss:
      L = max(0, d(a,p)² - d(a,n)² + margin)

    其中 d(·,·) 是 L2 距离.
    """
    d_ap = np.sum((np.asarray(anchor) - np.asarray(positive))**2)
    d_an = np.sum((np.asarray(anchor) - np.asarray(negative))**2)
    return max(0.0, d_ap - d_an + margin)


# ===========================================================================
#  §3  相空间结构对比诊断
# ===========================================================================
def contrastive_phase_space_diagnostic(v_grid, f_current, f_maxwellian,
                                        f_previous=None, temperature=0.5):
    """对比诊断: 衡量 f 与平衡态的结构差异.

    Parameters
    ----------
    v_grid : ndarray  速度网格
    f_current : ndarray  当前分布函数
    f_maxwellian : ndarray  参考 Maxwellian
    f_previous : ndarray or None  前一时间步的 f (可选)

    Returns
    -------
    diagnostic : dict  诊断结果
    """
    extractor = MultiScaleFeatureExtractor(v_grid)

    # 提取特征
    feat_current = extractor.extract(f_current)
    feat_maxwell = extractor.extract(f_maxwellian)

    # 全局矩对比
    gm_curr = feat_current["global_moments"]
    gm_maxw = feat_maxwell["global_moments"]

    # 余弦相似度 (全局)
    sim_global = cosine_similarity(f_current, f_maxwellian)

    # 对比损失
    # 正样本: 当前 f 的小扰动 (数值噪声)
    z_pos = f_current + 1e-6 * np.random.randn(len(f_current))
    # 负样本: Maxwellian (如果当前非 Maxwellian)
    z_neg = [f_maxwellian, 2.0 * f_maxwellian - f_current]

    if f_previous is not None:
        z_neg.append(f_previous)

    nt_xent = nt_xent_loss(f_current, z_pos, z_neg, temperature)

    # Triplet 损失
    trip = triplet_loss(f_current, z_pos, f_maxwellian, margin=1.0)

    # 各尺度差异
    scale_diffs = {}
    for key in feat_current:
        if key.startswith("local_mean") or key.startswith("local_var"):
            if key in feat_maxwell:
                diff = np.sqrt(np.mean(
                    (feat_current[key] - feat_maxwell[key])**2
                ))
                scale_diffs[key] = float(diff)

    return {
        "cosine_similarity_global": sim_global,
        "nt_xent_loss": nt_xent,
        "triplet_loss": trip,
        "density_ratio": gm_curr["n"] / max(gm_maxw["n"], 1e-30),
        "temperature_ratio": gm_curr["mean_v2"] / max(gm_maxw["mean_v2"], 1e-30),
        "scale_differences": scale_diffs,
    }
