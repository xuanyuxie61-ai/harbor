"""
磁电耦合连接器模块
===================
对应种子项目: 1136_SonyResearch_SVG (同步视频-音频生成 → 极化-磁化同步演化)

物理背景:
    多铁性材料中, 极化 P 和磁化 M 通过磁电耦合相互影响.
    其时间演化方程为:
        ∂P/∂t = -L_P · δF/δP + ξ_P     (LGD 方程)
        ∂M/∂t = -γ·(M × H_eff) - αγ·(M × (M × H_eff)) + ξ_M  (LLG 方程)

    这两个方程通过有效场中的磁电耦合项关联.
    本模块实现类似 SVG 模型中 "connector" 的跨场耦合机制:
        - P_connector: 从 M 场到 P 场的信息传递
        - M_connector: 从 P 场到 M 场的信息传递

SVG 连接器类比:
    在 SVG 中, audio UNet 和 video UNet 通过 cross-attention 连接器同步.
    类似地, P 和 M 场通过磁电有效场项同步:
        H_ME(P→M) = -∂f_ME/∂M  (影响 M 演化)
        H_ME(M→P) = -∂f_ME/∂P  (影响 P 演化)

    这类似于 cross-attention:
        Q = M 的投影, K = P 的投影, V = P 的信息
        attention(Q,K,V) = softmax(QKᵀ/√d)·V → 加到 M 的更新中

核心公式:
    磁电连接器 (线性):
        connector_P(M) = α_ME · M² / |M|²    (标量调制)
        connector_M(P) = α_ME · P              (矢量调制)

    磁电连接器 (双线性):
        connector_P(M) = 2γ·P·M²
        connector_M(P) = 2γ·P²·M

    同步强度:
        η = ||connector(P,M)|| / (||P||·||M||)
"""

import numpy as np


class MagnetoelectricConnector:
    """
    极化-磁化跨场耦合连接器.

    类比种子项目 1136 (SVG) 中的 cross-attention connector:
        - P_connector_in: 接收 M 场信息
        - P_connector_out: 输出修正后的 P 更新
        - M_connector_in: 接收 P 场信息
        - M_connector_out: 输出修正后的 M 更新
    """

    def __init__(self, alpha_ME=1.2e-10, gamma_ME=5.0e-4,
                 coupling_mode='bilinear'):
        """
        参数:
            alpha_ME: 线性磁电耦合系数 (s/m)
            gamma_ME: 双线性耦合系数 (J·m⁴/(C²·A))
            coupling_mode: 'linear', 'bilinear', 'cross_attention'
        """
        self.alpha_ME = alpha_ME
        self.gamma_ME = gamma_ME
        self.coupling_mode = coupling_mode

        # 连接器权重 (类似 cross-attention 的 QKV 投影)
        self._init_connector_weights()

    def _init_connector_weights(self):
        """
        初始化连接器权重矩阵 (3×3).

        在 cross-attention 模式下, 每个场被投影到 Q, K, V 空间.
        """
        # 小的随机初始化
        rng = np.random.RandomState(285)
        self.W_P_to_M = rng.randn(3, 3) * 0.01
        self.W_M_to_P = rng.randn(3, 3) * 0.01

        # 对称化 (物理约束: 互易性)
        self.W_P_to_M = 0.5 * (self.W_P_to_M + self.W_P_to_M.T)
        self.W_M_to_P = 0.5 * (self.W_M_to_P + self.W_M_to_P.T)

    # ============================================================
    # P → M 连接器 (极化影响磁化)
    # ============================================================

    def connector_P_to_M(self, P, M):
        """
        计算从 P 场到 M 场的耦合修正.

        对应 SVG 中 audio_connector_in:
            将 P 的信息注入到 M 的更新方程.

        在物理上, 这对应磁电有效场:
            ΔH_M = -∂f_ME/∂P · (投影到 M 空间)

        参数:
            P: 极化场, shape (nx, ny, 3)
            M: 磁化场, shape (nx, ny, 3)

        返回:
            delta_M: M 的耦合修正, shape (nx, ny, 3)
        """
        if self.coupling_mode == 'linear':
            # 线性: ΔH_M ~ α·P
            delta_M = self.alpha_ME * P

        elif self.coupling_mode == 'bilinear':
            # 双线性: ΔH_M ~ 2γ·P²·M
            P_sq = np.sum(P ** 2, axis=2, keepdims=True)
            delta_M = 2.0 * self.gamma_ME * P_sq * M

        elif self.coupling_mode == 'cross_attention':
            # Cross-attention 式:
            # Q = M·W_Q, K = P·W_K, V = P·W_V
            # attention = softmax(QK^T/√3)·V
            delta_M = self._cross_attention_step(M, P, self.W_M_to_P)

        else:
            delta_M = np.zeros_like(M)

        return delta_M

    # ============================================================
    # M → P 连接器 (磁化影响极化)
    # ============================================================

    def connector_M_to_P(self, P, M):
        """
        计算从 M 场到 P 场的耦合修正.

        对应 SVG 中 video_connector_in:
            将 M 的信息注入到 P 的更新方程.

        物理上:
            ΔH_P = -∂f_ME/∂M

        参数:
            P: 极化场, shape (nx, ny, 3)
            M: 磁化场, shape (nx, ny, 3)

        返回:
            delta_P: P 的耦合修正, shape (nx, ny, 3)
        """
        if self.coupling_mode == 'linear':
            # 线性: ΔH_P ~ α·M²
            M_sq = np.sum(M ** 2, axis=2, keepdims=True)
            delta_P = self.alpha_ME * M_sq * np.ones_like(P)

        elif self.coupling_mode == 'bilinear':
            # 双线性: ΔH_P ~ 2γ·M²·P
            M_sq = np.sum(M ** 2, axis=2, keepdims=True)
            delta_P = 2.0 * self.gamma_ME * M_sq * P

        elif self.coupling_mode == 'cross_attention':
            delta_P = self._cross_attention_step(P, M, self.W_P_to_M)
        else:
            delta_P = np.zeros_like(P)

        return delta_P

    # ============================================================
    # Cross-attention 步骤
    # ============================================================

    def _cross_attention_step(self, query_field, key_field, W_proj):
        """
        Cross-attention 风格的场耦合.

        步骤:
            1. 投影: Q = query·W, K = key·W
            2. 注意力权重: a = softmax(Q·K^T / √d_k)
            3. 输出: out = a · key

        对于空间网格:
            Q[i,j,:] = query[i,j,:] · W_proj
            attention[i,j] = exp(Q·K) / Σ exp(Q·K)

        参数:
            query_field: shape (nx, ny, 3)
            key_field: shape (nx, ny, 3)
            W_proj: 投影矩阵, shape (3, 3)

        返回:
            output: shape (nx, ny, 3)
        """
        nx, ny, d = query_field.shape
        d_k = float(d)

        # 投影
        Q = np.tensordot(query_field, W_proj, axes=([2], [0]))
        K = np.tensordot(key_field, W_proj, axes=([2], [0]))

        # 逐点注意力 (空间局部)
        # QK^T / sqrt(d_k) → 标量注意力权重
        attention_logits = np.sum(Q * K, axis=2, keepdims=True) / np.sqrt(d_k)

        # Softmax 归一化
        attention_max = np.max(attention_logits, axis=(0, 1), keepdims=True)
        attention_exp = np.exp(attention_logits - attention_max)
        attention_weights = attention_exp / (
            np.sum(attention_exp, axis=(0, 1), keepdims=True) + 1e-30
        )

        # 加权输出
        output = attention_weights * key_field

        return output

    # ============================================================
    # 同步强度度量
    # ============================================================

    def synchronization_strength(self, P, M):
        """
        计算 P 和 M 场之间的同步强度.

        η = <P·M> / (|<P>|·|<M>|)

        物理含义:
            η ≈ 1: 强耦合 (极化与磁化高度对齐)
            η ≈ 0: 弱耦合 (正交)
            η ≈ -1: 反平行耦合

        参数:
            P: 极化场, shape (nx, ny, 3)
            M: 磁化场, shape (nx, ny, 3)

        返回:
            eta: 同步强度 (标量, -1 到 1)
        """
        P_avg = np.mean(P, axis=(0, 1))
        M_avg = np.mean(M, axis=(0, 1))

        norm_P = np.linalg.norm(P_avg)
        norm_M = np.linalg.norm(M_avg)

        if norm_P < 1e-30 or norm_M < 1e-30:
            return 0.0

        eta = np.dot(P_avg, M_avg) / (norm_P * norm_M)
        return float(eta)

    def coupling_energy_density(self, P, M):
        """
        计算磁电耦合能密度分布.

        f_ME(r) = α·P·M² + γ·(P·M)²

        参数:
            P: 极化场, shape (nx, ny, 3)
            M: 磁化场, shape (nx, ny, 3)

        返回:
            f_ME: 能量密度, shape (nx, ny)
        """
        P_dot_M = np.sum(P * M, axis=2)
        M_sq = np.sum(M ** 2, axis=2)

        f_linear = self.alpha_ME * np.sum(P, axis=2) * M_sq
        f_bilinear = self.gamma_ME * P_dot_M ** 2

        return f_linear + f_bilinear

    # ============================================================
    # 连接器诊断
    # ============================================================

    def connector_diagnostics(self, P, M):
        """
        输出连接器诊断信息.

        类似 SVG 训练中的 connector 监控.
        """
        sync = self.synchronization_strength(P, M)
        f_ME = self.coupling_energy_density(P, M)

        diag = {
            'sync_strength': sync,
            'mean_coupling_energy': np.mean(f_ME),
            'max_coupling_energy': np.max(f_ME),
            'P_norm': np.mean(np.linalg.norm(P, axis=2)),
            'M_norm': np.mean(np.linalg.norm(M, axis=2)),
        }

        return diag
