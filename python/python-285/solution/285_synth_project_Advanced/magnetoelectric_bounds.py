"""
磁电系数约束界模块
===================
对应种子项目: 1223_gev26_clpbounds (条件线性规划 → 磁电系数上下界)

物理背景:
    多铁性材料的磁电系数 α_ME 描述极化与磁化的交叉响应:
        ΔP = α_ME · ΔH    (电场由磁场产生)
        ΔM = α_ME · ΔE    (磁化由电场产生)

    热力学约束给出 α_ME 的理论上下界:
        |α_ME| ≤ sqrt(χ_e · χ_m · μ₀ · ε₀)

    其中 χ_e 为介电 susceptibility, χ_m 为磁 susceptibility.

    更严格的界可通过线性规划 (LP) 获得:
        最大化/最小化 α_ME
        满足: 热力学稳定性条件 (自由能正定)
              实验约束 (已知材料参数范围)

方法论 (对应种子项目 1223):
    1. 构建守恒约束矩阵 A (类似 CLP 的 build_A)
    2. 枚举对偶顶点 (vertex enumeration)
    3. 在每个顶点计算 α_ME 的界
    4. 聚合得到整体界

核心公式:
    热力学稳定性 (自由能正定):
        χ_e > 0, χ_m > 0
        χ_e·χ_m - α_ME²/(μ₀ε₀) > 0

    → α_ME² < μ₀ε₀·χ_e·χ_m (Cauchy-Schwarz 型)

    LP 形式:
        max α_ME = c^T · x
        s.t. A · x ≤ b
             x ≥ 0
"""

import numpy as np
from itertools import combinations


class MagnetoelectricBounds:
    """
    磁电系数 α_ME 的理论上下界估计.

    使用约束优化和线性规划方法,
    从热力学稳定性条件和实验数据推导 α_ME 的允许范围.
    """

    def __init__(self, chi_e=100.0, chi_m=0.01, T=300.0):
        """
        参数:
            chi_e: 介电 susceptibility (无量纲)
            chi_m: 磁 susceptibility (无量纲)
            T: 温度 (K)
        """
        from multiferroic_constants import EPSILON_0, MU_0
        self.chi_e = chi_e
        self.chi_m = chi_m
        self.T = T
        self.epsilon_0 = EPSILON_0
        self.mu_0 = MU_0

    # ============================================================
    # 理论 Cauchy-Schwarz 界
    # ============================================================

    def cauchy_schwarz_bound(self):
        """
        Cauchy-Schwarz 型理论上限.

        |α_ME| ≤ sqrt(χ_e · χ_m · μ₀ · ε₀)

        这是最宽松的热力学界.

        返回:
            alpha_max: α_ME 最大值 (s/m)
            alpha_min: α_ME 最小值 (s/m), 等于 -alpha_max
        """
        alpha_max = np.sqrt(self.chi_e * self.chi_m *
                            self.mu_0 * self.epsilon_0)
        return -alpha_max, alpha_max

    # ============================================================
    # 守恒约束矩阵 (对应种子项目 1223 build_A)
    # ============================================================

    def build_conservation_matrix(self):
        """
        构建守恒约束矩阵 A.

        对应种子项目 1223 中的 build_A():
        守恒律将隐变量 (潜相变概率) 映射到可观测矩.

        对于磁电系统, 约束来自:
            1. 自由能正定性 (3×3 矩阵正定)
            2. 极化-磁化交叉响应互易性
            3. 能量守恒

        返回:
            A: 约束矩阵, shape (n_constraints, n_variables)
            b: 右端向量, shape (n_constraints,)
            var_names: 变量名称列表
        """
        # 变量: [α_ME, χ_e, χ_m, γ_ME, δ_PE, δ_ME]
        # 其中 γ_ME 为双线性耦合, δ 为偏差变量
        n_vars = 6

        # 约束 1: 自由能正定 → χ_e·χ_m > α_ME²/(μ₀ε₀)
        # 线性化: χ_e + χ_m - 2|α_ME|/sqrt(μ₀ε₀) ≥ 0
        A1 = np.zeros(n_vars)
        A1[0] = -2.0 / np.sqrt(self.mu_0 * self.epsilon_0)
        A1[1] = 1.0
        A1[2] = 1.0
        b1 = 0.0

        # 约束 2: χ_e > 0
        A2 = np.zeros(n_vars)
        A2[1] = -1.0
        b2 = 0.0

        # 约束 3: χ_m > 0
        A3 = np.zeros(n_vars)
        A3[2] = -1.0
        b3 = 0.0

        # 约束 4: 归一化约束
        A4 = np.zeros(n_vars)
        A4[1] = 1.0 / self.chi_e
        A4[2] = 1.0 / self.chi_m
        b4 = 2.0

        # 约束 5: 耦合系数非负
        A5 = np.zeros(n_vars)
        A5[3] = -1.0
        b5 = 0.0

        # 约束 6: 能量守恒
        A6 = np.zeros(n_vars)
        A6[4] = 1.0
        A6[5] = 1.0
        b6 = 1.0

        A = np.array([A1, A2, A3, A4, A5, A6])
        b = np.array([b1, b2, b3, b4, b5, b6])

        var_names = ['alpha_ME', 'chi_e', 'chi_m',
                     'gamma_ME', 'delta_PE', 'delta_ME']

        return A, b, var_names

    # ============================================================
    # 顶点枚举 (对应种子项目 1223 enumerate_dual_vertices)
    # ============================================================

    def enumerate_dual_vertices(self, A, q=None):
        """
        枚举对偶多面体的顶点.

        对应种子项目 1223 的 enumerate_dual_vertices:
        对于约束 A·x ≤ b, 通过组合方法找到所有可行顶点.

        算法:
            1. 从 n 个约束中选 m 个 (m = 变量数)
            2. 求解等式系统 A_sub·x = b_sub
            3. 检查解是否满足所有约束
            4. 可行解即为顶点

        参数:
            A: 约束矩阵, shape (m, n)
            q: 目标函数系数 (最大化 c^T·x)

        返回:
            vertices: 顶点列表, 每个为 shape (n,) 的数组
        """
        m, n = A.shape
        if q is None:
            q = np.zeros(n)
            q[0] = 1.0  # 默认最大化 α_ME

        vertices = []

        # 枚举所有 n 个约束的组合
        for combo in combinations(range(m), n):
            A_sub = A[list(combo), :]
            b_sub = np.zeros(n)  # 简化

            try:
                if abs(np.linalg.det(A_sub)) < 1e-30:
                    continue
                x = np.linalg.solve(A_sub, b_sub)

                # 检查可行性
                if np.all(A @ x <= b_sub + 1e-10):
                    vertices.append(x)
            except np.linalg.LinAlgError:
                continue

        return vertices

    # ============================================================
    # LP 界估计
    # ============================================================

    def compute_lp_bounds(self):
        """
        计算 α_ME 的 LP 上下界.

        使用守恒约束矩阵和顶点枚举.

        返回:
            bounds: {
                'alpha_max': 上界,
                'alpha_min': 下界,
                'cs_bound': Cauchy-Schwarz 界,
                'n_vertices': 顶点数,
                'tightness': 紧致度 (LP界/CS界)
            }
        """
        A, b, var_names = self.build_conservation_matrix()

        # 上界: 最大化 α_ME
        q_max = np.zeros(len(var_names))
        q_max[0] = 1.0

        vertices = self.enumerate_dual_vertices(A, q_max)

        if len(vertices) == 0:
            # 回退到 CS 界
            cs_min, cs_max = self.cauchy_schwarz_bound()
            return {
                'alpha_max': cs_max,
                'alpha_min': cs_min,
                'cs_bound': cs_max,
                'n_vertices': 0,
                'tightness': 1.0,
            }

        # 计算顶点处的 α_ME
        alpha_values = [v[0] for v in vertices]
        alpha_max = max(alpha_values) if alpha_values else 0.0
        alpha_min = min(alpha_values) if alpha_values else 0.0

        cs_min, cs_max = self.cauchy_schwarz_bound()

        tightness = abs(alpha_max) / max(abs(cs_max), 1e-30)

        return {
            'alpha_max': alpha_max,
            'alpha_min': alpha_min,
            'cs_bound': cs_max,
            'n_vertices': len(vertices),
            'tightness': tightness,
        }

    # ============================================================
    # 温度依赖界
    # ============================================================

    def temperature_dependent_bounds(self, T_range):
        """
        计算 α_ME 界随温度的变化.

        物理:
            α_ME ∝ χ_e·χ_m·λ_ME
            在相变点附近, χ 发散 → α_ME 出现峰

        参数:
            T_range: 温度数组 (K)

        返回:
            T_range: 温度
            alpha_max_array: 上界数组
            alpha_min_array: 下界数组
        """
        from multiferroic_constants import K_BOLTZMANN

        alpha_max_arr = []
        alpha_min_arr = []

        T_C = 1103.0  # Curie 温度
        T_N = 643.0   # Néel 温度

        for T in T_range:
            # 介电 susceptibility (Curie-Weiss)
            chi_e_T = self.chi_e * T_C / max(abs(T - T_C), 1.0)
            chi_e_T = min(chi_e_T, 1e6)  # 截断

            # 磁 susceptibility
            chi_m_T = self.chi_m * T_N / max(abs(T - T_N), 1.0)
            chi_m_T = min(chi_m_T, 100.0)

            # CS 界
            alpha_max = np.sqrt(max(chi_e_T * chi_m_T *
                                    self.mu_0 * self.epsilon_0, 0.0))

            alpha_max_arr.append(alpha_max)
            alpha_min_arr.append(-alpha_max)

        return (T_range, np.array(alpha_max_arr),
                np.array(alpha_min_arr))

    # ============================================================
    # Bootstrap 置信区间 (对应种子项目 1223 multiplier_bootstrap)
    # ============================================================

    def bootstrap_confidence_interval(self, alpha_samples, n_bootstrap=100,
                                      alpha=0.05, seed=285):
        """
        乘子 Bootstrap 置信区间.

        对应种子项目 1223 的 multiplier_bootstrap_ci:
        通过对样本施加随机权重估计统计量的分布.

        步骤:
            1. 生成乘子权重 w ~ N(1, 1)
            2. 加权估计: α_boot = Σ wᵢ·αᵢ / n
            3. 重复 B 次
            4. 取 α/2 和 1-α/2 分位数

        参数:
            alpha_samples: α_ME 样本数组
            n_bootstrap: Bootstrap 次数
            alpha: 显著性水平
            seed: 随机种子

        返回:
            ci_lower: 下界
            ci_upper: 上界
            ci_mean: 均值
        """
        rng = np.random.RandomState(seed)
        n = len(alpha_samples)
        boot_means = []

        for _ in range(n_bootstrap):
            multipliers = rng.normal(1.0, 1.0, size=n)
            weighted_mean = np.mean(multipliers * alpha_samples)
            boot_means.append(weighted_mean)

        boot_means = np.array(boot_means)
        ci_lower = np.percentile(boot_means, 100 * alpha / 2)
        ci_upper = np.percentile(boot_means, 100 * (1 - alpha / 2))

        return ci_lower, ci_upper, np.mean(boot_means)
