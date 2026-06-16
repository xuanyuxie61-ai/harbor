"""
partial_wave.py — 分波展开与散射振幅计算 (PROJECT_232)

融合种子项目:
  - 371_fem_basis: 有限元基函数构造 (fem_basis_1d, fem_basis_2d)
  - 1372_unicycle: 置换群循环结构 (用于全同粒子对称性)
  - 415_fem2d_scalar_display_brief: 二维标量场数据结构 (场振幅存储)

核心物理:
  对于中心势散射，分波展开:
    f(θ) = (1/k) Σ_{l=0}^{L_max} (2l+1) T_l(s) P_l(cos θ)

  其中 T_l(s) = (S_l - 1)/(2i) = e^{iδ_l} sin δ_l

  弹性散射幺正性:
    Im T_l = |T_l|² / ρ(s)  ⟹  |S_l| = 1  (弹性区)

  总截面 (光学定理):
    σ_tot = (4π/k²) Σ_{l} (2l+1) sin² δ_l
          = (4π/k) Im f(0)

  微分截面:
    dσ/dΩ = |f(θ)|²

  本模块将有限元基函数思想推广为分波展开的角向基函数，
  并使用二维标量场结构存储散射振幅 f(√s, θ)。
"""
import numpy as np
from constants import PI, TWOPI, FOURPI, EPS_MACH, MASS_PI_PLUS
from special_functions import legendre_p, permutation_cycle_type


# ---------------------------------------------------------------------------
# 分波振幅计算
# ---------------------------------------------------------------------------

def phase_shift_to_t_matrix(delta_l):
    """
    将相移 δ_l 转换为 T 矩阵元

    关系:
      S_l = e^{2iδ_l}
      T_l = (S_l - 1) / (2i) = e^{iδ_l} sin δ_l

    幺正性要求: Im T_l = |T_l|² = sin² δ_l

    Parameters
    ----------
    delta_l : complex or ndarray
        相移 (可为复数，吸收非弹性道)

    Returns
    -------
    T_l : complex or ndarray
        T 矩阵元
    """
    delta_l = np.asarray(delta_l, dtype=np.complex128)
    T_l = np.exp(1j * delta_l) * np.sin(delta_l)
    return T_l


def t_matrix_to_phase_shift(T_l):
    """
    从 T 矩阵元反推相移

    δ_l = -i/2 · ln(1 + 2i T_l)

    注意分支切割的选择: δ_l 取主值，Re(δ_l) ∈ (-π/2, π/2]

    Parameters
    ----------
    T_l : complex or ndarray
        T 矩阵元

    Returns
    -------
    delta_l : complex or ndarray
        相移
    """
    T_l = np.asarray(T_l, dtype=np.complex128)
    arg = 1.0 + 2j * T_l
    # 防止 log(0)
    arg = np.where(np.abs(arg) < EPS_MACH, EPS_MACH, arg)
    delta_l = -0.5j * np.log(arg)
    return delta_l


def effective_range_params(sqrt_s_grid, delta_l_grid, l_quantum,
                           m_a=MASS_PI_PLUS, m_b=MASS_PI_PLUS):
    """
    从相移数据提取有效范围参数 (散射长度 a_l, 有效力程 r_l)

    有效范围展开 (Effective Range Expansion, ERE):
      k^{2l+1} cot δ_l = -1/a_l + (1/2) r_l k² + v₂ k⁴ + ...

    对于 S 波 (l=0):
      k cot δ₀ = -1/a₀ + (1/2) r₀ k² + O(k⁴)

    对于 P 波 (l=1):
      k³ cot δ₁ = -1/a₁ + (1/2) r₁ k² + O(k⁴)

    方法: 对 k^{2l+1} cot δ_l 关于 k² 做线性拟合，
    截距 = -1/a_l，斜率 = r_l/2

    Parameters
    ----------
    sqrt_s_grid : ndarray
        √s 网格 [GeV]
    delta_l_grid : ndarray
        相移 δ_l(√s) [弧度]
    l_quantum : int
        角动量量子数
    m_a, m_b : float
        粒子质量 [GeV]

    Returns
    -------
    a_l : complex
        散射长度 (散射体积 for l>0)
    r_l : complex
        有效力程参数
    k_grid : ndarray
        质心动量网格
    k2l1_cot_delta : ndarray
        k^{2l+1} cot δ_l 数据
    """
    from constants import cm_momentum
    s_grid = sqrt_s_grid ** 2
    k_grid = np.array([cm_momentum(s, m_a, m_b) for s in s_grid])

    # 阈值以上
    valid = k_grid > 1e-6
    if np.sum(valid) < 3:
        return complex(0), complex(0), k_grid, np.zeros_like(k_grid)

    k_valid = k_grid[valid]
    delta_valid = delta_l_grid[valid]

    # k^{2l+1} cot δ_l
    k2l1 = k_valid ** (2 * l_quantum + 1)
    cot_delta = 1.0 / np.tan(delta_valid + 0j)
    k2l1_cot_delta = k2l1 * cot_delta

    k_sq = k_valid ** 2

    # 线性拟合: k^{2l+1} cot δ_l = A + B k²
    # A = -1/a_l, B = r_l/2
    if len(k_sq) >= 2:
        A_mat = np.column_stack([np.ones_like(k_sq), k_sq])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A_mat, k2l1_cot_delta, rcond=None)
            A_coeff = coeffs[0]  # -1/a_l
            B_coeff = coeffs[1]  # r_l/2
            a_l = -1.0 / A_coeff if abs(A_coeff) > EPS_MACH else complex(np.inf)
            r_l = 2.0 * B_coeff
        except np.linalg.LinAlgError:
            a_l = complex(0)
            r_l = complex(0)
    else:
        a_l = complex(0)
        r_l = complex(0)

    return a_l, r_l, k_grid, k2l1_cot_delta


# ---------------------------------------------------------------------------
# 分波展开 — 散射振幅构造
# ---------------------------------------------------------------------------

def scattering_amplitude_partial_wave(cos_theta, k_cm, delta_l_array):
    """
    通过分波展开计算散射振幅 f(θ)

    f(θ) = (1/k) Σ_{l=0}^{L_max} (2l+1) e^{iδ_l} sin(δ_l) P_l(cos θ)

    Parameters
    ----------
    cos_theta : float or ndarray
        散射角余弦
    k_cm : float
        质心系动量 [GeV]
    delta_l_array : ndarray, shape (L_max+1,)
        各分波相移 δ_l

    Returns
    -------
    f_theta : complex or ndarray
        散射振幅 [GeV⁻¹]
    """
    if k_cm < EPS_MACH:
        return np.zeros_like(cos_theta, dtype=np.complex128)

    L_max = len(delta_l_array) - 1
    cos_theta = np.asarray(cos_theta, dtype=np.float64)
    P_l = legendre_p(L_max, cos_theta)

    f_theta = np.zeros_like(cos_theta, dtype=np.complex128)
    for l in range(L_max + 1):
        T_l = np.exp(1j * delta_l_array[l]) * np.sin(delta_l_array[l])
        f_theta += (2 * l + 1) * T_l * P_l[l]
    f_theta /= k_cm
    return f_theta


def differential_cross_section(cos_theta, k_cm, delta_l_array):
    """
    计算微分截面 dσ/dΩ

    dσ/dΩ = |f(θ)|²

    Parameters
    ----------
    cos_theta : float or ndarray
        散射角余弦
    k_cm : float
        质心系动量
    delta_l_array : ndarray
        分波相移

    Returns
    -------
    ds_domega : float or ndarray
        微分截面 [GeV⁻²]
    """
    f_theta = scattering_amplitude_partial_wave(cos_theta, k_cm, delta_l_array)
    return np.abs(f_theta) ** 2


def total_cross_section(k_cm, delta_l_array, L_max=None):
    """
    计算总截面 σ_tot (光学定理)

    σ_tot = (4π/k²) Σ_{l=0}^{L_max} (2l+1) sin²(δ_l)

    等价形式 (光学定理):
    σ_tot = (4π/k) Im f(0)

    Parameters
    ----------
    k_cm : float
        质心系动量
    delta_l_array : ndarray
        分波相移
    L_max : int, optional
        截断角动量 (默认为 len-1)

    Returns
    -------
    sigma_tot : float
        总截面 [GeV⁻²]
    """
    if k_cm < EPS_MACH:
        return 0.0
    if L_max is None:
        L_max = len(delta_l_array) - 1
    sigma = 0.0
    for l in range(min(L_max + 1, len(delta_l_array))):
        sigma += (2 * l + 1) * np.sin(np.real(delta_l_array[l])) ** 2
    return FOURPI / (k_cm ** 2) * sigma


def elastic_cross_section(k_cm, delta_l_array):
    """
    弹性截面 σ_el

    σ_el = ∫ |f(θ)|² dΩ = (4π/k²) Σ_l (2l+1) |T_l|²

    对于纯弹性散射: σ_el = σ_tot

    Parameters
    ----------
    k_cm : float
        质心系动量
    delta_l_array : ndarray
        分波相移

    Returns
    -------
    sigma_el : float
        弹性截面 [GeV⁻²]
    """
    if k_cm < EPS_MACH:
        return 0.0
    sigma = 0.0
    for l, delta in enumerate(delta_l_array):
        T_l = np.exp(1j * delta) * np.sin(delta)
        sigma += (2 * l + 1) * np.abs(T_l) ** 2
    return FOURPI / (k_cm ** 2) * sigma


# ---------------------------------------------------------------------------
# 振幅场结构 (融合 415_fem2d_scalar_display_brief)
# ---------------------------------------------------------------------------

class ScatteringAmplitudeField:
    """
    二维散射振幅场 f(√s, cos θ*) 的离散表示

    借鉴 415_fem2d_scalar_display_brief 中的标量场数据结构，
    存储双变量散射振幅场，支持:
      - 场值查询与插值
      - 范数计算
      - 对称性检查

    Attributes
    ----------
    sqrt_s_values : ndarray
        √s 网格
    cos_theta_values : ndarray
        cos θ* 网格
    field : ndarray, shape (n_s, n_theta)
        复数散射振幅场
    """

    def __init__(self, sqrt_s_values, cos_theta_values):
        self.sqrt_s_values = np.asarray(sqrt_s_values)
        self.cos_theta_values = np.asarray(cos_theta_values)
        n_s = len(self.sqrt_s_values)
        n_theta = len(self.cos_theta_values)
        self.field = np.zeros((n_s, n_theta), dtype=np.complex128)
        self.n_s = n_s
        self.n_theta = n_theta

    def set_from_partial_waves(self, delta_l_array_func, k_func):
        """
        从分波相移函数填充振幅场

        Parameters
        ----------
        delta_l_array_func : callable
            函数 sqrt_s -> delta_l_array
        k_func : callable
            函数 sqrt_s -> k_cm
        """
        for i, ss in enumerate(self.sqrt_s_values):
            k = k_func(ss)
            if k < EPS_MACH:
                continue
            delta_l = delta_l_array_func(ss)
            for j, ct in enumerate(self.cos_theta_values):
                self.field[i, j] = scattering_amplitude_partial_wave(
                    ct, k, delta_l)

    def L2_norm(self):
        """
        计算振幅场的 L² 范数

        ||f||² = ∫∫ |f(√s, cos θ)|² d√s d(cos θ)

        使用梯形法则。

        Returns
        -------
        float
            L² 范数
        """
        intensity = np.abs(self.field) ** 2
        # 对 cos θ 积分
        integral_theta = np.trapz(intensity, self.cos_theta_values, axis=1)
        # 对 √s 积分
        total = np.trapz(integral_theta, self.sqrt_s_values)
        return np.sqrt(abs(total))

    def max_amplitude(self):
        """
        返回振幅场的最大绝对值

        Returns
        -------
        float
        """
        return float(np.max(np.abs(self.field)))

    def check_crossing_symmetry(self, tol=1e-6):
        """
        检查全同粒子散射的交叉对称性

        对于 ππ → ππ 散射:
          f(s, cos θ) = f(s, -cos θ)   (对于 I=0,2 通道)
          f(s, cos θ) = -f(s, -cos θ)  (对于 I=1 通道)

        Parameters
        ----------
        tol : float
            对称性检查容差

        Returns
        -------
        bool
            是否满足对称性
        """
        if self.n_theta < 2:
            return True
        # 检查 cos θ ↔ -cos θ 对称性 (以 θ=π/2 为中心)
        mid = self.n_theta // 2
        if mid == 0:
            return True
        for i in range(self.n_s):
            for j in range(min(mid, self.n_theta - mid)):
                j_mirror = self.n_theta - 1 - j
                if j_mirror < self.n_theta:
                    diff = abs(self.field[i, j] - self.field[i, j_mirror])
                    if diff > tol * max(abs(self.field[i, j]), 1.0):
                        return False
        return True


# ---------------------------------------------------------------------------
# 全同粒子对称化 (融合 1372_unicycle 循环结构)
# ---------------------------------------------------------------------------

def symmetrize_amplitude_identical_particles(f_direct, f_exchange):
    """
    全同玻色子散射振幅的对称化

    对于 a + a → a + a 散射 (如 π⁰π⁰ → π⁰π⁰):
      f_sym(θ) = f(θ) + f(π - θ)

    对于全同费米子:
      f_sym(θ) = f(θ) - f(π - θ)

    这与置换群 S_2 的不可约表示对应:
      对称 (玻色子): trivial representation
      反对称 (费米子): sign representation

    使用 1372_unicycle 中的循环分解来判断置换的符号。

    Parameters
    ----------
    f_direct : complex or ndarray
        直接散射振幅 f(θ)
    f_exchange : complex or ndarray
        交换散射振幅 f(π-θ)

    Returns
    -------
    f_boson : complex or ndarray
        玻色子对称化振幅
    f_fermion : complex or ndarray
        费米子反对称化振幅
    """
    f_boson = f_direct + f_exchange
    f_fermion = f_direct - f_exchange
    return f_boson, f_fermion


def multi_channel_symmetry_factor(n_particles, channel_permutation):
    """
    多通道散射的对称性因子

    对于 n 个全同粒子的散射，振幅在所有粒子置换下的变换:
      A_{σ(1)...σ(n)} = ε(σ) A_{1...n}

    其中 ε(σ) 由置换的循环类型决定 (来自 1372_unicycle):
      ε(σ) = ∏_{cycles} (-1)^{length - 1}

    Parameters
    ----------
    n_particles : int
        粒子数
    channel_permutation : list of int
        通道置换

    Returns
    -------
    sign : int
        对称性因子 (+1 或 -1)
    """
    cycles, sign = permutation_cycle_type(channel_permutation)
    return sign
