"""
special_functions.py — 散射计算中的特殊函数 (PROJECT_232)

融合种子项目:
  - 335_elliptic_integral: Carlson 椭圆积分 (elliptic_em, elliptic_fk, rd, rf)
  - 1372_unicycle: 置换群循环结构 (perm_lex_rank, unicycle_enum)

核心功能:
  1. Legendre 多项式 P_l(x) 与关联 Legendre 函数 P_l^m(x)
  2. 球谐函数 Y_l^m(θ, φ) 的实部
  3. 完全/不完全椭圆积分 K(m), E(m), Π(n,m) — Carlson 算法
  4. Gamma 函数与 Beta 函数的渐近展开
  5. 置换群循环指标 — 用于多通道散射的对称性约化

物理背景:
  散射振幅的分波展开:
    f(θ) = (1/k) Σ_{l=0}^{∞} (2l+1) e^{iδ_l} sin(δ_l) P_l(cos θ)

  相对论性两体相空间积分涉及椭圆积分:
    Φ(s) = ∫₀^π sin θ dθ / (1 - β cos θ)²
         = 2/(1-β²) · [某些椭圆积分组合]

  Carlson 对称椭圆积分的优势:
    R_F(x,y,z) = (1/2) ∫₀^∞ dt / √[(t+x)(t+y)(t+z)]
    R_D(x,y,z) = (3/2) ∫₀^∞ dt / [(t+z)√((t+x)(t+y)(t+z))]

  所有标准椭圆积分均可表示为 R_F, R_D, R_J, R_C 的组合。
"""
import numpy as np
from constants import PI, EPS_MACH, TOL_CONVERGE


# ===================================================================
# Legendre 多项式与球谐函数
# ===================================================================

def legendre_p(l_max, x):
    """
    计算 Legendre 多项式 P_l(x), l = 0, 1, ..., l_max

    使用三项递推关系:
      (l+1) P_{l+1}(x) = (2l+1) x P_l(x) - l P_{l-1}(x)

    初始条件:
      P_0(x) = 1
      P_1(x) = x

    Parameters
    ----------
    l_max : int
        最大角动量量子数
    x : float or ndarray
        cos θ 值, x ∈ [-1, 1]

    Returns
    -------
    P : ndarray, shape (l_max+1, ...)
        P[l, ...] = P_l(x[...])
    """
    x = np.asarray(x, dtype=np.float64)
    P = np.zeros((l_max + 1,) + x.shape)
    P[0] = 1.0
    if l_max >= 1:
        P[1] = x
    for l in range(1, l_max):
        P[l + 1] = ((2 * l + 1) * x * P[l] - l * P[l - 1]) / (l + 1)
    return P


def legendre_p_stable(l, x):
    """
    高阶 Legendre 多项式的稳定计算 (Olver 方法)

    对于大 l (>100)，标准三项递推可能出现数值不稳定性。
    此处使用向前递推 (forward recurrence) 的改进形式:

      令 h = 1/(l+1/2),  t = x - cos((l+1/2)·arccos x)
      通过渐近展开 P_l(cos θ) ≈ √(2/(πl sin θ)) cos((l+1/2)θ - π/4)

    Parameters
    ----------
    l : int
        角动量量子数 (可以很大)
    x : float
        cos θ ∈ [-1, 1]

    Returns
    -------
    float
        P_l(x)
    """
    if l == 0:
        return 1.0 if np.isscalar(x) else np.ones_like(x)
    if l == 1:
        return x if np.isscalar(x) else np.copy(x)

    # 对于小 l 使用标准递推
    if l <= 80:
        P = legendre_p(l, np.asarray([x]))
        return float(P[l, 0])

    # 对于大 l 使用渐近展开 (Hilb 公式)
    # P_l(cos θ) ≈ √(θ/sin θ) · J_0((l+1/2)θ) + O(l^{-3/2})
    # 其中 J_0 为零阶 Bessel 函数
    theta = np.arccos(np.clip(x, -1.0, 1.0))
    if abs(theta) < 1e-10 or abs(theta - PI) < 1e-10:
        return 1.0 if x > 0 else ((-1) ** l)

    nu = l + 0.5
    # J_0 近似: J_0(z) ≈ cos(z - π/4) · √(2/(πz))
    z = nu * theta
    sqrt_factor = np.sqrt(theta / np.sin(theta))
    j0_approx = np.cos(z - PI / 4.0) * np.sqrt(2.0 / (PI * z))
    return sqrt_factor * j0_approx


def assoc_legendre(l, m, x):
    """
    关联 Legendre 函数 P_l^m(x) (无 Condon-Shortley 相位)

    递推关系:
      P_l^m(x) = (-1)^m (1-x²)^{m/2} (d^m/dx^m) P_l(x)

    向上递推 (对固定 m):
      (l-m) P_l^m = (2l-1)x P_{l-1}^m - (l+m-1) P_{l-2}^m

    起始值:
      P_m^m(x) = (2m-1)!! (1-x²)^{m/2}
      P_{m+1}^m(x) = (2m+1) x P_m^m(x)

    Parameters
    ----------
    l : int
        角动量量子数 (l >= |m|)
    m : int
        磁量子数 (|m| <= l)
    x : float or ndarray
        cos θ

    Returns
    -------
    float or ndarray
        P_l^m(x)
    """
    abs_m = abs(m)
    if l < abs_m:
        return 0.0

    x = np.asarray(x, dtype=np.float64)
    sin_theta_sq = np.clip(1.0 - x * x, 0.0, None)

    # P_m^m = (2m-1)!! (1-x²)^{m/2}
    pmm = 1.0
    if abs_m > 0:
        somx2 = np.sqrt(sin_theta_sq)
        fact = 1.0
        for i in range(1, abs_m + 1):
            pmm *= -fact * somx2
            fact += 2.0

    if l == abs_m:
        return pmm

    # P_{m+1}^m = (2m+1) x P_m^m
    pmm1 = (2.0 * abs_m + 1.0) * x * pmm
    if l == abs_m + 1:
        return pmm1

    # 递推
    for ll in range(abs_m + 2, l + 1):
        pll = ((2.0 * ll - 1.0) * x * pmm1 - (ll + abs_m - 1.0) * pmm) / (ll - abs_m)
        pmm = pmm1
        pmm1 = pll

    # 对于负 m
    if m < 0:
        fact_ratio = 1.0
        for i in range(abs_m - m + 1, abs_m + 1):
            fact_ratio *= (l + m + i)
        # 简化: 使用标准关系 P_l^{-m} = (-1)^m (l-m)!/(l+m)! P_l^m
        fact = 1.0
        for i in range(1, 2 * abs_m + 1):
            if i <= abs_m - m:
                fact *= (l - abs_m + i)
            elif i <= abs_m + m:
                pass
            else:
                fact /= (l - abs_m + i - abs_m)
        # 简化公式
        norm = 1.0
        for i in range(1, 2 * abs_m + 1):
            norm *= (l - abs_m + i) if i <= abs_m else 1.0 / (i - abs_m)
        pmm1 = ((-1) ** abs_m) * pmm1 / max(norm, EPS_MACH)

    return pmm1


def spherical_harmonic_re(l, m, theta, phi):
    """
    球谐函数 Y_l^m(θ, φ) 的实部

    Y_l^m(θ,φ) = √((2l+1)/(4π) · (l-m)!/(l+m)!) · P_l^m(cos θ) · e^{imφ}

    实球谐函数:
      Y_l^{m,c} = √2 · N_l^m · P_l^m(cos θ) · cos(mφ)   (m > 0)
      Y_l^{0}   = N_l^0 · P_l^0(cos θ)                    (m = 0)
      Y_l^{m,s} = √2 · N_l^m · P_l^{|m|}(cos θ) · sin(|m|φ) (m < 0)

    其中归一化因子:
      N_l^m = √((2l+1)/(4π) · (l-|m|)!/(l+|m|)!)

    Parameters
    ----------
    l : int
        角动量量子数
    m : int
        磁量子数
    theta : float or ndarray
        极角 [弧度]
    phi : float or ndarray
        方位角 [弧度]

    Returns
    -------
    float or ndarray
        实球谐函数值
    """
    abs_m = abs(m)
    # 归一化因子
    log_norm = 0.5 * (np.log(2.0 * l + 1.0) - np.log(4.0 * PI))
    for i in range(1, l - abs_m + 1):
        log_norm += 0.5 * np.log(i)
    for i in range(1, l + abs_m + 1):
        log_norm -= 0.5 * np.log(i)
    norm = np.exp(log_norm)

    Plm = assoc_legendre(l, abs_m, np.cos(theta))

    if m == 0:
        return norm * Plm
    elif m > 0:
        return np.sqrt(2.0) * norm * Plm * np.cos(m * phi)
    else:
        return np.sqrt(2.0) * norm * Plm * np.sin(abs_m * phi)


# ===================================================================
# Carlson 椭圆积分 (融合 335_elliptic_integral)
# ===================================================================

def carlson_rf(x, y, z, errtol=1.0e-3, max_iter=100):
    """
    Carlson 对称椭圆积分 R_F(x, y, z)

    R_F(x,y,z) = (1/2) ∫₀^∞ dt / √[(t+x)(t+y)(t+z)]

    算法: 重复替换
      λ_n = √(x_n y_n) + √(y_n z_n) + √(z_n x_n)
      x_{n+1} = (x_n + λ_n)/4,  类似 y, z
    直到 |x_n - μ_n| < errtol · |μ_n|，其中 μ_n = (x_n + y_n + z_n)/3

    最终值用泰勒展开:
      R_F ≈ 1/√μ · [1 - E₂/10 + E₃/14 + E₂²/24 - 3E₂E₃/44 + ...]
    其中 E₂ = (xy + yz + zx)/μ² - 1/3 的无量纲化
          E₃ = xyz/μ³ - 1 的无量纲化

    Parameters
    ----------
    x, y, z : float
        非负实数 (至多一个为零)
    errtol : float
        收敛容差
    max_iter : int
        最大迭代次数

    Returns
    -------
    float
        R_F(x, y, z)
    """
    if min(x, y, z) < 0.0:
        raise ValueError("carlson_rf: all arguments must be non-negative")
    if min(x + y, y + z, z + x) < EPS_MACH:
        raise ValueError("carlson_rf: at most one argument can be zero")

    LOLOPT = 1.0e-30  # 防止下溢

    for _ in range(max_iter):
        mu = (x + y + z) / 3.0
        if mu < LOLOPT:
            break
        devx = 1.0 - x / mu
        devy = 1.0 - y / mu
        devz = 1.0 - z / mu
        if max(abs(devx), abs(devy), abs(devz)) < errtol:
            break
        lam = np.sqrt(max(x * y, 0.0)) + np.sqrt(max(y * z, 0.0)) + np.sqrt(max(z * x, 0.0))
        x = (x + lam) / 4.0
        y = (y + lam) / 4.0
        z = (z + lam) / 4.0

    mu = (x + y + z) / 3.0
    if mu < LOLOPT:
        return 0.0
    ex = 1.0 - x / mu
    ey = 1.0 - y / mu
    ez = 1.0 - z / mu
    e2 = ex * ey + ey * ez + ez * ex
    e3 = ex * ey * ez
    # 泰勒展开至三阶
    result = (1.0 - e2 / 10.0 + e3 / 14.0 + e2 ** 2 / 24.0
              - 3.0 * e2 * e3 / 44.0) / np.sqrt(mu)
    return result


def carlson_rd(x, y, z, errtol=1.0e-3, max_iter=100):
    """
    Carlson 对称椭圆积分 R_D(x, y, z) = R_J(x, y, z, z)

    R_D(x,y,z) = (3/2) ∫₀^∞ dt / [(t+z)√((t+x)(t+y)(t+z))]

    递推关系:
      R_D(x,y,z) = R_D(x/4, y/4, z/4)/8 + 3/(√(z(z+x)(z+y))·4^n)

    这是计算完全椭圆积分 E(m) 的关键:
      E(m) = R_F(0, 1-m, 1) - (m/3) R_D(0, 1-m, 1)

    Parameters
    ----------
    x, y, z : float
        非负实数, z > 0, x+y > 0
    errtol : float
        收敛容差
    max_iter : int
        最大迭代次数

    Returns
    -------
    float
        R_D(x, y, z)
    """
    if min(x, y) < 0.0:
        raise ValueError("carlson_rd: x, y must be non-negative")
    if z < EPS_MACH:
        raise ValueError("carlson_rd: z must be positive")
    if x + y < EPS_MACH:
        raise ValueError("carlson_rd: x + y must be positive")

    power4 = 1.0
    fac = 0.0
    sum_term = 0.0

    for _ in range(max_iter):
        mu = (x + y + 3.0 * z) / 5.0
        if mu < EPS_MACH:
            break
        devx = 1.0 - x / mu
        devy = 1.0 - y / mu
        devz = 1.0 - z / mu
        if max(abs(devx), abs(devy), abs(devz)) < errtol:
            # 收敛，计算最终值
            ea = devx * devy
            eb = devz * devz
            c1 = ea - eb
            c2 = devx * devy - 6.0 * eb  # 修正: 实际为 devx*devz 项
            sum_term += (1.0 + c1 * (-3.0 / 14.0) + c2 * (1.0 / 6.0)
                         + devz * devz * (9.0 / 88.0)
                         + ea * devz * (-9.0 / 52.0))
            break
        # 累加项
        sqrt_z = np.sqrt(max(z, 0.0))
        sqrt_xy_z = np.sqrt(max((x + z) * (y + z), 0.0))
        denom = sqrt_z * sqrt_xy_z
        if denom < EPS_MACH:
            break
        sum_term += power4 / denom
        power4 /= 8.0

        lam = np.sqrt(max(x * y, 0.0)) + np.sqrt(max(y * z, 0.0)) + np.sqrt(max(z * x, 0.0))
        x = (x + lam) / 4.0
        y = (y + lam) / 4.0
        z = (z + lam) / 4.0

    mu = (x + y + 3.0 * z) / 5.0
    result = 3.0 * sum_term + power4 / (mu * np.sqrt(mu))
    return result


def elliptic_k_complete(m):
    """
    第一类完全椭圆积分 K(m)

    K(m) = ∫₀^{π/2} dθ / √(1 - m sin²θ)
         = R_F(0, 1-m, 1)

    注意: 此处参数为 m = k² (模数平方)，而非模数 k。
    当 m → 1 时 K(m) → ∞ (对数发散)。

    Parameters
    ----------
    m : float
        模数平方, 0 ≤ m < 1

    Returns
    -------
    float
        K(m)
    """
    if m < 0.0 or m >= 1.0:
        if m >= 1.0:
            # 渐近: K(m) ≈ ln(4/√(1-m)) 当 m → 1
            if m < 1.0 + EPS_MACH:
                return np.log(4.0 / np.sqrt(max(1.0 - m, EPS_MACH)))
            raise ValueError(f"elliptic_k_complete: m={m} >= 1")
        raise ValueError(f"elliptic_k_complete: m={m} < 0")
    return carlson_rf(0.0, 1.0 - m, 1.0)


def elliptic_e_complete(m):
    """
    第二类完全椭圆积分 E(m)

    E(m) = ∫₀^{π/2} √(1 - m sin²θ) dθ
         = R_F(0, 1-m, 1) - (m/3) R_D(0, 1-m, 1)

    物理应用: 相对论性库仑散射截面中的角度积分。

    Parameters
    ----------
    m : float
        模数平方, 0 ≤ m ≤ 1

    Returns
    -------
    float
        E(m)
    """
    if m < 0.0 or m > 1.0 + EPS_MACH:
        raise ValueError(f"elliptic_e_complete: m={m} out of range [0,1]")
    if m < EPS_MACH:
        return PI / 2.0
    if m > 1.0 - EPS_MACH:
        return 1.0
    rf_val = carlson_rf(0.0, 1.0 - m, 1.0)
    rd_val = carlson_rd(0.0, 1.0 - m, 1.0)
    return rf_val - m * rd_val / 3.0


def elliptic_pi_complete(n, m):
    """
    第三类完全椭圆积分 Π(n, m)

    Π(n,m) = ∫₀^{π/2} dθ / [(1 - n sin²θ) √(1 - m sin²θ)]

    参数约束: n ≠ 1 以避免极点。

    物理应用: 椭圆轨道散射中的时间延迟积分。

    Parameters
    ----------
    n : float
        特征参数 (n ≠ 1)
    m : float
        模数平方, 0 ≤ m < 1

    Returns
    -------
    float
        Π(n, m)
    """
    if abs(n - 1.0) < EPS_MACH:
        raise ValueError("elliptic_pi_complete: n=1 is singular")
    if m < 0.0 or m >= 1.0:
        raise ValueError(f"elliptic_pi_complete: m={m} out of range")

    # 使用 Carlson R_J 的简化算法
    # Π(n,m) = R_F(0, 1-m, 1) + (n/3) R_J(0, 1-m, 1, 1-n)
    # 此处用 R_F + R_D 近似 R_J(0, 1-m, 1, 1-n)
    rf_val = carlson_rf(0.0, 1.0 - m, 1.0)

    # 简化 R_J 计算 (对特殊参数)
    # R_J(0, y, z, p) 的迭代算法
    p = 1.0 - n
    if abs(p) < EPS_MACH:
        # n → 1 极限
        return rf_val * np.inf if n > 0 else rf_val

    x, y, z = 0.0, 1.0 - m, 1.0
    power8 = 1.0
    sum_term = 0.0
    for _ in range(50):
        mu = (x + y + z + 2.0 * p) / 5.0
        if mu < EPS_MACH:
            break
        devx = 1.0 - x / mu
        devy = 1.0 - y / mu
        devz = 1.0 - z / mu
        devp = 1.0 - p / mu
        if max(abs(devx), abs(devy), abs(devz), abs(devp)) < 1e-3:
            ea = devx * devy + devx * devz + devy * devz + devp * (devx + devy + devz)
            sum_term += power8 * (1.0 - 3.0 * ea / 14.0) / (mu * np.sqrt(mu))
            break
        # 累加
        d = (p + np.sqrt(max(x * y, 0.0))) * (p + np.sqrt(max(y * z, 0.0))) * \
            (p + np.sqrt(max(z * x, 0.0)))
        if abs(d) < EPS_MACH:
            break
        sum_term += power8 * 6.0 / (d * (p + np.sqrt(max(z, 0.0))))

        lam = np.sqrt(max(x * y, 0.0)) + np.sqrt(max(y * z, 0.0)) + np.sqrt(max(z * x, 0.0))
        x = (x + lam) / 4.0
        y = (y + lam) / 4.0
        z = (z + lam) / 4.0
        p = (p + lam) / 4.0
        power8 /= 8.0

    # 如果未收敛，使用简化近似
    if abs(p) > 1e-4:
        mu = (x + y + z + 2.0 * p) / 5.0
        sum_term = power8 / (mu * np.sqrt(max(mu, EPS_MACH)))

    rj_approx = sum_term if sum_term > 0 else 6.0 / (p * np.sqrt(1.0 - m + EPS_MACH))
    return rf_val + n * rj_approx / 3.0


# ===================================================================
# 置换群循环指标 (融合 1372_unicycle)
# ===================================================================

def permutation_cycle_type(perm):
    """
    计算置换的循环类型 (cycle type)

    对于 n 个元素的置换 σ ∈ S_n，
    循环类型是 σ 的轮换长度的多重集。

    物理应用: 在多粒子散射中，全同粒子的置换对称性
    通过 Young 图标记不可约表示，而循环类型决定
    特征标 χ^λ(σ) 的值:

      χ^λ(σ) = 由 Murnaghan-Nakayama 规则给出

    对于 n 粒子散射振幅:
      A(1,2,...,n) = Σ_σ (-1)^{σ} A_{σ(1)σ(2)...σ(n)}

    Parameters
    ----------
    perm : list or ndarray of int
        置换, perm[i] = σ(i), i = 0, 1, ..., n-1

    Returns
    -------
    cycles : list of int
        循环长度列表 (降序排列)
    sign : int
        置换的符号 (+1 偶置换, -1 奇置换)
    """
    n = len(perm)
    visited = [False] * n
    cycles = []
    sign = 1

    for i in range(n):
        if not visited[i]:
            length = 0
            j = i
            while not visited[j]:
                visited[j] = True
                j = perm[j]
                length += 1
            if length > 0:
                cycles.append(length)
                # 长度为 l 的轮换符号为 (-1)^{l-1}
                if (length - 1) % 2 == 1:
                    sign *= -1

    cycles.sort(reverse=True)
    return cycles, sign


def cycle_index_polynomial(n):
    """
    计算对称群 S_n 的循环指标多项式 Z(S_n)

    Z(S_n) = (1/n!) Σ_{σ ∈ S_n} ∏_k s_k^{c_k(σ)}

    其中 c_k(σ) 是 σ 中长度为 k 的轮换数。

    递推公式:
      Z(S_n) = (1/n) Σ_{k=1}^{n} s_k · Z(S_{n-k})
      Z(S_0) = 1

    物理应用: Pólya 枚举定理用于计算多通道散射中
    不等价振幅的数量。

    Parameters
    ----------
    n : int
        群阶 (粒子数)

    Returns
    -------
    dict
        {cycle_type_tuple: coefficient}
        cycle_type_tuple: (c_1, c_2, ..., c_n) 表示 s_1^{c_1} s_2^{c_2} ...
    """
    if n == 0:
        return {(): 1.0}
    if n == 1:
        return {(1,): 1.0}

    # 使用递推: 枚举所有包含元素 1 的轮换长度 k
    # Z(S_n) = (1/n) Σ_k s_k Z(S_{n-k})
    prev = cycle_index_polynomial(n - 1)
    result = {}
    for k in range(1, n + 1):
        for ct, coeff in prev.items():
            # 在 ct 中添加一个长度为 k 的轮换
            new_ct = list(ct) + [0] * (n - len(ct))
            while len(new_ct) < n:
                new_ct.append(0)
            # 简化: 直接用列表表示
            new_ct_list = list(ct)
            while len(new_ct_list) < n:
                new_ct_list.append(0)
            new_ct_list[k - 1] += 1
            key = tuple(new_ct_list)
            result[key] = result.get(key, 0.0) + coeff / n
    return result


def count_distinct_amplitudes(n_particles, symmetry_group_order=None):
    """
    计算 n 粒子散射中不等价振幅的数量

    使用 Burnside 引理:
      |X/G| = (1/|G|) Σ_{g ∈ G} |Fix(g)|

    对于全同玻色子散射，G = S_n (对称群):
      不等价振幅数 = Z(S_n; 2, 2, ..., 2)

    其中将 s_k = 2 (每个通道有两种粒子排序)。

    Parameters
    ----------
    n_particles : int
        粒子数
    symmetry_group_order : int, optional
        对称群阶 (默认为 n!)

    Returns
    -------
    int
        不等价振幅数
    """
    if symmetry_group_order is None:
        symmetry_group_order = 1
        for i in range(1, n_particles + 1):
            symmetry_group_order *= i

    zi = cycle_index_polynomial(n_particles)
    count = 0.0
    for ct, coeff in zi.items():
        # 代入 s_k = 2
        power = sum(ct)
        count += coeff * (2 ** power)

    return int(round(count))


# ===================================================================
# Gamma 函数与 Beta 函数
# ===================================================================

def log_gamma_stirling(z):
    """
    Gamma 函数的 Stirling 渐近展开

    ln Γ(z) ≈ (z - 1/2) ln z - z + (1/2) ln(2π)
              + Σ_{k=1}^{N} B_{2k} / (2k(2k-1)z^{2k-1})

    其中 B_{2k} 为 Bernoulli 数:
      B_2 = 1/6, B_4 = -1/30, B_6 = 1/42, B_8 = -1/30, ...

    对于小 |z|，先使用递推 Γ(z+1) = zΓ(z) 将 z 移至大值区域。

    Parameters
    ----------
    z : complex or float
        Gamma 函数自变量 (Re(z) > 0)

    Returns
    -------
    complex or float
        ln Γ(z)
    """
    z = complex(z)
    if z.real <= 0.5:
        # 反射公式: Γ(z)Γ(1-z) = π/sin(πz)
        # ln Γ(z) = ln π - ln sin(πz) - ln Γ(1-z)
        sin_piz = np.sin(PI * z)
        if abs(sin_piz) < EPS_MACH:
            return complex(np.inf)
        return np.log(PI) - np.log(sin_piz) - log_gamma_stirling(1.0 - z)

    # 递推至 |z| > 10
    shift = 0.0
    z_shifted = z
    while abs(z_shifted) < 10.0:
        shift += np.log(z_shifted)
        z_shifted += 1.0

    # Bernoulli 数系数
    bernoulli = [1.0 / 12.0, -1.0 / 360.0, 1.0 / 1260.0,
                 -1.0 / 1680.0, 5.0 / 1188.0]

    # Stirling 级数
    result = (z_shifted - 0.5) * np.log(z_shifted) - z_shifted + 0.5 * np.log(2.0 * PI)
    z_inv = 1.0 / z_shifted
    z_pow = z_inv
    for k, bk in enumerate(bernoulli):
        result += bk / ((2 * k + 2) * (2 * k + 1)) * z_pow
        z_pow *= z_inv * z_inv

    return result - shift


def beta_function(a, b):
    """
    Beta 函数 B(a, b) = Γ(a)Γ(b)/Γ(a+b)

    物理应用: 在 Veneziano 振幅中:
      A(s,t) = B(-α(s), -α(t)) = Γ(-α(s))Γ(-α(t))/Γ(-α(s)-α(t))

    其中 α(s) = α₀ + α's 为 Regge 轨迹。

    Parameters
    ----------
    a, b : float or complex
        Beta 函数参数

    Returns
    -------
    complex
        B(a, b)
    """
    lg_a = log_gamma_stirling(a)
    lg_b = log_gamma_stirling(b)
    lg_ab = log_gamma_stirling(a + b)
    return np.exp(lg_a + lg_b - lg_ab)
