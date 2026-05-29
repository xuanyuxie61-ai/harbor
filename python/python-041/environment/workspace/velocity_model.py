"""
 velocity_model.py
 
 融合种子项目:
   - 440_florida_cvt_pop: Centroidal Voronoi Tessellation (CVT) 自适应网格
   - 518_hermite_cubic: Hermite 三次样条插值与积分
   - 601_ising_2d_simulation: 2D Ising 模型蒙特卡洛模拟
   - 655_leaf_chaos: 迭代函数系统分形映射（用于分形孔隙扰动）
 
 科学应用:
   在全波形反演中，速度模型是核心输入。真实的地下介质具有以下复杂特征:
   1. 多相性：不同岩相（沉积岩、火成岩、孔隙流体）具有截然不同的弹性波速
   2. 非均质性：速度在空间上连续变化，需要高阶插值保持光滑性
   3. 分形性：孔隙和裂缝网络具有尺度无关的分形统计特征
   
   本模块构建综合速度模型:
     V(x,z) = V0 + dV_ising(x,z) + dV_fractal(x,z) + dV_smooth(x,z)
   
   其中:
     - V0: 背景速度
     - dV_ising: Ising 模型生成的离散岩相分布（+1/-1 映射为高速/低速）
     - dV_fractal: 分形孔隙度扰动（通过 IFS 或 fBm 生成）
     - dV_smooth: CVT 自适应网格上的 Hermite 三次样条光滑插值
"""

import numpy as np
from fractal_scattering import fractal_porosity_field


def hermite_cubic_value(x1, f1, d1, x2, f2, d2, x):
    """
    在区间 [x1, x2] 上求 Hermite 三次多项式的值和导数。
    
    Hermite 插值多项式满足:
      H(x1) = f1,   H'(x1) = d1
      H(x2) = f2,   H'(x2) = d2
    
    多项式形式:
      H(x) = f1 + dx * (d1 + dx * (c2 + dx * c3))
    其中:
      h = x2 - x1
      df = (f2 - f1) / h
      c2 = -(2*d1 - 3*df + d2) / h
      c3 = (d1 - 2*df + d2) / h^2
    
    Parameters
    ----------
    x1, x2 : float
        区间端点。
    f1, f2 : float
        端点函数值。
    d1, d2 : float
        端点导数值。
    x : float or ndarray
        求值点。
    
    Returns
    -------
    f, d, s, t : float or ndarray
        函数值、一阶导、二阶导、三阶导。
    """
    h = x2 - x1
    if abs(h) < 1e-14:
        return f1, d1, 0.0, 0.0
    df = (f2 - f1) / h
    c2 = -(2.0 * d1 - 3.0 * df + d2) / h
    c3 = (d1 - 2.0 * df + d2) / (h ** 2)
    dx = x - x1
    f = f1 + dx * (d1 + dx * (c2 + dx * c3))
    d = d1 + dx * (2.0 * c2 + dx * 3.0 * c3)
    s = 2.0 * c2 + dx * 6.0 * c3
    t = 6.0 * c3
    return f, d, s, t


def hermite_cubic_spline_value(xn, fn, dn, x):
    """
    求 Hermite 三次样条在任意点处的值。
    
    给定数据点 (xn_i, fn_i) 和导数 dn_i，构造分段 Hermite 三次插值。
    
    Parameters
    ----------
    xn : ndarray, shape (nn,)
        数据点横坐标（严格递增）。
    fn : ndarray, shape (nn,)
        函数值。
    dn : ndarray, shape (nn,)
        导数值。
    x : float or ndarray
        求值点。
    
    Returns
    -------
    f : float or ndarray
        插值结果。
    """
    xn = np.asarray(xn, dtype=float)
    fn = np.asarray(fn, dtype=float)
    dn = np.asarray(dn, dtype=float)
    x_arr = np.atleast_1d(x)
    f_out = np.zeros_like(x_arr, dtype=float)
    nn = len(xn)
    for j in range(len(x_arr)):
        xv = x_arr[j]
        # 找到包含 xv 的区间
        if xv <= xn[0]:
            i1, i2 = 0, 1
        elif xv >= xn[-1]:
            i1, i2 = nn - 2, nn - 1
        else:
            i1 = np.searchsorted(xn, xv) - 1
            i2 = i1 + 1
        f_val, _, _, _ = hermite_cubic_value(
            xn[i1], fn[i1], dn[i1], xn[i2], fn[i2], dn[i2], xv
        )
        f_out[j] = f_val
    if np.isscalar(x):
        return float(f_out[0])
    return f_out


def hermite_cubic_spline_integral(xn, fn, dn):
    """
    计算 Hermite 三次样条的总积分。
    
    解析公式（逐段）:
      Q = sum_i 0.5 * h_i * (fn_i + fn_{i+1} + h_i * (dn_i - dn_{i+1}) / 6)
    
    Parameters
    ----------
    xn, fn, dn : ndarray
        数据点和导数。
    
    Returns
    -------
    q : float
        积分值。
    """
    xn = np.asarray(xn, dtype=float)
    fn = np.asarray(fn, dtype=float)
    dn = np.asarray(dn, dtype=float)
    nn = len(xn)
    if nn < 2:
        return 0.0
    il = np.arange(0, nn - 1)
    ir = np.arange(1, nn)
    h = xn[ir] - xn[il]
    q = np.sum(0.5 * h * (fn[il] + fn[ir] + h * (dn[il] - dn[ir]) / 6.0))
    return q


def ising_2d_initialize(m, n, thresh=0.5, rng=None):
    """
    初始化 2D Ising 模型自旋构型。
    
    每个格点取 +1（高速岩相）或 -1（低速岩相），
    初始分布由阈值 thresh 控制：
      P(spin = -1) = thresh
      P(spin = +1) = 1 - thresh
    
    Parameters
    ----------
    m, n : int
        网格尺寸。
    thresh : float
        -1 自旋的概率。
    rng : numpy.random.Generator, optional
    
    Returns
    -------
    c1 : ndarray, shape (m, n)
        自旋构型。
    """
    if rng is None:
        rng = np.random.default_rng()
    c1 = np.ones((m, n), dtype=int)
    r = rng.random((m, n))
    c1[r <= thresh] = -1
    return c1


def ising_2d_agree(m, n, c1):
    """
    计算每个格点的邻居一致数（含自身，共5个）。
    
    Parameters
    ----------
    m, n : int
        网格尺寸。
    c1 : ndarray, shape (m, n)
        自旋构型。
    
    Returns
    -------
    c5 : ndarray, shape (m, n)
        一致邻居数（1-5）。
    """
    c5 = (
        c1
        + np.roll(c1, -1, axis=0)
        + np.roll(c1, 1, axis=0)
        + np.roll(c1, -1, axis=1)
        + np.roll(c1, 1, axis=1)
    )
    pos_mask = c1 > 0
    neg_mask = c1 < 0
    c5 = c5.astype(float)
    c5[pos_mask] = (5.0 + c5[pos_mask]) / 2.0
    c5[neg_mask] = (5.0 - c5[neg_mask]) / 2.0
    return c5.astype(int)


def ising_2d_transition(m, n, iterations, prob, c1, rng=None):
    """
    执行 2D Ising 模型的蒙特卡洛状态转移。
    
    转移规则（Glauber/Metropolis 型）:
      对每个格点，根据其邻居一致数 c5 和概率表 prob 决定是否翻转自旋。
      c5 = 1..5 对应 prob[0..4]。
    
    物理意义：模拟岩相界面处的相变过程，高温对应随机分布，低温对应有序聚簇。
    
    Parameters
    ----------
    m, n : int
        网格尺寸。
    iterations : int
        迭代次数。
    prob : ndarray, shape (5,)
        翻转概率表（对应 c5=1..5）。
    c1 : ndarray
        初始构型（将被修改）。
    rng : numpy.random.Generator, optional
    
    Returns
    -------
    c1 : ndarray
        最终构型。
    """
    if rng is None:
        rng = np.random.default_rng()
    for step in range(iterations):
        c5 = ising_2d_agree(m, n, c1)
        threshold = np.zeros((m, n))
        for j in range(5):
            mask = (c5 == j + 1)
            threshold[mask] = prob[j]
        r = rng.random((m, n))
        flip = r < threshold
        c1[flip] = -c1[flip]
    return c1


def cvt_sample_generators(n, xlim=(0.0, 1.0), zlim=(0.0, 1.0), rng=None):
    """
    生成 CVT 的初始 generator 点。
    
    Parameters
    ----------
    n : int
        generator 数量。
    xlim, zlim : tuple
        空间范围。
    rng : numpy.random.Generator, optional
    
    Returns
    -------
    gen_x, gen_z : ndarray, shape (n,)
        Generator 坐标。
    """
    if rng is None:
        rng = np.random.default_rng()
    gen_x = rng.uniform(xlim[0], xlim[1], n)
    gen_z = rng.uniform(zlim[0], zlim[1], n)
    return gen_x, gen_z


def cvt_centroid_estimate(gen_x, gen_z, sample_num, xlim, zlim, rng=None):
    """
    使用蒙特卡洛采样估计 Voronoi 区域的重心。
    
    算法:
      1. 在区域内随机采样 sample_num 个点
      2. 将每个点分配给最近的 generator
      3. 计算每个 generator 对应点集的重心
    
    Parameters
    ----------
    gen_x, gen_z : ndarray
        Generator 坐标。
    sample_num : int
        采样点数。
    xlim, zlim : tuple
        区域范围。
    rng : numpy.random.Generator, optional
    
    Returns
    -------
    cen_x, cen_z : ndarray
        重心坐标。
    """
    if rng is None:
        rng = np.random.default_rng()
    n = len(gen_x)
    cen_x = np.zeros(n)
    cen_z = np.zeros(n)
    cen_count = np.zeros(n)
    x_samples = rng.uniform(xlim[0], xlim[1], sample_num)
    z_samples = rng.uniform(zlim[0], zlim[1], sample_num)
    for s in range(sample_num):
        dx = x_samples[s] - gen_x
        dz = z_samples[s] - gen_z
        dist2 = dx ** 2 + dz ** 2
        i_min = np.argmin(dist2)
        cen_x[i_min] += x_samples[s]
        cen_z[i_min] += z_samples[s]
        cen_count[i_min] += 1
    # 处理空单元
    for i in range(n):
        if cen_count[i] == 0:
            cen_x[i] = gen_x[i]
            cen_z[i] = gen_z[i]
        else:
            cen_x[i] /= cen_count[i]
            cen_z[i] /= cen_count[i]
    return cen_x, cen_z


def cvt_optimize(n, xlim, zlim, n_steps=10, sample_num=2000, rng=None):
    """
    优化 Centroidal Voronoi Tessellation (CVT)。
    
    Lloyd 算法:
      重复: generators <- Voronoi 单元重心
    
    在地震观测网优化中，CVT 可生成空间均匀覆盖的检波器布设方案。
    
    Parameters
    ----------
    n : int
        Generator 数量。
    xlim, zlim : tuple
        空间范围。
    n_steps : int
        Lloyd 迭代步数。
    sample_num : int
        每步采样点数。
    rng : numpy.random.Generator, optional
    
    Returns
    -------
    gen_x, gen_z : ndarray
        优化后的 generator 坐标。
    """
    if rng is None:
        rng = np.random.default_rng()
    gen_x, gen_z = cvt_sample_generators(n, xlim, zlim, rng=rng)
    for _ in range(n_steps):
        gen_x, gen_z = cvt_centroid_estimate(gen_x, gen_z, sample_num, xlim, zlim, rng=rng)
    return gen_x, gen_z


def build_velocity_model(nx, nz, v0=3000.0, dv_ising=500.0, dv_fractal=200.0,
                         ising_thresh=0.5, ising_iter=15, fractal_dim=1.8,
                         use_cvt=False, n_cvt=20, rng=None):
    """
    构建综合速度模型。
    
    速度模型公式:
      V(x,z) = v0 + dv_ising * I(x,z) + dv_fractal * F(x,z)
    
    其中 I(x,z) 为 Ising 相场（+1/-1），F(x,z) 为分形孔隙度扰动 [0,1]。
    若启用 CVT，则在 CVT 网格点处定义背景速度并通过 Hermite 样条插值到细网格。
    
    Parameters
    ----------
    nx, nz : int
        细网格尺寸。
    v0 : float
        背景速度（m/s）。
    dv_ising : float
        Ising 相速度扰动幅度。
    dv_fractal : float
        分形扰动幅度。
    ising_thresh : float
        Ising 初始阈值。
    ising_iter : int
        Ising MC 迭代次数。
    fractal_dim : float
        分形维数。
    use_cvt : bool
        是否使用 CVT 网格。
    n_cvt : int
        CVT generator 数量。
    rng : numpy.random.Generator, optional
    
    Returns
    -------
    velocity : ndarray, shape (nz, nx)
        速度模型（m/s）。
    x_coords : ndarray, shape (nx,)
        x 坐标（km）。
    z_coords : ndarray, shape (nz,)
        z 坐标（km）。
    """
    if rng is None:
        rng = np.random.default_rng()
    x_coords = np.linspace(0.0, 1.0, nx)
    z_coords = np.linspace(0.0, 1.0, nz)
    
    # Ising 相场
    ising_field = ising_2d_initialize(nz, nx, thresh=ising_thresh, rng=rng)
    prob = np.array([0.98, 0.85, 0.50, 0.15, 0.02])
    ising_field = ising_2d_transition(nz, nx, ising_iter, prob, ising_field, rng=rng)
    
    # 分形孔隙度扰动
    fractal_field = fractal_porosity_field(nx, nz, fractal_dim=fractal_dim, rng=rng)
    
    # 基础速度场
    velocity = v0 + dv_ising * ising_field + dv_fractal * (fractal_field - 0.5)
    
    # CVT 自适应网格插值（可选）
    if use_cvt and n_cvt >= 2:
        gen_x, gen_z = cvt_optimize(n_cvt, (0.0, 1.0), (0.0, 1.0), n_steps=5,
                                     sample_num=1000, rng=rng)
        # 在 generator 位置采样当前速度
        gen_v = np.zeros(n_cvt)
        for i in range(n_cvt):
            ix = min(int(gen_x[i] * (nx - 1)), nx - 1)
            iz = min(int(gen_z[i] * (nz - 1)), nz - 1)
            gen_v[i] = velocity[iz, ix]
        # 计算数值导数用于 Hermite 插值
        gen_dv = np.zeros(n_cvt)
        gen_dv[0] = (gen_v[1] - gen_v[0]) / (gen_x[1] - gen_x[0])
        gen_dv[-1] = (gen_v[-1] - gen_v[-2]) / (gen_x[-1] - gen_x[-2])
        for i in range(1, n_cvt - 1):
            gen_dv[i] = 0.5 * ((gen_v[i] - gen_v[i - 1]) / (gen_x[i] - gen_x[i - 1])
                                + (gen_v[i + 1] - gen_v[i]) / (gen_x[i + 1] - gen_x[i]))
        # 对 x 方向进行 Hermite 样条插值
        for iz in range(nz):
            # 在 CVT generator x 位置处的速度剖面
            v_profile = hermite_cubic_spline_value(gen_x, gen_v, gen_dv, x_coords)
            velocity[iz, :] = v_profile
    
    # 确保速度为正且合理
    velocity = np.clip(velocity, 1000.0, 8000.0)
    return velocity, x_coords, z_coords
