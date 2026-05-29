"""
 fractal_scattering.py
 
 融合种子项目:
   - 710_mandelbrot: Mandelbrot 集合迭代与逃逸时间
   - 655_leaf_chaos: 迭代函数系统 (IFS) 分形映射
 
 科学应用:
   地球物理介质中的多孔结构和裂缝网络常表现出分形特征。
   本项目将 Mandelbrot 迭代用于检测地震波在复杂介质中的"混沌散射区域"，
   并将 IFS 分形映射用于生成具有分形统计特征的孔隙度扰动场，
   以模拟地震波在分形多孔介质中的散射衰减效应。
"""

import numpy as np


def mandelbrot_escape_time(cx, cy, count_max=50, escape_radius=2.0):
    """
    计算复平面上点的 Mandelbrot 逃逸时间。
    
    Mandelbrot 迭代定义为:
      z_{n+1} = z_n^2 + c,  z_0 = 0
    若 |z_n| <= 2 对所有 n <= count_max 成立，则 c 属于 Mandelbrot 集。
    
    在地震波散射理论中，此迭代可用于类比波在强散射介质中的能量逃逸过程:
    逃逸时间越短，表明波在该区域被快速散射耗散；逃逸时间越长，
    表明波能量被"困住"，对应于强共振散射区域。
    
    Parameters
    ----------
    cx, cy : float or ndarray
        复数 c = cx + i*cy 的实部和虚部。
    count_max : int
        最大迭代次数。
    escape_radius : float
        逃逸半径。
    
    Returns
    -------
    escape : int or ndarray
        逃逸迭代次数（未逃逸则返回 count_max + 1）。
    """
    is_scalar = np.isscalar(cx)
    cx = np.asarray(cx, dtype=float)
    cy = np.asarray(cy, dtype=float)
    zr = np.zeros_like(cx)
    zi = np.zeros_like(cx)
    escape = np.full_like(cx, count_max + 1, dtype=int)
    for i in range(count_max):
        # 使用边界检查抑制数值溢出
        zr = np.clip(zr, -1e6, 1e6)
        zi = np.clip(zi, -1e6, 1e6)
        zr_new = zr * zr - zi * zi + cx
        zi_new = 2.0 * zr * zi + cy
        zr, zi = zr_new, zi_new
        mag = zr * zr + zi * zi
        mask = (mag > escape_radius ** 2) & (escape == count_max + 1)
        escape[mask] = i + 1
    if is_scalar:
        return int(escape.item())
    return escape


def compute_scattering_strength(x_grid, y_grid, center_x=0.0, center_y=0.0,
                                 scale=1.0, count_max=30):
    """
    基于 Mandelbrot 逃逸时间计算地震波散射强度场。
    
    散射强度模型:
      S(x,y) = exp(-escape_time / tau) * Gaussian_envelope
    
    其中 tau 为弛豫时间参数。逃逸时间短的区域（快速逃逸）对应强散射耗散，
    逃逸时间长的区域对应能量局域化和强共振散射。
    
    Parameters
    ----------
    x_grid, y_grid : ndarray
        网格坐标。
    center_x, center_y : float
        Mandelbrot 区域中心。
    scale : float
        缩放因子。
    count_max : int
        最大迭代次数。
    
    Returns
    -------
    strength : ndarray
        散射强度场，范围 [0, 1]。
    """
    cx = (x_grid - center_x) / scale
    cy = (y_grid - center_y) / scale
    escape = mandelbrot_escape_time(cx, cy, count_max=count_max)
    tau = count_max / 3.0
    # 逃逸时间短 -> 高散射（能量快速耗散）
    # 逃逸时间长 -> 低散射（能量被局域化，但此处我们建模为低散射以区分区域）
    strength = np.exp(-escape / tau)
    # 添加高斯包络避免边界效应
    r2 = cx ** 2 + cy ** 2
    envelope = np.exp(-r2 / 4.0)
    return strength * envelope


def ifs_leaf_fractal(n_points=5000, rng=None):
    """
    使用迭代函数系统 (IFS) 生成分形叶片状点集。
    
    IFS 定义为一组仿射变换 {A_i, b_i}，以概率 p_i 迭代应用:
      x_{k+1} = A_i * x_k + b_i
    
    在地球物理中，此 IFS 可用于生成具有分形特征的裂缝网络或孔隙通道
    的几何表示。分形维数 D 满足:
      sum_i |det(A_i)|^{D/2} = 1
    
    Parameters
    ----------
    n_points : int
        生成的点数。
    rng : numpy.random.Generator, optional
    
    Returns
    -------
    points : ndarray, shape (n_points, 2)
        分形点集。
    """
    if rng is None:
        rng = np.random.default_rng()
    # 定义四个仿射变换（Barnsley 蕨类变体）
    A = np.array([
        [[0.80, 0.00], [0.00, 0.80]],
        [[0.50, 0.00], [0.00, 0.50]],
        [[0.355, -0.355], [0.355, 0.355]],
        [[0.355, 0.355], [-0.355, 0.355]]
    ])
    b = np.array([
        [0.10, 0.04],
        [0.25, 0.40],
        [0.266, 0.078],
        [0.378, 0.434]
    ])
    # 均匀概率
    probs = np.array([0.25, 0.25, 0.25, 0.25])
    points = np.zeros((n_points, 2))
    x = rng.random(2)
    # 前 100 次迭代用于"burn-in"
    for _ in range(100):
        j = rng.choice(4, p=probs)
        x = A[j] @ x + b[j]
    for i in range(n_points):
        j = rng.choice(4, p=probs)
        x = A[j] @ x + b[j]
        points[i] = x
    return points


def fractal_porosity_field(nx, ny, fractal_dim=1.8, rng=None):
    """
    生成具有分形统计特征的孔隙度扰动场。
    
    使用分数布朗运动 (fBm) 的谱方法:
      phi(k) ~ |k|^{-(2H+2)/2}
    其中 H = 3 - D 为 Hurst 指数，D 为分形维数。
    
    孔隙度场:
      phi(x) = phi_0 + delta_phi * fBm(x) / max(|fBm|)
    
    Parameters
    ----------
    nx, ny : int
        网格尺寸。
    fractal_dim : float
        目标分形维数，范围 (1, 2)。
    rng : numpy.random.Generator, optional
    
    Returns
    -------
    porosity : ndarray, shape (ny, nx)
        归一化到 [0, 1] 的孔隙度场。
    """
    if rng is None:
        rng = np.random.default_rng()
    H = 2.0 - fractal_dim  # Hurst 指数（二维情况 H = 3 - D，但这里调整使 D in (1,2)）
    # 生成随机相位
    phase = rng.random((ny, nx)) * 2.0 * np.pi
    # 构建波数网格
    kx = np.fft.fftfreq(nx, d=1.0 / nx)
    ky = np.fft.fftfreq(ny, d=1.0 / ny)
    KX, KY = np.meshgrid(kx, ky)
    k_mag = np.sqrt(KX ** 2 + KY ** 2)
    k_mag[0, 0] = 1e-10  # 避免除零
    # 功率谱: P(k) ~ k^{-(2H+1)}
    spectrum = k_mag ** (-(2.0 * H + 1.0) / 2.0)
    spectrum[0, 0] = 0.0
    # 构建复随机场
    noise = spectrum * np.exp(1j * phase)
    field = np.real(np.fft.ifft2(noise))
    # 归一化到 [0, 1]
    f_min, f_max = np.min(field), np.max(field)
    if abs(f_max - f_min) < 1e-14:
        return np.zeros((ny, nx))
    porosity = (field - f_min) / (f_max - f_min)
    return porosity
