"""
morphology_evolution.py
催化剂层形貌复杂演化与状态枚举模块

基于 mandelbrot_area (711) 与 ubvec (1370) 改造
用于分析 PEM 燃料电池催化剂层中 Pt/C 聚集体在衰减过程中的形貌演化。

核心公式:
  催化剂层中的复杂形貌可用分形维数 D_f 描述:
    N(r) ~ r^(-D_f)
    
  其中 N(r) 为覆盖形貌所需边长为 r 的盒子数。
  
  有效比表面积与分形维数关系:
    A_eff = A_0 * (L / l_0)^(D_f - 2)
    
  催化剂层孔隙网络可用迭代函数系统 (IFS) 描述，
  其连通性可用 Mandelbrot-Julia 集的概念类比。
  
  微观状态枚举使用格雷码 (Gray Code) 遍历:
    G(n) = n XOR (n >> 1)
    
  用于枚举催化剂表面吸附位点的可能状态组合。
"""

import numpy as np


def box_counting_dimension(x_coords, y_coords, n_scales=12):
    """
    使用盒计数法计算二维分形维数。
    
    公式:
        D_f = -lim_{r->0} log(N(r)) / log(r)
    
    参数:
        x_coords, y_coords: 形貌边界坐标
        n_scales: 尺度分级数
    
    返回:
        D_f: 分形维数估计
    """
    if len(x_coords) < 2 or len(y_coords) < 2:
        return 1.0
    
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    
    L = max(x_max - x_min, y_max - y_min)
    if L < 1e-15:
        return 0.0
    
    points = np.column_stack((x_coords, y_coords))
    n_pts = len(points)
    
    N_list = []
    epsilon_list = []
    
    # 从覆盖整个区域的最大盒子到单个点尺度的最小盒子
    for scale in range(1, n_scales + 1):
        epsilon = L / scale
        if epsilon < 1e-15:
            break
        
        nx = max(1, int(np.ceil((x_max - x_min) / epsilon)))
        ny = max(1, int(np.ceil((y_max - y_min) / epsilon)))
        
        occupied = set()
        for pt in points:
            ix = min(int((pt[0] - x_min) / epsilon), nx - 1)
            iy = min(int((pt[1] - y_min) / epsilon), ny - 1)
            occupied.add((ix, iy))
        
        N = len(occupied)
        if N <= 1 or N >= n_pts:
            continue
        
        N_list.append(float(N))
        epsilon_list.append(float(epsilon))
    
    if len(N_list) < 3:
        return 1.0
    
    # 转换为对数并线性回归
    log_eps = np.log(epsilon_list)
    log_N = np.log(N_list)
    
    # 使用最小二乘回归 log(N) = C - D_f * log(epsilon)
    A = np.vstack([log_eps, np.ones(len(log_eps))]).T
    coeff, residuals, _, _ = np.linalg.lstsq(A, log_N, rcond=None)
    D_f = -coeff[0]
    
    # 边界保护: 一维曲线 D_f ∈ [0.9, 1.1], 二维曲面 D_f ∈ [1.9, 2.1]
    # 这里保守地 clip 到 [0, 2]
    D_f = float(np.clip(D_f, 0.0, 2.0))
    
    return D_f


def effective_surface_area_fractal(A0, L_scale, l0, D_f):
    """
    基于分形维数计算有效比表面积。
    
    公式:
        A_eff = A0 * (L_scale / l0)^(D_f - 2)
    """
    if l0 <= 0 or L_scale <= 0 or A0 <= 0:
        return A0
    
    ratio = L_scale / l0
    if ratio <= 0:
        return A0
    
    A_eff = A0 * (ratio ** (D_f - 2.0))
    
    # 边界保护
    A_eff = np.clip(A_eff, A0 * 0.01, A0 * 100.0)
    
    return float(A_eff)


def ubvec_next_gray(t):
    """
    计算无符号二进制向量的下一个格雷码。
    
    基于 ubvec_next_gray (1370) 改造。
    
    参数:
        t: 二进制向量 (0/1 数组)
    
    返回:
        t_next: 下一个格雷码向量
    """
    t = np.array(t, dtype=int)
    n = len(t)
    
    if n <= 0:
        return t
    
    weight = np.sum(t)
    
    t_next = t.copy()
    
    if weight % 2 == 0:
        # 偶数重量: 翻转最后一位
        t_next[n - 1] = 1 - t_next[n - 1]
    else:
        # 奇数重量: 找到最右边的 1，翻转其左侧一位
        flipped = False
        for i in range(n - 1, 0, -1):
            if t[i] == 1:
                t_next[i - 1] = 1 - t_next[i - 1]
                flipped = True
                break
        
        if not flipped:
            # 最后一个元素，回到全零
            t_next[:] = 0
    
    return t_next


def enumerate_catalyst_surface_states(n_sites, max_states=1024):
    """
    枚举催化剂表面 n_sites 个吸附位点的可能状态。
    
    每个位点可以是:
      0: 空位
      1: 吸附物 (O, OH, H 等)
    
    使用格雷码遍历，保证相邻状态仅有一位不同。
    
    返回:
        states: (num_states, n_sites) 状态矩阵
    """
    if n_sites <= 0:
        return np.zeros((1, 0))
    
    total = 2 ** n_sites
    num_states = min(total, max_states)
    
    states = np.zeros((num_states, n_sites), dtype=int)
    
    t = np.zeros(n_sites, dtype=int)
    states[0, :] = t
    
    for i in range(1, num_states):
        t = ubvec_next_gray(t)
        states[i, :] = t
    
    return states


def mandelbrot_like_escape_time(c_real, c_imag, max_iter=50, escape_radius=2.0):
    """
    计算类似 Mandelbrot 集的逃逸时间。
    
    迭代公式:
        z_{n+1} = z_n^2 + c
    
    用于分析催化剂层中孔隙网络的连通性类比。
    
    参数:
        c_real, c_imag: 复平面坐标
        max_iter: 最大迭代次数
        escape_radius: 逃逸半径
    
    返回:
        iter_count: 逃逸所需迭代次数 (max_iter 表示未逃逸)
    """
    z_real = 0.0
    z_imag = 0.0
    
    for i in range(max_iter):
        zr2 = z_real * z_real
        zi2 = z_imag * z_imag
        
        if zr2 + zi2 > escape_radius * escape_radius:
            return i
        
        z_imag = 2.0 * z_real * z_imag + c_imag
        z_real = zr2 - zi2 + c_real
    
    return max_iter


def pore_network_connectivity_map(n_grid=64, max_iter=50):
    """
    生成催化剂层孔隙网络连通性图。
    
    使用 Mandelbrot 集类比: 逃逸时间短的区域对应连通性好的孔隙，
    逃逸时间长的区域对应被堵塞或孤立的孔隙。
    
    返回:
        connectivity: (n_grid, n_grid) 连通性矩阵
        x_range, y_range: 坐标范围
    """
    x_min, x_max = -2.0, 2.0
    y_min, y_max = -2.0, 2.0
    
    x = np.linspace(x_min, x_max, n_grid)
    y = np.linspace(y_min, y_max, n_grid)
    
    connectivity = np.zeros((n_grid, n_grid))
    
    for i in range(n_grid):
        for j in range(n_grid):
            c_real = x[i]
            c_imag = y[j]
            escape = mandelbrot_like_escape_time(c_real, c_imag, max_iter)
            # 连通性: 逃逸越快，连通性越差 (值越小)
            # 转换为 0-1 范围，1 表示高连通性
            connectivity[j, i] = 1.0 - float(escape) / max_iter
    
    return connectivity, (x_min, x_max), (y_min, y_max)


def morphology_degradation_index(D_f_initial, D_f_current, connectivity_drop):
    """
    计算形貌退化指数。
    
    公式:
        MDI = w_1 * (D_f_initial - D_f_current) / D_f_initial 
            + w_2 * connectivity_drop
    """
    if D_f_initial <= 0:
        return 0.0
    
    w1, w2 = 0.6, 0.4
    
    frac_loss = max(0.0, D_f_initial - D_f_current) / D_f_initial
    conn_loss = np.clip(connectivity_drop, 0.0, 1.0)
    
    mdi = w1 * frac_loss + w2 * conn_loss
    return float(np.clip(mdi, 0.0, 1.0))


if __name__ == "__main__":
    # 生成圆形边界测试分形维数
    theta = np.linspace(0, 2*np.pi, 100)
    x = np.cos(theta)
    y = np.sin(theta)
    D_f = box_counting_dimension(x, y)
    print(f"圆的分形维数: {D_f:.4f} (理论值: 1.0)")
    
    states = enumerate_catalyst_surface_states(4, max_states=16)
    print(f"枚举 {len(states)} 个表面状态 (4位点)")
    
    conn, xr, yr = pore_network_connectivity_map(n_grid=32, max_iter=30)
    print(f"孔隙连通性均值: {np.mean(conn):.4f}")
