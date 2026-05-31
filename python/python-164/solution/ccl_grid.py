"""
ccl_grid.py
催化剂层多维参数网格生成模块

基于 hypercube_grid (558) 改造
用于生成 PEM 燃料电池阴极催化剂层 (CCL) 的多维物理参数采样网格。

核心公式:
  - 网格点坐标: x_i^{(k)} = ((s_k - j) * a_k + (j - 1) * b_k) / (s_k - 1)
  - 参数空间维度 M 包括: 温度 T, 相对湿度 RH, 电池电位 E, Pt 负载 L_pt, 碳比表面积 S_c
"""

import numpy as np


def hypercube_grid(m, n, ns, a, b, c):
    """
    在 M 维超立方体内部生成规则网格点。
    
    参数:
        m: 空间维度数
        n: 总网格点数，n = prod(ns)
        ns: 各维度上的点数列表，长度 m
        a: 各维度下限列表
        b: 各维度上限列表
        c: 各维度网格中心模式 (1-5)
    
    返回:
        x: (m, n) 数组，网格点坐标
    """
    if m <= 0:
        raise ValueError("维度 m 必须为正整数")
    if len(ns) != m or len(a) != m or len(b) != m or len(c) != m:
        raise ValueError("ns, a, b, c 长度必须与 m 一致")
    if n != int(np.prod(ns)):
        raise ValueError(f"n={n} 必须等于 ns 的乘积 {int(np.prod(ns))}")
    
    x = np.zeros((m, n))
    
    for i in range(m):
        s = ns[i]
        xs = np.zeros(s)
        
        for j in range(s):
            if c[i] == 1:
                if s == 1:
                    xs[j] = 0.5 * (a[i] + b[i])
                else:
                    xs[j] = ((s - j - 1) * a[i] + j * b[i]) / (s - 1)
            elif c[i] == 2:
                xs[j] = ((s - j) * a[i] + (j + 1) * b[i]) / (s + 1)
            elif c[i] == 3:
                xs[j] = ((s - j) * a[i] + j * b[i]) / s
            elif c[i] == 4:
                xs[j] = ((s - j - 1) * a[i] + (j + 1) * b[i]) / s
            elif c[i] == 5:
                xs[j] = ((2 * s - 2 * j - 1) * a[i] + (2 * j + 1) * b[i]) / (2 * s)
            else:
                raise ValueError(f"中心模式 c[{i}]={c[i]} 必须在 1-5 之间")
        
        x = r8vec_direct_product(i, s, xs, m, n, x)
    
    return x


def r8vec_direct_product(factor_index, factor_order, factor_value, m, n, x):
    """
    将第 factor_index 维的一维网格 factor_value 与现有网格 x 做直积。
    """
    if factor_index < 0 or factor_index >= m:
        raise ValueError("factor_index 超出范围")
    if factor_order < 1:
        raise ValueError("factor_order 必须 >= 1")
    
    # 计算各维度的重复块大小
    rep = 1
    for i in range(factor_index + 1, m):
        rep *= int(x.shape[1] / (np.prod([1] + [int(x.shape[1])])))
    # 更安全的计算
    if factor_index + 1 < m:
        # 这里不需要复杂计算，因为x已经被逐维填充
        pass
    
    # 简化的实现：直接填充对应行
    skip = n // factor_order
    if skip * factor_order != n:
        # 对于渐进式构建，需要特殊处理
        # 使用已有网格中的模式
        pass
    
    # 更可靠的方式：根据已有x中非零列数推断当前状态
    # 但这里我们直接采用标准直积算法
    contig = 1
    for i in range(factor_index):
        # 查找前factor_index维已经有多少离散点
        # 从x已有数据推断
        nonzero_cols = np.where(np.any(x != 0, axis=0))[0]
        if len(nonzero_cols) > 0:
            # 已有部分填充
            pass
    
    # 标准实现：对于每一列，根据列索引计算该维度的取值
    for j in range(n):
        # 计算当前列在factor_index维上的索引
        block = 1
        for k in range(factor_index + 1, m):
            # 无法预知后续维度，但可以用总点数的已知部分来推断
            pass
        
        # 采用更直接的递归方式：根据当前j计算该维度索引
        # 这里假设ns的前factor_index维的乘积为prev, 本维factor_order, 后续为post
        # 但x的构造过程中各维是顺序添加的
        # 简化：使用重复模式
        if factor_index == 0:
            idx = j % factor_order
        else:
            # 统计之前已确定维度的点数
            prev_points = 1
            for k in range(factor_index):
                # 从前面的x行中非零唯一值数量推断
                unique_vals = np.unique(x[k, :])
                unique_vals = unique_vals[unique_vals != 0] if len(unique_vals) > 1 else unique_vals
                prev_points = max(prev_points, len(unique_vals))
            idx = (j // prev_points) % factor_order
        
        x[factor_index, j] = factor_value[idx]
    
    return x


def generate_ccl_parameter_grid():
    """
    生成 CCL 典型操作参数网格。
    
    物理参数空间:
      - T    : 操作温度 [K],      范围 [333.15, 353.15]
      - RH   : 相对湿度 [%],       范围 [60, 100]
      - E    : 电池电位 [V vs. RHE], 范围 [0.6, 1.0]
      - L_pt : Pt 负载 [mg/cm^2],  范围 [0.05, 0.4]
      - S_c  : 碳载体比表面积 [m^2/g], 范围 [100, 400]
    
    返回:
        params: 字典，包含网格点和参数范围
    """
    m = 5
    ns = [3, 3, 4, 3, 3]
    n = int(np.prod(ns))
    
    # 参数范围
    a = np.array([333.15, 60.0, 0.60, 0.05, 100.0])
    b = np.array([353.15, 100.0, 1.00, 0.40, 400.0])
    c = np.array([1, 1, 1, 1, 1])  # 均匀网格
    
    x = hypercube_grid(m, n, ns, a, b, c)
    
    param_names = ['temperature_K', 'relative_humidity_pct', 
                   'cell_potential_V', 'pt_loading_mg_cm2', 'carbon_surface_area_m2_g']
    
    params = {
        'grid': x,
        'names': param_names,
        'ns': ns,
        'ranges': list(zip(a, b)),
        'num_points': n
    }
    
    return params


def sample_operating_condition(grid_data, index):
    """
    从参数网格中提取单个操作条件。
    """
    if index < 0 or index >= grid_data['num_points']:
        raise IndexError("操作条件索引越界")
    
    point = grid_data['grid'][:, index]
    condition = {}
    for i, name in enumerate(grid_data['names']):
        condition[name] = point[i]
    
    return condition


if __name__ == "__main__":
    params = generate_ccl_parameter_grid()
    print(f"生成 CCL 参数网格: {params['num_points']} 个采样点")
    cond = sample_operating_condition(params, 0)
    print("首个操作条件:", cond)
