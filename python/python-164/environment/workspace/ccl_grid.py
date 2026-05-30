
import numpy as np


def hypercube_grid(m, n, ns, a, b, c):
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
    if factor_index < 0 or factor_index >= m:
        raise ValueError("factor_index 超出范围")
    if factor_order < 1:
        raise ValueError("factor_order 必须 >= 1")
    

    rep = 1
    for i in range(factor_index + 1, m):
        rep *= int(x.shape[1] / (np.prod([1] + [int(x.shape[1])])))

    if factor_index + 1 < m:

        pass
    

    skip = n // factor_order
    if skip * factor_order != n:


        pass
    


    contig = 1
    for i in range(factor_index):


        nonzero_cols = np.where(np.any(x != 0, axis=0))[0]
        if len(nonzero_cols) > 0:

            pass
    

    for j in range(n):

        block = 1
        for k in range(factor_index + 1, m):

            pass
        




        if factor_index == 0:
            idx = j % factor_order
        else:

            prev_points = 1
            for k in range(factor_index):

                unique_vals = np.unique(x[k, :])
                unique_vals = unique_vals[unique_vals != 0] if len(unique_vals) > 1 else unique_vals
                prev_points = max(prev_points, len(unique_vals))
            idx = (j // prev_points) % factor_order
        
        x[factor_index, j] = factor_value[idx]
    
    return x


def generate_ccl_parameter_grid():
    m = 5
    ns = [3, 3, 4, 3, 3]
    n = int(np.prod(ns))
    

    a = np.array([333.15, 60.0, 0.60, 0.05, 100.0])
    b = np.array([353.15, 100.0, 1.00, 0.40, 400.0])
    c = np.array([1, 1, 1, 1, 1])
    
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
