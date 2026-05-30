# -*- coding: utf-8 -*-

import numpy as np
import os


def write_xy_data(filename, x, y, header="X Y"):
    data = np.column_stack((x, y))
    np.savetxt(filename, data, fmt='%.10e', header=header, comments='# ')


def read_xy_data(filename):
    try:
        data = np.loadtxt(filename, comments='#')
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return data[:, 0], data[:, 1]
    except Exception as e:
        print(f"读取 {filename} 失败: {e}")
        return None, None


def write_vtk_structured_grid(filename, x, y, z, scalars=None, vectors=None):
    nx, ny, nz = x.shape
    n_points = nx * ny * nz

    with open(filename, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Hypersonic Boundary Layer Data\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_GRID\n")
        f.write(f"DIMENSIONS {nx} {ny} {nz}\n")
        f.write(f"POINTS {n_points} float\n")

        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    f.write(f"{x[i,j,k]:.6e} {y[i,j,k]:.6e} {z[i,j,k]:.6e}\n")

        f.write(f"\nPOINT_DATA {n_points}\n")

        if scalars:
            for name, arr in scalars.items():
                f.write(f"SCALARS {name} float 1\n")
                f.write("LOOKUP_TABLE default\n")
                for k in range(nz):
                    for j in range(ny):
                        for i in range(nx):
                            f.write(f"{arr[i,j,k]:.6e}\n")

        if vectors:
            for name, (u, v, w) in vectors.items():
                f.write(f"VECTORS {name} float\n")
                for k in range(nz):
                    for j in range(ny):
                        for i in range(nx):
                            f.write(f"{u[i,j,k]:.6e} {v[i,j,k]:.6e} {w[i,j,k]:.6e}\n")


def write_tecplot_zone(filename, title, x, y, scalars):
    n = len(x)
    var_names = ['X', 'Y'] + list(scalars.keys())
    with open(filename, 'w') as f:
        f.write(f'TITLE = "{title}"\n')
        f.write(f'VARIABLES = "' + '", "'.join(var_names) + '"\n')
        f.write(f'ZONE I={n}, F=POINT\n')
        for i in range(n):
            line = f"{x[i]:.10e} {y[i]:.10e}"
            for name in scalars.keys():
                line += f" {scalars[name][i]:.10e}"
            f.write(line + "\n")


def read_tecplot_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    title = ""
    variables = []
    data_start = 0
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if line_stripped.upper().startswith('TITLE'):
            title = line_stripped.split('=')[1].strip().strip('"')
        elif line_stripped.upper().startswith('VARIABLES'):
            parts = line_stripped.split('=')[1].strip()
            variables = [v.strip().strip('"\'') for v in parts.split(',')]
        elif line_stripped.upper().startswith('ZONE'):
            data_start = i + 1
            break

    data = []
    for line in lines[data_start:]:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        try:
            vals = [float(v) for v in line.split()]
            data.append(vals)
        except ValueError:
            continue

    data = np.array(data)
    return {'title': title, 'variables': variables, 'data': data}


def write_eigenvalue_spectrum(filename, alphas, omegas, labels=None):
    with open(filename, 'w') as f:
        f.write("# Alpha   Omega_r   Omega_i   Label\n")
        for i in range(len(alphas)):
            label = labels[i] if labels else f"mode_{i}"
            f.write(f"{alphas[i]:.8e} {omegas[i].real:.8e} {omegas[i].imag:.8e} {label}\n")


def write_transition_report(filename, results):
    with open(filename, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("  高超声速边界层转捩预测计算报告\n")
        f.write("=" * 70 + "\n\n")

        f.write("【基流参数】\n")
        for key in ['Ma', 'Re', 'Pr', 'Tw_Te']:
            if key in results:
                f.write(f"  {key} = {results[key]}\n")
        f.write("\n")

        if 'thermal' in results:
            th = results['thermal']
            f.write("【热场求解】\n")
            f.write(f"  迭代次数: {th.get('iterations', 'N/A')}\n")
            f.write(f"  收敛残差: {th.get('diff', 'N/A'):.4e}\n")
            f.write(f"  壁面温度: {th.get('T', [None])[0]:.6f}\n")
            f.write("\n")

        if 'stability' in results:
            st = results['stability']
            f.write("【稳定性分析】\n")
            f.write(f"  最大时间增长率: {st.get('max_temporal_growth_rate', 'N/A'):.6e}\n")
            f.write(f"  模态矩阵条件数: {st.get('condition_number', 'N/A'):.4e}\n")
            f.write(f"  最大 Jordan 块: {st.get('max_jordan_block', 'N/A')}\n")
            f.write("\n")

        if 'transition' in results:
            tr = results['transition']
            f.write("【转捩预测】\n")
            f.write(f"  转捩雷诺数 Re_xt: {tr.get('mean_Re_xt', 'N/A'):.4e}\n")
            f.write(f"  标准差: {tr.get('std_Re_xt', 'N/A'):.4e}\n")
            f.write(f"  转捩前沿光滑性: {tr.get('smoothness', 'N/A'):.4e}\n")
            f.write("\n")

        if 'monte_carlo' in results:
            mc = results['monte_carlo']
            f.write("【不确定性量化】\n")
            f.write(f"  转捩雷诺数均值: {mc.get('mean', 'N/A'):.4e}\n")
            f.write(f"  标准差: {mc.get('std', 'N/A'):.4e}\n")
            ci = mc.get('ci95', (np.nan, np.nan))
            f.write(f"  95% 置信区间: [{ci[0]:.4e}, {ci[1]:.4e}]\n")
            f.write("\n")

        f.write("=" * 70 + "\n")
        f.write("  计算完成\n")
        f.write("=" * 70 + "\n")
