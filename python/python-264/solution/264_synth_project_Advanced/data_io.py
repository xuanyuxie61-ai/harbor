# -*- coding: utf-8 -*-
"""
data_io.py
==========

相空间数据 I/O 模块.

本模块处理辐射带模拟数据的读写:
  - XYZ 点云格式 (来自 xyz_io 项目)
  - 校验和验证 (来自 bank 项目)
  - 二进制网格数据

物理背景:
  辐射带模拟输出 (L, E, f) 三维数据需要高效存储和传输.
  使用 XYZ 格式便于后处理, 使用校验和确保数据完整性.

参考文献:
  [1] Burkardt, J., "XYZ File Format", Virginia Tech (2007)
"""

import numpy as np
import os
import hashlib
import physical_constants as pc


# =============================================================================
#  校验和计算 (来自 bank 项目)
# =============================================================================

def compute_checksum(data):
    """
    计算数据的校验和.

    使用 MD5 哈希算法生成 128 位校验和.

    参数
    ----
    data : ndarray 或 bytes
        数据

    返回
    -------
    checksum : str
        十六进制校验和
    """
    if isinstance(data, np.ndarray):
        data_bytes = data.tobytes()
    elif isinstance(data, bytes):
        data_bytes = data
    else:
        data_bytes = str(data).encode('utf-8')

    return hashlib.md5(data_bytes).hexdigest()


def verify_checksum(data, expected_checksum):
    """
    验证数据校验和.

    参数
    ----
    data : ndarray 或 bytes
        数据
    expected_checksum : str
        期望的校验和

    返回
    -------
    valid : bool
        是否匹配
    """
    actual = compute_checksum(data)
    return actual == expected_checksum


def luhn_checksum(number_str):
    """
    Luhn 算法校验 (来自 bank 项目).

    用于验证数值数据的完整性.

    参数
    ----
    number_str : str
        数字字符串

    返回
    -------
    valid : bool
        是否通过校验
    """
    digits = [int(d) for d in str(number_str) if d.isdigit()]
    if not digits:
        return False

    # Luhn 算法
    total = 0
    reverse = digits[::-1]
    for i, d in enumerate(reverse):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d

    return total % 10 == 0


# =============================================================================
#  XYZ 点云格式 (来自 xyz_io 项目)
# =============================================================================

def write_xyz(filename, L, E, f, comment=""):
    """
    写入 XYZ 格式文件.

    XYZ 格式:
      Line 1: N (点数)
      Line 2: comment
      Lines 3+: x y z (L E f)

    参数
    ----
    filename : str
        文件名
    L, E, f : ndarray
        数据数组
    comment : str
        注释
    """
    n_points = len(L) * len(E) if f.ndim == 2 else len(L)

    with open(filename, 'w') as fout:
        fout.write(f"{n_points}\n")
        fout.write(f"{comment}\n")

        if f.ndim == 2:
            for i, Li in enumerate(L):
                for j, Ej in enumerate(E):
                    fout.write(f"{Li:.6f} {Ej:.6f} {f[i,j]:.6e}\n")
        else:
            for i in range(len(L)):
                fout.write(f"{L[i]:.6f} {E[i]:.6f} {f[i]:.6e}\n")


def read_xyz(filename):
    """
    读取 XYZ 格式文件.

    参数
    ----
    filename : str
        文件名

    返回
    -------
    data : ndarray, shape (N, 3)
        点云数据 (L, E, f)
    comment : str
        注释
    """
    with open(filename, 'r') as fin:
        n_points = int(fin.readline().strip())
        comment = fin.readline().strip()

        data = []
        for line in fin:
            parts = line.strip().split()
            if len(parts) >= 3:
                data.append([float(p) for p in parts[:3]])

    return np.array(data), comment


# =============================================================================
#  二进制网格数据
# =============================================================================

def save_grid_data(filename, L, E, f, metadata=None):
    """
    保存网格数据为二进制格式.

    格式:
      - 头部: L_min, L_max, n_L, E_min, E_max, n_E
      - 数据: f 数组 (float64)
      - 校验和: 16 字节 MD5

    参数
    ----
    filename : str
        文件名
    L, E : ndarray
        坐标数组
    f : ndarray
        分布函数
    metadata : dict, optional
        元数据
    """
    header = np.array([
        L[0], L[-1], len(L),
        E[0], E[-1], len(E)
    ], dtype=np.float64)

    checksum = compute_checksum(f)

    with open(filename, 'wb') as fout:
        # 写入头部
        header.tofile(fout)
        # 写入数据
        f.astype(np.float64).tofile(fout)
        # 写入校验和
        fout.write(checksum.encode('utf-8'))


def load_grid_data(filename):
    """
    加载网格数据.

    参数
    ----
    filename : str
        文件名

    返回
    -------
    L, E : ndarray
        坐标数组
    f : ndarray
        分布函数
    checksum_valid : bool
        校验和是否有效
    """
    with open(filename, 'rb') as fin:
        # 读取头部
        header = np.fromfile(fin, dtype=np.float64, count=6)
        L_min, L_max, n_L, E_min, E_max, n_E = header
        n_L, n_E = int(n_L), int(n_E)

        # 重建坐标
        L = np.linspace(L_min, L_max, n_L)
        E = np.linspace(E_min, E_max, n_E)

        # 读取数据
        f = np.fromfile(fin, dtype=np.float64, count=n_L * n_E)
        f = f.reshape((n_L, n_E))

        # 读取校验和
        stored_checksum = fin.read().decode('utf-8')

    # 验证
    checksum_valid = verify_checksum(f, stored_checksum)

    return L, E, f, checksum_valid


# =============================================================================
#  模拟结果导出
# =============================================================================

def export_simulation_results(results, grid, output_dir='.'):
    """
    导出模拟结果.

    参数
    ----
    results : dict
        模拟结果
    grid : MagnetosphereGrid
        相空间网格
    output_dir : str
        输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    # 保存最终分布
    f_final = results.get('f_final')
    if f_final is not None:
        filename = os.path.join(output_dir, 'distribution_final.xyz')
        write_xyz(filename, grid.L, grid.E_MeV, f_final,
                  comment="Final distribution function")

        # 二进制格式
        filename_bin = os.path.join(output_dir, 'distribution_final.bin')
        save_grid_data(filename_bin, grid.L, grid.E_MeV, f_final)

    # 保存历史
    history = results.get('history', [])
    if history:
        filename_hist = os.path.join(output_dir, 'history.txt')
        with open(filename_hist, 'w') as fout:
            fout.write("# time[s] step total_particles max_f min_f L_peak\n")
            for h in history:
                fout.write(f"{h.get('time', 0):.4e} "
                           f"{h.get('step', 0)} "
                           f"{h.get('total_particles', 0):.4e} "
                           f"{h.get('max_f', 0):.4e} "
                           f"{h.get('min_f', 0):.4e} "
                           f"{h.get('L_peak', 0):.4e}\n")

    print(f"结果已导出到: {output_dir}")


if __name__ == "__main__":
    print("数据 I/O 模块自检验证")

    # 校验和测试
    data = np.array([1.0, 2.0, 3.0])
    checksum = compute_checksum(data)
    print(f"  校验和: {checksum}")
    print(f"  验证: {verify_checksum(data, checksum)}")

    # Luhn 校验
    print(f"  Luhn(79927398713): {luhn_checksum('79927398713')}")

    # XYZ 测试
    L = np.array([1.0, 2.0, 3.0])
    E = np.array([0.5, 1.0, 2.0])
    f = np.ones((3, 3))
    write_xyz('test.xyz', L, E, f, comment="Test")
    data, comment = read_xyz('test.xyz')
    print(f"  XYZ 读写: {len(data)} 点, 注释: {comment}")
    os.remove('test.xyz')
