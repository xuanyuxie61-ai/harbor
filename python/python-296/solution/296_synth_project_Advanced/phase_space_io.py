# -*- coding: utf-8 -*-
"""
phase_space_io.py
=================
相空间数据 I/O (XYZ 格式扩展).

核心算法 (来自 1424_xyz_io):
----------------------------
XYZ 文件格式:
    第 1 行: 点数 N
    第 2 行: 注释/标题
    第 3~N+2 行: x y z 数据

扩展到等离子体相空间:
    第 1 行: 电子数 N
    第 2 行: 物理参数注释
    第 3~N+2 行: x[m]  y[m]  E[eV]  vx[m/s]  vy[m/s]  vz[m/s]

核心来源 (种子项目映射):
- 1424_xyz_io: XYZ 文件读写
"""

import os
import math


# ============================================================
# XYZ 文件头 (来自 1424_xyz_io/xyz_header_read/write)
# ============================================================
def write_xyz_header(f, n_points, comment=""):
    """
    写入 XYZ 文件头 (来自 1424_xyz_io/xyz_header_write).

    参数:
        f        : 文件对象
        n_points : 点数
        comment  : 注释行
    """
    f.write("{}\n".format(n_points))
    f.write("{}\n".format(comment if comment else "Phase space data"))


def read_xyz_header(f):
    """
    读取 XYZ 文件头 (来自 1424_xyz_io/xyz_header_read).

    返回:
        (n_points, comment)
    """
    line1 = f.readline().strip()
    n_points = int(line1) if line1 else 0
    comment = f.readline().strip()
    return n_points, comment


# ============================================================
# 相空间数据写入 (来自 1424_xyz_io/xyz_data_write)
# ============================================================
def write_phase_space_xyz(filename, particles, comment=""):
    """
    将相空间数据写入 XYZ 格式文件.

    每行格式: x[m]  y[m]  E[eV]  vx[m/s]  vy[m/s]  vz[m/s]

    参数:
        filename : 输出文件名
        particles: 粒子列表 (dict)
        comment  : 注释
    """
    n = len(particles)
    with open(filename, 'w') as f:
        write_xyz_header(f, n, comment)
        for p in particles:
            x = p.get("x", 0.0)
            y = p.get("y", 0.0)
            E = p.get("energy_ev", 0.0)
            vx = p.get("vx", 0.0)
            vy = p.get("vy", 0.0)
            vz = p.get("vz", 0.0)
            f.write("{:16.8e}  {:16.8e}  {:16.8e}  "
                    "{:16.8e}  {:16.8e}  {:16.8e}\n".format(
                        x, y, E, vx, vy, vz))


def read_phase_space_xyz(filename):
    """
    读取 XYZ 格式相空间数据.

    返回:
        list[dict]: 粒子列表
    """
    particles = []
    with open(filename, 'r') as f:
        n_points, comment = read_xyz_header(f)
        for _ in range(n_points):
            line = f.readline().strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 6:
                particles.append({
                    "x": float(parts[0]),
                    "y": float(parts[1]),
                    "energy_ev": float(parts[2]),
                    "vx": float(parts[3]),
                    "vy": float(parts[4]),
                    "vz": float(parts[5]),
                })
    return particles, comment


# ============================================================
# 能量沉积场 I/O
# ============================================================
def write_energy_density_field(filename, u_2d, x_1d, y_1d, t):
    """
    写入二维能量密度场.

    格式: 文本文件, 每行 x y u(x,y)
    """
    ny = len(u_2d)
    nx = len(u_2d[0]) if ny > 0 else 0
    with open(filename, 'w') as f:
        f.write("# Energy density field at t={:.4e} s\n".format(t))
        f.write("# nx={} ny={}\n".format(nx, ny))
        f.write("# x[m]  y[m]  u[J/m^3]\n")
        for j in range(ny):
            for i in range(nx):
                f.write("{:16.8e}  {:16.8e}  {:16.8e}\n".format(
                    x_1d[i], y_1d[j], u_2d[j][i]))


def write_1d_profile(filename, x, u, label="u"):
    """写入一维剖面."""
    with open(filename, 'w') as f:
        f.write("# x  {}\n".format(label))
        for i in range(len(x)):
            f.write("{:16.8e}  {:16.8e}\n".format(x[i], u[i]))


def write_simulation_summary(filename, summary_dict):
    """写入模拟摘要 (键值对格式)."""
    with open(filename, 'w') as f:
        for key, value in summary_dict.items():
            if isinstance(value, float):
                f.write("{} = {:.8e}\n".format(key, value))
            else:
                f.write("{} = {}\n".format(key, value))


def print_io_summary(filename, n_written=None, n_read=None):
    """打印 I/O 摘要."""
    print("\n  相空间 I/O (XYZ 格式):")
    if n_written is not None:
        print("    写入文件  : {}".format(filename))
        print("    写入点数  : {}".format(n_written))
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print("    文件大小  : {} bytes".format(size))
    if n_read is not None:
        print("    读取点数  : {}".format(n_read))
