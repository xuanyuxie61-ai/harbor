#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_io.py
数据输入输出与网格格式转换

融合种子项目：
  - 425_ffmatlib (ffreaddata): 多维 ASCII 数据读取
  - 1322_triangle_to_xml: 网格格式转换
  - 1320_triangle_to_fem: 网格数据到 FEM 格式输出

核心功能：
  1. MT 观测数据读写（频率、阻抗、视电阻率、相位）
  2. 模型参数文件读写
  3. 网格节点/单元数据文件读写
  4. 反演结果报告生成
  5. 数据格式转换与校验
"""

import numpy as np
import os


class MTDataReader:
    """
    MT 观测数据读取器

    支持格式：
      - 简单 ASCII 列格式：频率  阻抗实部  阻抗虚部  误差
      - 扩展格式：频率  Z_xy_r  Z_xy_i  Z_yx_r  Z_yx_i  rho_xy  rho_yx  phi_xy  phi_yx
    """

    @staticmethod
    def read_simple(filename):
        """读取简单格式数据"""
        if not os.path.exists(filename):
            raise FileNotFoundError(f"文件不存在: {filename}")
        data = np.loadtxt(filename, dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return data

    @staticmethod
    def read_complex_impedance(filename):
        """
        读取复阻抗数据

        文件格式（每行）：
            freq_hz  Z_real  Z_imag  error
        """
        data = MTDataReader.read_simple(filename)
        frequencies = data[:, 0]
        Z = data[:, 1] + 1j * data[:, 2]
        errors = data[:, 3] if data.shape[1] > 3 else np.ones(len(frequencies)) * 0.05
        return frequencies, Z, errors

    @staticmethod
    def read_full_mt_data(filename):
        """
        读取完整 MT 数据

        文件格式（每行）：
            freq  Zxy_r  Zxy_i  Zyx_r  Zyx_i  rho_xy  rho_yx  phi_xy  phi_yx  err
        """
        data = MTDataReader.read_simple(filename)
        result = {
            'frequencies': data[:, 0],
            'Zxy': data[:, 1] + 1j * data[:, 2],
            'Zyx': data[:, 3] + 1j * data[:, 4],
            'rho_xy': data[:, 5],
            'rho_yx': data[:, 6],
            'phi_xy': data[:, 7],
            'phi_yx': data[:, 8],
            'errors': data[:, 9] if data.shape[1] > 9 else np.ones(len(data)) * 0.05
        }
        return result


class MTDataWriter:
    """MT 数据写入器"""

    @staticmethod
    def write_simple(filename, frequencies, values, header=""):
        """写入简单两列数据"""
        data = np.column_stack([frequencies, values])
        np.savetxt(filename, data, fmt='%.6e', header=header, comments='# ')

    @staticmethod
    def write_complex_impedance(filename, frequencies, Z, errors=None):
        """写入复阻抗数据"""
        if errors is None:
            errors = np.ones(len(frequencies)) * 0.05
        data = np.column_stack([
            frequencies,
            np.real(Z),
            np.imag(Z),
            errors
        ])
        header = "# freq_hz  Z_real  Z_imag  error"
        np.savetxt(filename, data, fmt='%.6e', header=header, comments='')

    @staticmethod
    def write_full_mt_data(filename, frequencies, Zxy, Zyx,
                           rho_xy, rho_yx, phi_xy, phi_yx, errors=None):
        """写入完整 MT 数据"""
        if errors is None:
            errors = np.ones(len(frequencies)) * 0.05
        data = np.column_stack([
            frequencies,
            np.real(Zxy), np.imag(Zxy),
            np.real(Zyx), np.imag(Zyx),
            rho_xy, rho_yx,
            phi_xy, phi_yx,
            errors
        ])
        header = ("# freq  Zxy_r  Zxy_i  Zyx_r  Zyx_i  "
                  "rho_xy  rho_yx  phi_xy  phi_yx  error")
        np.savetxt(filename, data, fmt='%.6e', header=header, comments='')


class MeshDataIO:
    """
    网格数据 I/O（融合 1320_triangle_to_fem 和 1322_triangle_to_xml 思想）
    """

    @staticmethod
    def write_nodes(filename, points):
        """
        写入节点坐标文件

        格式（基于 triangle_to_fem 的 r8mat_write）：
            每行一个节点：x y [z]
        """
        points = np.asarray(points, dtype=np.float64)
        np.savetxt(filename, points, fmt='%g')

    @staticmethod
    def read_nodes(filename):
        """读取节点坐标"""
        return np.loadtxt(filename, dtype=np.float64)

    @staticmethod
    def write_elements(filename, triangles):
        """
        写入单元连接性文件

        格式（基于 triangle_to_fem 的 i4mat_write）：
            每行一个单元：node1 node2 node3
        """
        triangles = np.asarray(triangles, dtype=np.int32)
        np.savetxt(filename, triangles, fmt='%d')

    @staticmethod
    def read_elements(filename):
        """读取单元连接性"""
        return np.loadtxt(filename, dtype=np.int32)

    @staticmethod
    def write_xml_mesh(filename, points, triangles):
        """
        写入 DOLFIN XML 网格文件

        融合 1322_triangle_to_xml 的 xml_mesh2d_write 核心逻辑。
        """
        points = np.asarray(points, dtype=np.float64)
        triangles = np.asarray(triangles, dtype=np.int32)

        # 确保 0-based 索引
        tri_min = np.min(triangles)
        if tri_min == 1:
            triangles = triangles - 1

        n_points = len(points)
        n_triangles = len(triangles)
        dim = points.shape[1]

        with open(filename, 'w') as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<dolfin xmlns:dolfin="http://www.fenics.org/dolfin/">\n')
            f.write(f'  <mesh celltype="triangle" dim="{dim}">\n')
            f.write(f'    <vertices size="{n_points}">\n')
            for i, p in enumerate(points):
                if dim == 2:
                    f.write(f'      <vertex index ="{i}" x ="{p[0]:g}" y ="{p[1]:g}"/>\n')
                else:
                    f.write(f'      <vertex index ="{i}" x ="{p[0]:g}" y ="{p[1]:g}" z ="{p[2]:g}"/>\n')
            f.write('    </vertices>\n')
            f.write(f'    <cells size="{n_triangles}">\n')
            for i, tri in enumerate(triangles):
                f.write(f'      <triangle index ="{i}" v0 ="{tri[0]}" v1 ="{tri[1]}" v2 ="{tri[2]}"/>\n')
            f.write('    </cells>\n')
            f.write('  </mesh>\n')
            f.write('</dolfin>\n')

    @staticmethod
    def write_model_report(filename, model, predictions, residuals,
                           inversion_stats, frequencies):
        """
        生成反演结果报告

        Parameters
        ----------
        filename : str
            输出文件名
        model : dict
            模型参数
        predictions : dict
            预测数据
        residuals : dict
            残差
        inversion_stats : dict
            反演统计信息
        frequencies : ndarray
            频率列表
        """
        with open(filename, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("大地电磁测深反演结果报告\n")
            f.write("=" * 70 + "\n\n")

            f.write("【模型参数】\n")
            for key, val in model.items():
                f.write(f"  {key}: {val}\n")
            f.write("\n")

            f.write("【反演统计】\n")
            for key, val in inversion_stats.items():
                f.write(f"  {key}: {val}\n")
            f.write("\n")

            f.write("【频率响应对比】\n")
            f.write(f"{'频率(Hz)':>12} {'预测ρ_a':>14} {'观测ρ_a':>14} {'残差':>14} {'相位(°)':>10}\n")
            for i in range(len(frequencies)):
                pred_rho = predictions.get('rho_a', [0])[i] if i < len(predictions.get('rho_a', [])) else 0
                obs_rho = residuals.get('obs_rho_a', [0])[i] if i < len(residuals.get('obs_rho_a', [])) else 0
                res = residuals.get('rho_a', [0])[i] if i < len(residuals.get('rho_a', [])) else 0
                phi = predictions.get('phi', [0])[i] if i < len(predictions.get('phi', [])) else 0
                f.write(f"{frequencies[i]:>12.4e} {pred_rho:>14.4f} {obs_rho:>14.4f} "
                        f"{res:>14.4f} {phi:>10.2f}\n")
            f.write("\n" + "=" * 70 + "\n")


class DataValidator:
    """数据验证器"""

    @staticmethod
    def validate_mt_data(frequencies, rho_a, phi):
        """验证 MT 数据的物理合理性"""
        issues = []

        if np.any(frequencies <= 0):
            issues.append("频率必须为正")

        if np.any(rho_a <= 0):
            issues.append("视电阻率必须为正")

        if np.any(np.abs(phi) > 90.0):
            issues.append("相位绝对值不应超过90度")

        if len(frequencies) < 2:
            issues.append("至少需要两个频率点")

        # 检查阻抗相位与电阻率的一致性
        # 在低频区，相位应接近 45°；高频区可能偏离
        if np.mean(phi[:max(1, len(phi)//4)]) < 30.0:
            issues.append("低频相位偏低，可能存在近场效应或数据质量问题")

        return len(issues) == 0, issues

    @staticmethod
    def validate_model(resistivities, thicknesses):
        """验证层状模型的物理合理性"""
        issues = []

        if np.any(resistivities <= 0):
            issues.append("电阻率必须为正")

        if len(thicknesses) > 0 and np.any(thicknesses <= 0):
            issues.append("厚度必须为正")

        if np.any(resistivities > 1e6):
            issues.append("存在极高电阻率，可能为绝缘体")

        if np.any(resistivities < 1e-3):
            issues.append("存在极低电阻率，可能为良导体")

        return len(issues) == 0, issues


if __name__ == "__main__":
    # 自检
    freqs = np.logspace(-2, 2, 10)
    Z = np.random.randn(10) + 1j * np.random.randn(10)
    MTDataWriter.write_complex_impedance("test_mt_data.txt", freqs, Z)
    f2, Z2, err2 = MTDataReader.read_complex_impedance("test_mt_data.txt")
    print(f"数据读写测试: 误差 = {np.max(np.abs(Z - Z2)):.2e}")
    os.remove("test_mt_data.txt")

    points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float64)
    tris = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    MeshDataIO.write_xml_mesh("test_mesh.xml", points, tris)
    print("XML网格写入成功")
    os.remove("test_mesh.xml")

    ok, issues = DataValidator.validate_mt_data(freqs, np.abs(Z)**2, np.angle(Z, deg=True))
    print(f"数据验证: {'通过' if ok else '未通过'}, {issues}")
