#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os


class MTDataReader:

    @staticmethod
    def read_simple(filename):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"文件不存在: {filename}")
        data = np.loadtxt(filename, dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return data

    @staticmethod
    def read_complex_impedance(filename):
        data = MTDataReader.read_simple(filename)
        frequencies = data[:, 0]
        Z = data[:, 1] + 1j * data[:, 2]
        errors = data[:, 3] if data.shape[1] > 3 else np.ones(len(frequencies)) * 0.05
        return frequencies, Z, errors

    @staticmethod
    def read_full_mt_data(filename):
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

    @staticmethod
    def write_simple(filename, frequencies, values, header=""):
        data = np.column_stack([frequencies, values])
        np.savetxt(filename, data, fmt='%.6e', header=header, comments='# ')

    @staticmethod
    def write_complex_impedance(filename, frequencies, Z, errors=None):
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

    @staticmethod
    def write_nodes(filename, points):
        points = np.asarray(points, dtype=np.float64)
        np.savetxt(filename, points, fmt='%g')

    @staticmethod
    def read_nodes(filename):
        return np.loadtxt(filename, dtype=np.float64)

    @staticmethod
    def write_elements(filename, triangles):
        triangles = np.asarray(triangles, dtype=np.int32)
        np.savetxt(filename, triangles, fmt='%d')

    @staticmethod
    def read_elements(filename):
        return np.loadtxt(filename, dtype=np.int32)

    @staticmethod
    def write_xml_mesh(filename, points, triangles):
        points = np.asarray(points, dtype=np.float64)
        triangles = np.asarray(triangles, dtype=np.int32)


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

    @staticmethod
    def validate_mt_data(frequencies, rho_a, phi):
        issues = []

        if np.any(frequencies <= 0):
            issues.append("频率必须为正")

        if np.any(rho_a <= 0):
            issues.append("视电阻率必须为正")

        if np.any(np.abs(phi) > 90.0):
            issues.append("相位绝对值不应超过90度")

        if len(frequencies) < 2:
            issues.append("至少需要两个频率点")



        if np.mean(phi[:max(1, len(phi)//4)]) < 30.0:
            issues.append("低频相位偏低，可能存在近场效应或数据质量问题")

        return len(issues) == 0, issues

    @staticmethod
    def validate_model(resistivities, thicknesses):
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
