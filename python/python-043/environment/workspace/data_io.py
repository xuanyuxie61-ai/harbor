"""
data_io.py — 地核发电机模拟数据读写与后处理格式转换模块

原项目映射:
  - 1310_triangle_io: Triangle网格格式读写
  - 351_fd_to_tec: FD数据转TECPLOT格式
  - 1062_scip_solution_read: SCIP优化解文件读取

改造思路:
  1. 将triangle_io的节点/单元数据读写改造为地核发电机网格I/O。
  2. 将fd_to_tec的格式转换改造为发电机场数据的结构化文本输出。
  3. 将scip_solution_read改造为参数扫描最优解文件的读取。
  4. 新增地磁学专用的球谐系数输出 (IGRF-like格式)。

科学背景:
  地磁场数据通常以球谐系数形式存储和交换 (如IGRF模型):
    g_l^m, h_l^m  (nT)
  本模块支持将数值模拟结果输出为标准格式，便于与观测数据对比。
"""

import numpy as np
from typing import Tuple, List, Dict, Optional


class DataIO:
    """
    地核发电机模拟数据读写管理器。
    """

    @staticmethod
    def write_field_snapshot(
        filename: str,
        r_grid: np.ndarray,
        theta_grid: np.ndarray,
        S_field: np.ndarray,
        T_field: np.ndarray,
    ):
        """
        将场快照写入结构化文本文件。

        文件格式:
          # Dynamo field snapshot
          # NR NTHETA
          # r_1 ... r_NR
          # theta_1 ... theta_NTHETA
          # S(r_i, theta_j)  (按行优先)
          # T(r_i, theta_j)
        """
        nr, ntheta = S_field.shape
        with open(filename, "w") as f:
            f.write("# Dynamo field snapshot\n")
            f.write(f"# {nr} {ntheta}\n")
            f.write(" ".join(f"{r:.8e}" for r in r_grid) + "\n")
            f.write(" ".join(f"{t:.8e}" for t in theta_grid) + "\n")
            for i in range(nr):
                f.write(" ".join(f"{S_field[i, j]:.8e}" for j in range(ntheta)) + "\n")
            for i in range(nr):
                f.write(" ".join(f"{T_field[i, j]:.8e}" for j in range(ntheta)) + "\n")

    @staticmethod
    def read_field_snapshot(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        从文件读取场快照。
        """
        with open(filename, "r") as f:
            lines = f.readlines()

        header = lines[1].strip().split()
        nr, ntheta = int(header[1]), int(header[2])
        r_grid = np.array([float(x) for x in lines[2].split()])
        theta_grid = np.array([float(x) for x in lines[3].split()])

        S_field = np.zeros((nr, ntheta))
        T_field = np.zeros((nr, ntheta))

        base = 4
        for i in range(nr):
            S_field[i, :] = [float(x) for x in lines[base + i].split()]
        base += nr
        for i in range(nr):
            T_field[i, :] = [float(x) for x in lines[base + i].split()]

        return r_grid, theta_grid, S_field, T_field

    @staticmethod
    def write_spherical_harmonics_coeffs(
        filename: str,
        coeffs_g: Dict[Tuple[int, int], float],
        coeffs_h: Dict[Tuple[int, int], float],
        epoch: float = 0.0,
    ):
        """
        输出球谐系数为类IGRF格式。

        格式:
          # Epoch: t
          # l  m  g_l^m  h_l^m
        """
        with open(filename, "w") as f:
            f.write(f"# Epoch: {epoch:.4f}\n")
            f.write("# l  m  g_l^m  h_l^m\n")
            max_l = max((l for l, m in coeffs_g.keys()), default=0)
            for l in range(1, max_l + 1):
                for m in range(l + 1):
                    g = coeffs_g.get((l, m), 0.0)
                    h = coeffs_h.get((l, m), 0.0)
                    f.write(f"{l:3d} {m:3d} {g:16.8e} {h:16.8e}\n")

    @staticmethod
    def write_tecplot_format(
        filename: str,
        r_grid: np.ndarray,
        theta_grid: np.ndarray,
        fields: Dict[str, np.ndarray],
    ):
        """
        将轴对称场数据输出为TECPLOT ASCII格式 (.dat)。
        改造自 fd_to_tec.m 的tec_write函数。

        格式:
          TITLE = "..."
          VARIABLES = "R", "Theta", "S", "T", ...
          ZONE I=nr, J=ntheta, F=POINT
          [数据行]
        """
        nr, ntheta = len(r_grid), len(theta_grid)
        var_names = ["R", "Theta"] + list(fields.keys())

        with open(filename, "w") as f:
            f.write(f'TITLE = "{filename}"\n')
            f.write(f'VARIABLES = "{var_names[0]}"')
            for v in var_names[1:]:
                f.write(f', "{v}"')
            f.write("\n")
            f.write(f"ZONE I={nr}, J={ntheta}, F=POINT\n")

            for j in range(ntheta):
                for i in range(nr):
                    line = f"{r_grid[i]:.8e} {theta_grid[j]:.8e}"
                    for key in fields:
                        line += f" {fields[key][i, j]:.8e}"
                    f.write(line + "\n")

    @staticmethod
    def read_scip_solution(filename: str, nx: int) -> np.ndarray:
        """
        读取SCIP优化求解器的解文件。
        改造自 scip_solution_read.m，用于读取参数扫描中的最优配置。

        假设文件格式:
          前两行为头部
          后续每行: "xNNN ..." 表示 x[NNN] = 1
        """
        x = np.zeros(nx, dtype=int)
        with open(filename, "r") as f:
            lines = f.readlines()

        for line in lines[2:]:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if parts and parts[0].startswith("x"):
                try:
                    idx = int(parts[0][1:])
                    if 1 <= idx <= nx:
                        x[idx - 1] = 1
                except ValueError:
                    pass
        return x

    @staticmethod
    def write_reversal_statistics(
        filename: str,
        reversal_times: List[float],
        reversal_durations: List[float],
        dipole_moments: np.ndarray,
    ):
        """
        输出极性反转统计结果。
        """
        with open(filename, "w") as f:
            f.write("# Geomagnetic Reversal Statistics\n")
            f.write(f"# Number of reversals: {len(reversal_times)}\n")
            f.write(f"# Mean interval: {np.mean(np.diff(reversal_times)) if len(reversal_times) > 1 else 0.0:.4f}\n")
            f.write(f"# Mean duration: {np.mean(reversal_durations) if reversal_durations else 0.0:.4f}\n")
            f.write("# Reversal_time  Duration\n")
            for t, d in zip(reversal_times, reversal_durations):
                f.write(f"{t:.6f} {d:.6f}\n")

    @staticmethod
    def compute_br_from_S(S: np.ndarray, r_grid: np.ndarray, theta_grid: np.ndarray) -> np.ndarray:
        """
        从极型势 S 计算径向磁场分量 B_r。

        在轴对称下:
          B_r = (1/(r² sinθ)) · ∂/∂θ(sinθ · S/(r sinθ)) · r sinθ
              = (1/r²) · ∂S/∂θ · ...

        实际公式:
          S = r sinθ · A_φ
          B_r = (1/(r sinθ)) · ∂A_φ/∂θ · sinθ = (1/r) · ∂(S/(r sinθ))/∂θ · sinθ

        简化: 使用中心差分近似
        """
        nr, ntheta = S.shape
        Br = np.zeros_like(S)
        dtheta = theta_grid[1] - theta_grid[0] if ntheta > 1 else 1.0

        for i in range(nr):
            r = r_grid[i]
            for j in range(1, ntheta - 1):
                # dS/dθ 的中心差分
                dS_dtheta = (S[i, j + 1] - S[i, j - 1]) / (2.0 * dtheta)
                sin_t = np.sin(theta_grid[j])
                # B_r ∝ (1/r²) * dS/dθ / sinθ * ... 简化为比例关系
                # 精确关系较复杂，此处用简化表达式
                Br[i, j] = dS_dtheta / (r * r * sin_t + 1e-30)

        return Br
