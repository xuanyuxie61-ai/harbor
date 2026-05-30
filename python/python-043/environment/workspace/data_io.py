
import numpy as np
from typing import Tuple, List, Dict, Optional


class DataIO:

    @staticmethod
    def write_field_snapshot(
        filename: str,
        r_grid: np.ndarray,
        theta_grid: np.ndarray,
        S_field: np.ndarray,
        T_field: np.ndarray,
    ):
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
        nr, ntheta = S.shape
        Br = np.zeros_like(S)
        dtheta = theta_grid[1] - theta_grid[0] if ntheta > 1 else 1.0

        for i in range(nr):
            r = r_grid[i]
            for j in range(1, ntheta - 1):

                dS_dtheta = (S[i, j + 1] - S[i, j - 1]) / (2.0 * dtheta)
                sin_t = np.sin(theta_grid[j])


                Br[i, j] = dS_dtheta / (r * r * sin_t + 1e-30)

        return Br
