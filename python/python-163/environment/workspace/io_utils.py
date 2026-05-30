
import numpy as np
import os


class SimulationIO:

    def __init__(self, output_dir="./thm_output"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def write_field(self, filename, field, header=""):
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            if header:
                f.write(f"# {header}\n")
            f.write(f"# shape: {field.shape}\n")
            np.savetxt(f, field.reshape(-1, field.shape[-1]) if field.ndim >= 2 else field.reshape(1, -1))

    def write_table(self, filename, columns, headers):
        if len(columns) != len(headers):
            raise ValueError("columns and headers must have the same length.")
        n_rows = len(columns[0])
        for col in columns:
            if len(col) != n_rows:
                raise ValueError("All columns must have the same length.")

        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            header_line = "  ".join(f"{h:>16s}" for h in headers)
            f.write(header_line + "\n")
            f.write("\n")
            for i in range(n_rows):
                row = "  ".join(f"{col[i]:>16.8e}" for col in columns)
                f.write(row + "\n")

    def write_summary(self, filename, summary_dict):
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            for key, value in summary_dict.items():
                if isinstance(value, float):
                    f.write(f"{key:30s}: {value:20.12e}\n")
                elif isinstance(value, np.ndarray):
                    f.write(f"{key:30s}: array shape {value.shape}\n")
                else:
                    f.write(f"{key:30s}: {value}\n")

    def read_matrix_file(self, filepath):
        rows = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) == 0:
                    continue
                try:
                    vals = [float(p) for p in parts]
                    rows.append(vals)
                except ValueError:
                    continue
        if not rows:
            return np.array([])
        return np.array(rows, dtype=np.float64)


def generate_parameter_table(output_path="parameter_table.txt"):
    params = {
        "Temperature_Initial_K": 423.15,
        "Temperature_Injection_K": 323.15,
        "Pressure_Initial_Pa": 20.0e6,
        "Porosity": 0.15,
        "Matrix_Permeability_m2": 1.0e-14,
        "Rock_Density_kg_m3": 2700.0,
        "Fluid_Density_kg_m3": 1000.0,
        "Rock_Thermal_Conductivity_W_mK": 2.5,
        "Fluid_Thermal_Conductivity_W_mK": 0.6,
        "Young_Modulus_Pa": 30.0e9,
        "Poisson_Ratio": 0.25,
        "Biot_Coefficient": 0.8,
        "Thermal_Expansion_Rock_1_K": 1.0e-5,
    }
    with open(output_path, 'w') as f:
        f.write("# Geothermal THM Simulation Parameters\n")
        f.write("#\n")
        f.write(f"{'Parameter':<40s} {'Value':>20s}\n")
        f.write("\n")
        for key, value in params.items():
            f.write(f"{key:<40s} {value:>20.6e}\n")
    return output_path
