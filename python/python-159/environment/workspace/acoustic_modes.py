
import numpy as np
from utils import cordic_cos_sin, circle_monomial_integral, safe_divide, robust_sqrt, check_finite_array


class FEMBasis2DTriangle:
    
    def __init__(self, degree: int = 2):
        self.degree = degree
        self.n_basis = (degree + 1) * (degree + 2) // 2
        

        self.nodes = []
        for i in range(degree + 1):
            for j in range(degree + 1 - i):
                k = degree - i - j

                x = i / degree
                y = j / degree
                self.nodes.append((x, y, i, j, k))
    
    def evaluate_basis(self, idx: int, x: float, y: float) -> float:
        node = self.nodes[idx]
        i, j, k = node[2], node[3], node[4]
        d = self.degree
        
        value = 1.0
        denom = 1.0
        

        for p in range(i):
            value *= (d * x - p)
            denom *= (i - p)
        

        for p in range(j):
            value *= (d * y - p)
            denom *= (j - p)
        

        for p in range(k):
            value *= (d * (x + y) - (d - p))
            denom *= ((i + j) - (d - p))
        
        if abs(denom) < 1e-14:
            return 0.0
        
        return value / denom
    
    def evaluate_gradient(self, idx: int, x: float, y: float) -> tuple:
        h = 1e-8
        L_pdx = self.evaluate_basis(idx, x + h, y)
        L_mdx = self.evaluate_basis(idx, x - h, y)
        L_pdy = self.evaluate_basis(idx, x, y + h)
        L_mdy = self.evaluate_basis(idx, x, y - h)
        
        dLdx = (L_pdx - L_mdx) / (2 * h)
        dLdy = (L_pdy - L_mdy) / (2 * h)
        
        return dLdx, dLdy


class AcousticModeAnalyzer:
    
    def __init__(self,
                 chamber_length: float = 0.60,
                 chamber_radius: float = 0.15,
                 sound_speed: float = 1200.0,
                 n_longitudinal: int = 5,
                 n_radial: int = 3,
                 n_azimuthal: int = 3):
        
        self.L = chamber_length
        self.R = chamber_radius
        self.a = sound_speed
        self.nL = n_longitudinal
        self.nR = n_radial
        self.nM = n_azimuthal
        


        self.bessel_zeros = {
            (0, 1): 2.4048, (0, 2): 5.5201, (0, 3): 8.6537,
            (1, 1): 3.8317, (1, 2): 7.0156, (1, 3): 10.1735,
            (2, 1): 5.1356, (2, 2): 8.4172, (2, 3): 11.6198,
            (3, 1): 6.3802, (3, 2): 9.7610, (3, 3): 13.0152,
        }
    
    def longitudinal_modes(self) -> dict:






        modes = []
        frequencies = []
        
        for n in range(1, self.nL + 1):
            f_n = 0.0
            frequencies.append(f_n)
            modes.append({
                "type": "L",
                "n": n,
                "frequency": f_n,
                "wavelength": 0.0,
                "mode_shape": lambda z: 0.0
            })
        
        return {
            "modes": modes,
            "frequencies": np.array(frequencies)
        }
    
    def radial_modes(self) -> dict:
        modes = []
        frequencies = []
        
        for m in range(self.nM):
            for n in range(1, self.nR + 1):
                alpha = self.bessel_zeros.get((m, n), (n + 0.25 * m) * np.pi)
                f_mn = alpha * self.a / (2.0 * np.pi * self.R)
                frequencies.append(f_mn)
                modes.append({
                    "type": "R" if m == 0 else "T",
                    "m": m,
                    "n": n,
                    "frequency": f_mn,
                    "alpha": alpha,
                    "mode_shape_radial": lambda r, alpha=alpha: self._bessel_j0_approx(alpha * r / self.R)
                })
        
        return {
            "modes": modes,
            "frequencies": np.array(frequencies)
        }
    
    def _bessel_j0_approx(self, x: float) -> float:
        x = float(x)
        if x < 0:
            x = -x
        
        if x < 3.0:

            x2 = x * x
            return 1.0 - x2 / 4.0 + x2 * x2 / 64.0 - x2 * x2 * x2 / 2304.0
        else:

            return np.sqrt(2.0 / (np.pi * x)) * np.cos(x - 0.25 * np.pi)
    
    def compute_mode_coupling_matrix(self) -> np.ndarray:
        n_total = self.nL + self.nM * self.nR
        C = np.eye(n_total)
        

        for i in range(self.nL - 1):
            C[i, i+1] = 0.1
            C[i+1, i] = 0.1
        
        return C
    
    def compute_orthogonality_integrals(self, mode_type: str = "L") -> np.ndarray:
        if mode_type == "L":
            n = self.nL
            z = np.linspace(0, self.L, 200)
            integrals = np.zeros((n, n))
            
            for i in range(n):
                for j in range(n):
                    fi = np.cos((2*i + 1) * np.pi * z / (2*self.L))
                    fj = np.cos((2*j + 1) * np.pi * z / (2*self.L))
                    integrals[i, j] = np.trapezoid(fi * fj, z)
            
            return integrals
        
        return np.array([])
    
    def rayleigh_criterion(self, heat_release_oscillation: np.ndarray,
                           pressure_mode: np.ndarray) -> float:
        if len(heat_release_oscillation) != len(pressure_mode):
            raise ValueError("Arrays must have same length")
        
        rayleigh = np.trapezoid(pressure_mode * heat_release_oscillation)
        return float(rayleigh)
    
    def compute_damping_rate(self, mode_index: int,
                             boundary_absorption: float = 0.05,
                             viscosity_damping: float = 0.02) -> float:

        Ma = 0.3
        alpha_rad = boundary_absorption * Ma
        

        alpha_visc = viscosity_damping
        
        return alpha_rad + alpha_visc


class FEMHelmholtzSolver:
    
    def __init__(self, length: float = 0.60, n_elements: int = 50):
        self.L = length
        self.ne = n_elements
        self.n_nodes = n_elements + 1
        self.h = length / n_elements
        self.x = np.linspace(0, length, self.n_nodes)
    
    def solve_eigenvalue(self, n_modes: int = 5) -> dict:

        K = np.zeros((self.n_nodes, self.n_nodes))
        M = np.zeros((self.n_nodes, self.n_nodes))
        
        for e in range(self.ne):
            i, j = e, e + 1

            K[i, i] += 1.0 / self.h
            K[i, j] += -1.0 / self.h
            K[j, i] += -1.0 / self.h
            K[j, j] += 1.0 / self.h
            

            M[i, i] += self.h / 3.0
            M[i, j] += self.h / 6.0
            M[j, i] += self.h / 6.0
            M[j, j] += self.h / 3.0
        


        K_red = K[:-1, :-1]
        M_red = M[:-1, :-1]
        

        eigenvalues, eigenvectors = np.linalg.eig(np.linalg.solve(M_red, K_red))
        

        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        

        k = np.sqrt(np.maximum(eigenvalues, 0.0))
        
        return {
            "wave_numbers": k[:n_modes],
            "eigenvectors": eigenvectors[:, :n_modes],
            "frequencies": k[:n_modes] * 1200.0 / (2 * np.pi)
        }


if __name__ == "__main__":
    analyzer = AcousticModeAnalyzer()
    
    L_modes = analyzer.longitudinal_modes()
    print("Longitudinal acoustic modes:")
    for m in L_modes["modes"]:
        print(f"  L{m['n']}: f = {m['frequency']:.1f} Hz")
    
    R_modes = analyzer.radial_modes()
    print("\nRadial/Tangential modes:")
    for m in R_modes["modes"][:6]:
        print(f"  {m['type']}{m['m']}{m['n']}: f = {m['frequency']:.1f} Hz")
    
    ortho = analyzer.compute_orthogonality_integrals("L")
    print(f"\nOrthogonality check (diagonal dominance):")
    diag = np.diag(ortho)
    offdiag_max = np.max(np.abs(ortho - np.diag(diag)))
    print(f"  Max off-diagonal: {offdiag_max:.6e}")
    
    fem = FEMHelmholtzSolver()
    fem_result = fem.solve_eigenvalue(n_modes=5)
    print(f"\nFEM eigenfrequencies: {fem_result['frequencies']} Hz")
    

    z = np.linspace(0, analyzer.L, 100)
    p_mode = np.cos(np.pi * z / (2 * analyzer.L))
    q_osc = np.exp(-10 * (z - analyzer.L * 0.3) ** 2)
    rayleigh = analyzer.rayleigh_criterion(q_osc, p_mode)
    print(f"\nRayleigh criterion: {rayleigh:.4e} (positive -> unstable)")
