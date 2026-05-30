
import numpy as np
from utils import cordic_cos_sin, cordic_arctan2, safe_divide, robust_sqrt, check_finite_array


class CombustionChamberGeometry:
    
    def __init__(self,
                 chamber_length: float = 0.60,
                 chamber_diameter: float = 0.30,
                 throat_radius: float = 0.075,
                 exit_radius: float = 0.30,
                 nozzle_half_angle_deg: float = 15.0,
                 convergent_radius: float = 0.15,
                 convergent_length: float = 0.10):
        

        if chamber_length <= 0 or chamber_diameter <= 0:
            raise ValueError("Chamber dimensions must be positive")
        if throat_radius >= chamber_diameter / 2.0:
            raise ValueError("Throat radius must be smaller than chamber radius")
        if exit_radius <= throat_radius:
            raise ValueError("Exit radius must be larger than throat radius")
        
        self.L_c = chamber_length
        self.D_c = chamber_diameter
        self.R_c = chamber_diameter / 2.0
        self.r_t = throat_radius
        self.r_e = exit_radius
        self.epsilon = (exit_radius / throat_radius) ** 2
        

        theta_n_rad = np.deg2rad(nozzle_half_angle_deg)
        self.cos_theta_n, self.sin_theta_n = cordic_cos_sin(theta_n_rad, n_iter=50)
        self.theta_n = theta_n_rad
        
        self.r_conv = convergent_radius
        self.L_conv = convergent_length
        

        self.z_injector = 0.0
        self.z_chamber_end = self.L_c
        self.z_throat = self.L_c + self.L_conv
        


        self.L_div = safe_divide(self.r_e - self.r_t, self.tan_theta_n(), default=0.5)
        self.z_exit = self.z_throat + self.L_div
        

        self._compute_volume()
        

        self._vertices = []
        self._elements = []
        self._vertex_labels = []
    
    def tan_theta_n(self) -> float:
        return safe_divide(self.sin_theta_n, self.cos_theta_n, default=0.2679)
    
    def _compute_volume(self):
        V_cyl = np.pi * self.R_c ** 2 * self.L_c
        V_conv = (np.pi / 3.0) * self.L_conv * \
                 (self.R_c ** 2 + self.R_c * self.r_conv + self.r_conv ** 2)
        V_div = (np.pi / 3.0) * self.L_div * \
                (self.r_t ** 2 + self.r_t * self.r_e + self.r_e ** 2)
        V_t = np.pi * self.r_t ** 2 * 0.02
        self.volume = V_cyl + V_conv + V_div + V_t
    
    def radius_at_z(self, z: float) -> float:
        if z < 0:
            return self.R_c
        elif z <= self.L_c:
            return self.R_c
        elif z <= self.z_throat:

            frac = safe_divide(z - self.L_c, self.L_conv, default=0.0)
            return self.R_c + (self.r_t - self.R_c) * frac
        elif z <= self.z_exit:

            return self.r_t + (z - self.z_throat) * self.tan_theta_n()
        else:
            return self.r_e
    
    def area_at_z(self, z: float) -> float:
        r = self.radius_at_z(z)
        return np.pi * r ** 2
    
    def cross_section_moment(self, z: float, order: int = 1) -> float:
        r = self.radius_at_z(z)
        if order < 0:
            raise ValueError("Order must be nonnegative")
        return 2.0 * np.pi * (r ** (order + 2)) / (order + 2)
    
    def generate_axisymmetric_grid(self, n_z: int = 100, n_r: int = 30) -> dict:
        if n_z < 3 or n_r < 2:
            raise ValueError("Grid resolution too low")
        

        z_nodes = self._distribute_axial_nodes(n_z)
        
        vertices = []
        vertex_labels = []
        elements = []
        
        for i, z in enumerate(z_nodes):
            r_max = self.radius_at_z(z)

            r_nodes = self._distribute_radial_nodes(r_max, n_r)
            
            for j, r in enumerate(r_nodes):
                vertices.append([z, r])
                

                if j == 0:
                    label = 1
                elif abs(r - r_max) < 1e-12:
                    if i == 0:
                        label = 3
                    elif i == len(z_nodes) - 1:
                        label = 4
                    else:
                        label = 2
                else:
                    label = 0
                vertex_labels.append(label)
        
        vertices = np.array(vertices)
        vertex_labels = np.array(vertex_labels, dtype=int)
        

        for i in range(n_z - 1):
            for j in range(n_r - 1):
                n0 = i * n_r + j
                n1 = (i + 1) * n_r + j
                n2 = (i + 1) * n_r + (j + 1)
                n3 = i * n_r + (j + 1)
                elements.append([n0, n1, n2, n3])
        
        elements = np.array(elements, dtype=int)
        
        self._vertices = vertices
        self._elements = elements
        self._vertex_labels = vertex_labels
        
        return {
            "vertices": vertices,
            "elements": elements,
            "vertex_labels": vertex_labels,
            "n_vertices": len(vertices),
            "n_elements": len(elements)
        }
    
    def _distribute_axial_nodes(self, n_z: int) -> np.ndarray:

        z = np.zeros(n_z)
        

        frac1 = self.L_c / self.z_exit
        frac2 = self.L_conv / self.z_exit
        frac3 = self.L_div / self.z_exit
        
        n1 = int(n_z * frac1)
        n2 = int(n_z * frac2)
        n3 = n_z - n1 - n2
        
        if n1 < 2:
            n1 = 2
        if n2 < 2:
            n2 = 2
        if n3 < 2:
            n3 = 2
        

        total = n1 + n2 + n3
        if total != n_z:
            n3 = n_z - n1 - n2
        

        z[:n1] = np.linspace(0, self.L_c, n1)

        t2 = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, n2)))
        z[n1:n1+n2] = self.L_c + t2 * self.L_conv
        t3 = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, n3)))
        z[n1+n2:] = self.z_throat + t3 * self.L_div
        
        return np.sort(np.unique(z))
    
    def _distribute_radial_nodes(self, r_max: float, n_r: int) -> np.ndarray:
        t = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, n_r)))
        return t * r_max
    
    def acoustic_length(self) -> float:
        n = 500
        z = np.linspace(0, self.z_exit, n)
        A_min = np.pi * self.r_t ** 2
        integrand = np.array([safe_divide(A_min, self.area_at_z(zi), default=1.0) for zi in z])
        L_eff = np.trapezoid(integrand, z)
        return float(L_eff)
    
    def longitudinal_mode_frequencies(self, n_modes: int = 5, sound_speed: float = 1200.0) -> np.ndarray:
        L_eff = self.acoustic_length()
        if L_eff <= 0:
            return np.zeros(n_modes)
        
        modes = np.array([(2 * n + 1) * sound_speed / (4.0 * L_eff) for n in range(n_modes)])
        return modes
    
    def apply_joukowsky_nozzle_contour(self, center_offset: float = -0.05, circle_radius: float = 0.12) -> np.ndarray:
        n_points = 200
        theta = np.linspace(0, 2 * np.pi, n_points)
        

        z_circle = center_offset + circle_radius * np.exp(1j * theta)
        

        w = 0.5 * (z_circle + 1.0 / z_circle)
        

        z_phys = np.real(w) * self.L_conv + self.L_c
        r_phys = np.abs(np.imag(w)) * self.R_c * 0.5 + self.r_t
        

        for i in range(1, len(r_phys)):
            if r_phys[i] < r_phys[i-1] and z_phys[i] > self.z_throat:
                r_phys[i] = r_phys[i-1]
        
        contour = np.column_stack([z_phys, r_phys])
        check_finite_array(contour.flatten(), "joukowsky contour")
        return contour


if __name__ == "__main__":
    geo = CombustionChamberGeometry()
    print(f"Chamber volume: {geo.volume:.6f} m^3")
    print(f"Expansion ratio: {geo.epsilon:.2f}")
    print(f"Acoustic length: {geo.acoustic_length():.4f} m")
    freqs = geo.longitudinal_mode_frequencies(n_modes=5)
    print(f"Longitudinal acoustic modes: {freqs} Hz")
    
    grid = geo.generate_axisymmetric_grid(n_z=50, n_r=20)
    print(f"Grid: {grid['n_vertices']} vertices, {grid['n_elements']} elements")
    
    contour = geo.apply_joukowsky_nozzle_contour()
    print(f"Joukowsky nozzle contour: {contour.shape}")
