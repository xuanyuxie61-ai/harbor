
import numpy as np
from itertools import combinations
from utils import cordic_cos_sin, cordic_arctan2, safe_divide, robust_sqrt


class InjectorLayoutOptimizer:
    
    def __init__(self,
                 panel_radius: float = 0.12,
                 element_outer_diameter: float = 8.0e-3,
                 element_mass: float = 0.30,
                 target_total_flow: float = 80.0,
                 min_spacing_factor: float = 1.5):
        
        if panel_radius <= 0 or element_outer_diameter <= 0:
            raise ValueError("Panel dimensions must be positive")
        
        self.R_panel = panel_radius
        self.d_outer = element_outer_diameter
        self.element_mass = element_mass
        self.target_flow = target_total_flow
        self.min_spacing = min_spacing_factor * element_outer_diameter
        

        self.N_target = int(np.floor(target_total_flow / element_mass))
        

        self.candidates = []
        self.candidate_values = []
        self.candidate_weights = []
    
    def generate_candidate_positions_polar(self, n_rings: int = 8, n_sectors: int = 24) -> int:
        self.candidates = []
        self.candidate_values = []
        self.candidate_weights = []
        


        r_nodes = np.linspace(0.15 * self.R_panel, 0.95 * self.R_panel, n_rings)
        
        for r in r_nodes:

            n_angular = max(4, int(np.floor(2 * np.pi * r / self.min_spacing)))
            
            for k in range(n_angular):

                theta = 2.0 * np.pi * k / n_angular
                cos_t, sin_t = cordic_cos_sin(theta, n_iter=40)
                x = r * cos_t
                y = r * sin_t
                


                r_norm = r / self.R_panel
                efficiency_weight = np.exp(-4.0 * (r_norm - 0.55) ** 2)
                
                self.candidates.append((x, y, r, theta))
                self.candidate_values.append(efficiency_weight)
                self.candidate_weights.append(self.element_mass)
        
        return len(self.candidates)
    
    def generate_candidate_positions_triangular(self, n_layers: int = 10) -> int:
        self.candidates = []
        self.candidate_values = []
        self.candidate_weights = []
        
        spacing = self.min_spacing
        

        L = self.R_panel * 1.1
        nx = int(2 * L / spacing) + 1
        ny = int(2 * L / (spacing * np.sqrt(3) / 2)) + 1
        
        for j in range(-ny, ny + 1):
            y = j * spacing * np.sqrt(3) / 2
            offset = 0.0 if j % 2 == 0 else spacing / 2
            for i in range(-nx, nx + 1):
                x = i * spacing + offset
                r = np.sqrt(x ** 2 + y ** 2)
                
                if r > self.R_panel or r < 0.1 * self.R_panel:
                    continue
                
                theta = cordic_arctan2(y, x, n_iter=40)
                

                r_norm = r / self.R_panel
                wall_effect = 1.0 - 0.3 * np.exp(-10.0 * (1.0 - r_norm) ** 2)
                center_effect = 1.0 - 0.2 * np.exp(-20.0 * r_norm ** 2)
                efficiency_weight = wall_effect * center_effect
                
                self.candidates.append((x, y, r, theta))
                self.candidate_values.append(efficiency_weight)
                self.candidate_weights.append(self.element_mass)
        
        return len(self.candidates)
    
    def _check_spacing_constraint(self, selected_indices: list) -> bool:
        for i, j in combinations(selected_indices, 2):
            x1, y1, _, _ = self.candidates[i]
            x2, y2, _, _ = self.candidates[j]
            dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            if dist < self.min_spacing * 0.99:
                return False
        return True
    
    def solve_brute_force_knapsack(self, max_elements: int = None, time_limit_seconds: float = 10.0) -> dict:
        import time
        start_time = time.time()
        
        if max_elements is None:
            max_elements = self.N_target
        
        n = len(self.candidates)
        if n == 0:
            raise RuntimeError("No candidate positions generated. Call generate_candidate_positions_* first.")
        

        if n > 30:

            value_density = [v / max(w, 1e-10) for v, w in zip(self.candidate_values, self.candidate_weights)]
            sorted_idx = np.argsort(value_density)[::-1][:30]
            candidates_sub = [self.candidates[i] for i in sorted_idx]
            values_sub = [self.candidate_values[i] for i in sorted_idx]
            weights_sub = [self.candidate_weights[i] for i in sorted_idx]
        else:
            candidates_sub = self.candidates
            values_sub = self.candidate_values
            weights_sub = self.candidate_weights
            sorted_idx = list(range(n))
        
        n_sub = len(candidates_sub)
        

        dist_matrix = np.zeros((n_sub, n_sub))
        for i in range(n_sub):
            for j in range(i + 1, n_sub):
                x1, y1, _, _ = candidates_sub[i]
                x2, y2, _, _ = candidates_sub[j]
                dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        
        best_value = -1.0
        best_selection = []
        best_weight = 0.0
        

        total_subsets = 2 ** n_sub
        checked = 0
        
        for mask in range(total_subsets):
            if time.time() - start_time > time_limit_seconds:
                break
            

            selection = [i for i in range(n_sub) if (mask >> i) & 1]
            
            if len(selection) > max_elements:
                continue
            
            total_weight = sum(weights_sub[i] for i in selection)
            if total_weight > self.target_flow * 1.05:
                continue
            

            valid = True
            for ii in range(len(selection)):
                for jj in range(ii + 1, len(selection)):
                    if dist_matrix[selection[ii], selection[jj]] < self.min_spacing * 0.99:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            total_value = sum(values_sub[i] for i in selection)
            if total_value > best_value:
                best_value = total_value
                best_selection = selection
                best_weight = total_weight
            
            checked += 1
        

        if sorted_idx is not None and len(sorted_idx) != n:
            best_selection_orig = [sorted_idx[i] for i in best_selection]
        else:
            best_selection_orig = best_selection
        

        uniformity = self._compute_uniformity(best_selection_orig)
        
        return {
            "selected_indices": best_selection_orig,
            "n_selected": len(best_selection_orig),
            "total_value": best_value,
            "total_weight": best_weight,
            "uniformity_index": uniformity,
            "candidates_checked": checked,
            "positions": [self.candidates[i] for i in best_selection_orig]
        }
    
    def solve_greedy_heuristic(self, max_elements: int = None) -> dict:
        if max_elements is None:
            max_elements = self.N_target
        
        n = len(self.candidates)
        if n == 0:
            raise RuntimeError("No candidate positions generated.")
        

        indices = list(range(n))
        indices.sort(key=lambda i: self.candidate_values[i] / max(self.candidate_weights[i], 1e-10), reverse=True)
        
        selected = []
        total_weight = 0.0
        total_value = 0.0
        
        for idx in indices:
            if len(selected) >= max_elements:
                break
            if total_weight + self.candidate_weights[idx] > self.target_flow * 1.05:
                continue
            

            valid = True
            x_new, y_new, _, _ = self.candidates[idx]
            for s_idx in selected:
                x_s, y_s, _, _ = self.candidates[s_idx]
                dist = np.sqrt((x_new - x_s) ** 2 + (y_new - y_s) ** 2)
                if dist < self.min_spacing * 0.99:
                    valid = False
                    break
            
            if valid:
                selected.append(idx)
                total_weight += self.candidate_weights[idx]
                total_value += self.candidate_values[idx]
        
        uniformity = self._compute_uniformity(selected)
        
        return {
            "selected_indices": selected,
            "n_selected": len(selected),
            "total_value": total_value,
            "total_weight": total_weight,
            "uniformity_index": uniformity,
            "positions": [self.candidates[i] for i in selected]
        }
    
    def _compute_uniformity(self, selected_indices: list) -> float:
        if len(selected_indices) < 2:
            return 0.0
        
        positions = [(self.candidates[i][0], self.candidates[i][1]) for i in selected_indices]
        

        min_distances = []
        for i, (x1, y1) in enumerate(positions):
            min_d = np.inf
            for j, (x2, y2) in enumerate(positions):
                if i == j:
                    continue
                d = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                if d < min_d:
                    min_d = d
            min_distances.append(min_d)
        
        if len(min_distances) < 2:
            return 0.0
        
        mean_d = np.mean(min_distances)
        std_d = np.std(min_distances)
        

        uniformity = safe_divide(std_d, mean_d, default=0.0)
        return float(uniformity)
    
    def compute_mixture_ratio_distribution(self, selected_indices: list,
                                           ox_flow_fraction: float = 0.72) -> dict:
        n = len(selected_indices)
        if n == 0:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        


        local_mr = []
        
        for idx in selected_indices:
            x, y, r, theta = self.candidates[idx]
            r_norm = r / self.R_panel
            

            wall_enrichment = 0.1 * np.exp(-15.0 * (1.0 - r_norm) ** 2)
            
            local_ox_frac = ox_flow_fraction * (1.0 - wall_enrichment)
            local_fuel_frac = 1.0 - local_ox_frac
            

            mr = safe_divide(local_ox_frac, local_fuel_frac, default=2.56)
            local_mr.append(mr)
        
        return {
            "mean": float(np.mean(local_mr)),
            "std": float(np.std(local_mr)),
            "min": float(np.min(local_mr)),
            "max": float(np.max(local_mr)),
            "values": np.array(local_mr)
        }


if __name__ == "__main__":
    opt = InjectorLayoutOptimizer()
    n_cand = opt.generate_candidate_positions_triangular(n_layers=6)
    print(f"Generated {n_cand} candidate positions")
    
    result_greedy = opt.solve_greedy_heuristic()
    print(f"Greedy solution: {result_greedy['n_selected']} elements, "
          f"uniformity={result_greedy['uniformity_index']:.4f}")
    
    if n_cand <= 30:
        result_bf = opt.solve_brute_force_knapsack()
        print(f"Brute force solution: {result_bf['n_selected']} elements, "
              f"uniformity={result_bf['uniformity_index']:.4f}")
    
    mr_dist = opt.compute_mixture_ratio_distribution(result_greedy["selected_indices"])
    print(f"Mixture ratio: mean={mr_dist['mean']:.3f}, std={mr_dist['std']:.4f}")
