"""
injector_layout.py - 喷注器面板优化布局
=========================================
基于组合优化理论的火箭发动机喷注器孔位最优布局设计。

原项目映射:
- 623_knapsack_brute -> 在质量约束下选择最优喷注单元组合
- 847_pariomino      -> 拼板理论用于喷孔密铺布局

科学背景:
=========
液体火箭发动机喷注器面板的布局直接影响:
1. 推进剂混合效率 (影响燃烧效率 η_c)
2. 燃烧稳定性 (影响热声耦合强度)
3. 燃烧室壁面热载荷分布

典型的同轴式喷注单元布局问题可建模为:
- 在圆形面板 (半径 R_p) 上布置 N 个喷注单元
- 每个单元占据一定面积 (类似拼板)
- 目标: 最大化覆盖均匀性，同时最小化单元间干扰

数学模型:
=========
决策变量: x_i ∈ {0,1}, i=1,...,M (M个候选位置)
目标函数: max Σ v_i·x_i  (v_i为位置i的燃烧效率权重)
约束条件:
    (1) Σ w_i·x_i ≤ W_max       (总质量/流量约束)
    (2) x_i + x_j ≤ 1  若 d(i,j) < d_min  (最小间距约束)
    (3) Σ x_i = N_target           (目标单元数)

其中 d_min 由液滴穿透深度和火焰相互作用距离决定:
    d_min ≈ 2·L_pen · tan(α_spray)
    L_pen = ρ_d · u_d^2 · D_d / (18·μ_g · u_g)  (Stokes穿透深度)
"""

import numpy as np
from itertools import combinations
from utils import cordic_cos_sin, cordic_arctan2, safe_divide, robust_sqrt


class InjectorLayoutOptimizer:
    """
    喷注器面板布局优化器。
    
    面板几何:
        圆形面板, 半径 R_panel
        每个喷注单元: 同轴式, 外径 d_outer, 中心间距 p
    
    优化目标:
        在给定面板尺寸和流量约束下，找到最优喷注单元布局
        使得燃烧效率最大且分布最均匀。
    """
    
    def __init__(self,
                 panel_radius: float = 0.12,
                 element_outer_diameter: float = 8.0e-3,
                 element_mass: float = 0.30,  # kg/s per element (coaxial injector pair)
                 target_total_flow: float = 80.0,  # kg/s total
                 min_spacing_factor: float = 1.5):
        
        if panel_radius <= 0 or element_outer_diameter <= 0:
            raise ValueError("Panel dimensions must be positive")
        
        self.R_panel = panel_radius
        self.d_outer = element_outer_diameter
        self.element_mass = element_mass
        self.target_flow = target_total_flow
        self.min_spacing = min_spacing_factor * element_outer_diameter
        
        # 目标单元数
        self.N_target = int(np.floor(target_total_flow / element_mass))
        
        # 候选位置网格
        self.candidates = []
        self.candidate_values = []  # 每个位置的效率权重
        self.candidate_weights = []  # 每个位置的流量权重
    
    def generate_candidate_positions_polar(self, n_rings: int = 8, n_sectors: int = 24) -> int:
        """
        基于极坐标生成候选喷注位置。
        
        布局策略:
            - 内区 (r < 0.3R): 较少单元, 避免中心过浓燃烧
            - 中区 (0.3R ≤ r < 0.7R): 主燃烧区, 密集布置
            - 外区 (0.7R ≤ r ≤ R): 边区冷却, 适当稀疏
        
        返回:
            候选位置数量
        """
        self.candidates = []
        self.candidate_values = []
        self.candidate_weights = []
        
        # 径向环分布
        # 使用CORDIC计算圆周上的均匀分布角度
        r_nodes = np.linspace(0.15 * self.R_panel, 0.95 * self.R_panel, n_rings)
        
        for r in r_nodes:
            # 周向单元数随半径增加
            n_angular = max(4, int(np.floor(2 * np.pi * r / self.min_spacing)))
            
            for k in range(n_angular):
                # 使用CORDIC计算精确角度位置
                theta = 2.0 * np.pi * k / n_angular
                cos_t, sin_t = cordic_cos_sin(theta, n_iter=40)
                x = r * cos_t
                y = r * sin_t
                
                # 效率权重: 中区最高, 内区和外区递减
                # 用高斯型分布模拟
                r_norm = r / self.R_panel
                efficiency_weight = np.exp(-4.0 * (r_norm - 0.55) ** 2)
                
                self.candidates.append((x, y, r, theta))
                self.candidate_values.append(efficiency_weight)
                self.candidate_weights.append(self.element_mass)
        
        return len(self.candidates)
    
    def generate_candidate_positions_triangular(self, n_layers: int = 10) -> int:
        """
        基于三角形密铺生成候选位置 (类似Pariomino的拼板思想)。
        
        三角形网格在圆形面板上提供最均匀的覆盖，
        每个单元有6个最近邻，距离相等。
        
        原项目映射: 847_pariomino 的密铺思想
        """
        self.candidates = []
        self.candidate_values = []
        self.candidate_weights = []
        
        spacing = self.min_spacing
        
        # 在方形区域内生成三角形网格，再筛选圆形内点
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
                
                # 效率权重: 考虑壁面边界层影响
                r_norm = r / self.R_panel
                wall_effect = 1.0 - 0.3 * np.exp(-10.0 * (1.0 - r_norm) ** 2)
                center_effect = 1.0 - 0.2 * np.exp(-20.0 * r_norm ** 2)
                efficiency_weight = wall_effect * center_effect
                
                self.candidates.append((x, y, r, theta))
                self.candidate_values.append(efficiency_weight)
                self.candidate_weights.append(self.element_mass)
        
        return len(self.candidates)
    
    def _check_spacing_constraint(self, selected_indices: list) -> bool:
        """检查选中的位置是否满足最小间距约束。"""
        for i, j in combinations(selected_indices, 2):
            x1, y1, _, _ = self.candidates[i]
            x2, y2, _, _ = self.candidates[j]
            dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            if dist < self.min_spacing * 0.99:
                return False
        return True
    
    def solve_brute_force_knapsack(self, max_elements: int = None, time_limit_seconds: float = 10.0) -> dict:
        """
        使用暴力枚举求解喷注器布局优化问题。
        
        问题形式化:
            max Σ v_i·x_i
            s.t. Σ w_i·x_i ≤ W_max
                 x_i + x_j ≤ 1  if d(i,j) < d_min
                 x_i ∈ {0,1}
        
        由于喷注器布局候选数通常较大，此处采用:
        1. 先筛选高价值候选
        2. 对缩减后的问题进行暴力枚举
        
        原项目映射: 623_knapsack_brute
        """
        import time
        start_time = time.time()
        
        if max_elements is None:
            max_elements = self.N_target
        
        n = len(self.candidates)
        if n == 0:
            raise RuntimeError("No candidate positions generated. Call generate_candidate_positions_* first.")
        
        # 如果候选太多，先排序并截断
        if n > 30:
            # 按价值密度排序
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
        
        # 预计算距离矩阵 (用于间距约束)
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
        
        # 枚举所有子集 (2^n_sub，对于n_sub≤30可处理)
        total_subsets = 2 ** n_sub
        checked = 0
        
        for mask in range(total_subsets):
            if time.time() - start_time > time_limit_seconds:
                break
            
            # 快速位运算提取选中索引
            selection = [i for i in range(n_sub) if (mask >> i) & 1]
            
            if len(selection) > max_elements:
                continue
            
            total_weight = sum(weights_sub[i] for i in selection)
            if total_weight > self.target_flow * 1.05:
                continue
            
            # 检查间距约束
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
        
        # 映射回原始索引
        if sorted_idx is not None and len(sorted_idx) != n:
            best_selection_orig = [sorted_idx[i] for i in best_selection]
        else:
            best_selection_orig = best_selection
        
        # 计算布局均匀性指标
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
        """
        贪心启发式算法快速求解布局问题。
        
        算法:
            1. 按价值密度降序排序
            2. 依次选择不违反约束的候选
        """
        if max_elements is None:
            max_elements = self.N_target
        
        n = len(self.candidates)
        if n == 0:
            raise RuntimeError("No candidate positions generated.")
        
        # 按价值密度排序
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
            
            # 检查间距
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
        """
        计算布局均匀性指标。
        
        基于CVT能量函数思想:
            U = Σ_i ∫_{V_i} ||x - c_i||^2 ρ(x) dA
        
        其中 V_i 是第i个单元的Voronoi区域，c_i是单元中心。
        均匀性越高，U越小。
        """
        if len(selected_indices) < 2:
            return 0.0
        
        positions = [(self.candidates[i][0], self.candidates[i][1]) for i in selected_indices]
        
        # 简化的均匀性: 最近邻距离的标准差
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
        
        # 均匀性指数: 0=完全均匀, 越大越不均匀
        uniformity = safe_divide(std_d, mean_d, default=0.0)
        return float(uniformity)
    
    def compute_mixture_ratio_distribution(self, selected_indices: list,
                                           ox_flow_fraction: float = 0.72) -> dict:
        """
        计算选定布局下的氧燃比分布。
        
        对于同轴式喷注器:
            - 每个单元中心管为液氧
            - 环缝为燃料
            - 总体氧燃比由流量分配决定
        
        参数:
            selected_indices: 选中的喷注单元索引
            ox_flow_fraction: 液氧占总流量比例 (典型0.72对应r=2.57)
        
        返回:
            局部氧燃比分布统计
        """
        n = len(selected_indices)
        if n == 0:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        
        # 理想情况: 所有单元相同
        # 实际情况: 考虑壁面附近燃料膜冷却的影响
        local_mr = []
        
        for idx in selected_indices:
            x, y, r, theta = self.candidates[idx]
            r_norm = r / self.R_panel
            
            # 壁面附近燃料比例增加 (膜冷却效应)
            wall_enrichment = 0.1 * np.exp(-15.0 * (1.0 - r_norm) ** 2)
            
            local_ox_frac = ox_flow_fraction * (1.0 - wall_enrichment)
            local_fuel_frac = 1.0 - local_ox_frac
            
            # 氧燃比
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
