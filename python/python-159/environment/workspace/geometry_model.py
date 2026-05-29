"""
geometry_model.py - 火箭发动机燃烧室几何建模
===============================================
基于保角变换与CORDIC算法的燃烧室-喷管型线生成与网格管理。

原项目映射:
- 611_joukowsky_transform -> Joukowsky变换生成翼型-like燃烧室型线
- 570_ice_io              -> 网格数据结构管理
- 219_cordic              -> CORDIC高精度几何角度计算

科学背景:
=========
液体火箭发动机燃烧室几何对燃烧稳定性有决定性影响。
燃烧室长度 L_c、喉部半径 r_t、扩张段半锥角 θ_n 等参数
决定了纵向声学模态频率:

    f_n = n · a / (2·L_c)   (n = 1, 2, 3, ...)

其中 a 为当地声速。本模块提供:
1. 基于Joukowsky变换的喷管型线生成
2. 燃烧室-喷管网格拓扑管理
3. 截面特性计算
"""

import numpy as np
from utils import cordic_cos_sin, cordic_arctan2, safe_divide, robust_sqrt, check_finite_array


class CombustionChamberGeometry:
    """
    火箭发动机燃烧室-喷管几何模型。
    
    几何参数 (典型RD-180级别液氧/煤油发动机):
        L_c       = 0.60 m   燃烧室圆柱段长度
        D_c       = 0.30 m   燃烧室直径
        r_t       = 0.075 m  喉部半径
        r_e       = 0.30 m   出口半径 (扩张比 ε = 16)
        θ_n       = 15°      扩张段半锥角
        r_conv    = 0.15 m   收敛段大端半径
        L_conv    = 0.10 m   收敛段长度
    
    坐标系:
        轴向 z: 从喷注面板 (z=0) 指向喷管出口
        径向 r: 从中心轴线向外
    """
    
    def __init__(self,
                 chamber_length: float = 0.60,
                 chamber_diameter: float = 0.30,
                 throat_radius: float = 0.075,
                 exit_radius: float = 0.30,
                 nozzle_half_angle_deg: float = 15.0,
                 convergent_radius: float = 0.15,
                 convergent_length: float = 0.10):
        
        # 参数验证
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
        self.epsilon = (exit_radius / throat_radius) ** 2  # 面积扩张比
        
        # 使用CORDIC高精度计算半锥角三角函数
        theta_n_rad = np.deg2rad(nozzle_half_angle_deg)
        self.cos_theta_n, self.sin_theta_n = cordic_cos_sin(theta_n_rad, n_iter=50)
        self.theta_n = theta_n_rad
        
        self.r_conv = convergent_radius
        self.L_conv = convergent_length
        
        # 预计算关键轴向位置
        self.z_injector = 0.0
        self.z_chamber_end = self.L_c
        self.z_throat = self.L_c + self.L_conv
        
        # 扩张段长度 (由面积比和锥角决定)
        # r_e = r_t + L_div · tan(θ_n)
        self.L_div = safe_divide(self.r_e - self.r_t, self.tan_theta_n(), default=0.5)
        self.z_exit = self.z_throat + self.L_div
        
        # 总体积 (近似)
        self._compute_volume()
        
        # 网格数据 (ICE-like结构)
        self._vertices = []
        self._elements = []
        self._vertex_labels = []
    
    def tan_theta_n(self) -> float:
        """使用CORDIC结果计算tan(θ_n)。"""
        return safe_divide(self.sin_theta_n, self.cos_theta_n, default=0.2679)
    
    def _compute_volume(self):
        """
        计算燃烧室-喷管内部总体积。
        
        V = V_cylinder + V_convergent + V_divergent
        
        圆柱段: V_cyl = π·R_c^2·L_c
        收敛段 (近似圆锥台): V_conv = (π/3)·L_conv·(R_c^2 + R_c·r_conv + r_conv^2)
        扩张段 (圆锥台): V_div = (π/3)·L_div·(r_t^2 + r_t·r_e + r_e^2)
        喉部 (近似): V_t = π·r_t^2·0.02
        """
        V_cyl = np.pi * self.R_c ** 2 * self.L_c
        V_conv = (np.pi / 3.0) * self.L_conv * \
                 (self.R_c ** 2 + self.R_c * self.r_conv + self.r_conv ** 2)
        V_div = (np.pi / 3.0) * self.L_div * \
                (self.r_t ** 2 + self.r_t * self.r_e + self.r_e ** 2)
        V_t = np.pi * self.r_t ** 2 * 0.02
        self.volume = V_cyl + V_conv + V_div + V_t
    
    def radius_at_z(self, z: float) -> float:
        """
        计算给定轴向位置z处的截面半径。
        
        分段定义:
            z ∈ [0, L_c]:        r(z) = R_c  (圆柱段)
            z ∈ [L_c, z_throat]: r(z) 线性收缩
            z ∈ [z_throat, exit]: r(z) = r_t + (z-z_throat)·tan(θ_n)  (圆锥扩张)
        """
        if z < 0:
            return self.R_c
        elif z <= self.L_c:
            return self.R_c
        elif z <= self.z_throat:
            # 线性收敛
            frac = safe_divide(z - self.L_c, self.L_conv, default=0.0)
            return self.R_c + (self.r_t - self.R_c) * frac
        elif z <= self.z_exit:
            # 圆锥扩张
            return self.r_t + (z - self.z_throat) * self.tan_theta_n()
        else:
            return self.r_e
    
    def area_at_z(self, z: float) -> float:
        """计算轴向位置z处的截面积。"""
        r = self.radius_at_z(z)
        return np.pi * r ** 2
    
    def cross_section_moment(self, z: float, order: int = 1) -> float:
        """
        计算截面矩:
            M_n = ∫_A r^n dA = 2π ∫_0^{R(z)} r^{n+1} dr = 2π · R(z)^{n+2} / (n+2)
        
        参数:
            z: 轴向位置
            order: 矩的阶数 n
        
        返回:
            截面矩的值
        """
        r = self.radius_at_z(z)
        if order < 0:
            raise ValueError("Order must be nonnegative")
        return 2.0 * np.pi * (r ** (order + 2)) / (order + 2)
    
    def generate_axisymmetric_grid(self, n_z: int = 100, n_r: int = 30) -> dict:
        """
        生成轴对称结构化网格 (ICE-like数据格式)。
        
        返回字典包含:
            vertices: (N, 2) 顶点坐标 (z, r)
            elements: (M, 4) 四边形单元顶点索引
            vertex_labels: 边界标签 (0=内部, 1=轴线, 2=壁面, 3=入口, 4=出口)
        
        原项目映射: 570_ice_io 的网格数据结构思想
        """
        if n_z < 3 or n_r < 2:
            raise ValueError("Grid resolution too low")
        
        # 轴向节点分布 (在喉部附近加密)
        z_nodes = self._distribute_axial_nodes(n_z)
        
        vertices = []
        vertex_labels = []
        elements = []
        
        for i, z in enumerate(z_nodes):
            r_max = self.radius_at_z(z)
            # 径向分布: 在壁面附近加密
            r_nodes = self._distribute_radial_nodes(r_max, n_r)
            
            for j, r in enumerate(r_nodes):
                vertices.append([z, r])
                
                # 标记边界
                if j == 0:
                    label = 1  # 轴线对称边界
                elif abs(r - r_max) < 1e-12:
                    if i == 0:
                        label = 3  # 入口 (喷注面板)
                    elif i == len(z_nodes) - 1:
                        label = 4  # 出口
                    else:
                        label = 2  # 壁面
                else:
                    label = 0  # 内部
                vertex_labels.append(label)
        
        vertices = np.array(vertices)
        vertex_labels = np.array(vertex_labels, dtype=int)
        
        # 生成四边形单元
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
        """
        轴向节点分布，在收敛段和喉部附近加密。
        
        使用变换:
            z = z_throat + L·sinh(ξ·s) / sinh(ξ)
        
        其中 ξ 控制加密程度。
        """
        # 在关键区域加密: [0, L_c], [L_c, z_throat], [z_throat, exit]
        z = np.zeros(n_z)
        
        # 三段比例
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
        
        # 重新平衡
        total = n1 + n2 + n3
        if total != n_z:
            n3 = n_z - n1 - n2
        
        # 圆柱段均匀
        z[:n1] = np.linspace(0, self.L_c, n1)
        # 收敛段和扩张段加密 (使用cosine分布)
        t2 = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, n2)))
        z[n1:n1+n2] = self.L_c + t2 * self.L_conv
        t3 = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, n3)))
        z[n1+n2:] = self.z_throat + t3 * self.L_div
        
        return np.sort(np.unique(z))
    
    def _distribute_radial_nodes(self, r_max: float, n_r: int) -> np.ndarray:
        """径向节点分布，壁面附近加密。"""
        t = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, n_r)))
        return t * r_max
    
    def acoustic_length(self) -> float:
        """
        计算声学等效长度。
        
        对于变截面管道，等效长度为:
            L_eff = ∫_0^{L_total} (A_min/A(z)) dz
        
        这与纵向声学模态频率相关:
            f_1 = a / (2·L_eff)
        """
        n = 500
        z = np.linspace(0, self.z_exit, n)
        A_min = np.pi * self.r_t ** 2
        integrand = np.array([safe_divide(A_min, self.area_at_z(zi), default=1.0) for zi in z])
        L_eff = np.trapezoid(integrand, z)
        return float(L_eff)
    
    def longitudinal_mode_frequencies(self, n_modes: int = 5, sound_speed: float = 1200.0) -> np.ndarray:
        """
        计算纵向声学模态频率。
        
        理论公式 (对于等效直管):
            f_n = (2n - 1) · a / (4 · L_eff)   (一端封闭一端开放)
            或 f_n = n · a / (2 · L_eff)       (两端开放)
        
        燃烧室情况: 喷注面板近似封闭 (高阻抗), 喷管出口开放
        因此采用四分之一波长管近似:
            f_n = (2n - 1) · a / (4 · L_eff)
        
        参数:
            n_modes: 计算的模态数
            sound_speed: 当地声速, m/s
        
        返回:
            模态频率数组, Hz
        """
        L_eff = self.acoustic_length()
        if L_eff <= 0:
            return np.zeros(n_modes)
        
        modes = np.array([(2 * n + 1) * sound_speed / (4.0 * L_eff) for n in range(n_modes)])
        return modes
    
    def apply_joukowsky_nozzle_contour(self, center_offset: float = -0.05, circle_radius: float = 0.12) -> np.ndarray:
        """
        使用Joukowsky变换生成更真实的喷管型线。
        
        Joukowsky变换:
            w = 0.5 · (z + 1/z)
        
        将圆映射为翼型。对于喷管型线，我们将变换应用于
        收敛-扩张段的截面轮廓，生成带有圆弧过渡的型线。
        
        参数:
            center_offset: 圆心偏移量
            circle_radius: 变换圆的半径
        
        返回:
            变换后的型线点 (z, r)
        
        原项目映射: 611_joukowsky_transform
        """
        n_points = 200
        theta = np.linspace(0, 2 * np.pi, n_points)
        
        # 圆上点 (复平面)
        z_circle = center_offset + circle_radius * np.exp(1j * theta)
        
        # Joukowsky变换
        w = 0.5 * (z_circle + 1.0 / z_circle)
        
        # 映射回物理空间 (缩放和平移)
        z_phys = np.real(w) * self.L_conv + self.L_c
        r_phys = np.abs(np.imag(w)) * self.R_c * 0.5 + self.r_t
        
        # 确保单调性
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
