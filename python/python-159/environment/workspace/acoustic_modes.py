"""
acoustic_modes.py - 燃烧室声场模态有限元分析
=============================================
基于有限元方法的燃烧室纵向/径向声学模态计算。

原项目映射:
- 371_fem_basis  -> 三角形单元Lagrange基函数用于声压场离散
- 179_circle_integrals -> 圆截面积分用于模态正交性验证

科学背景:
=========
燃烧不稳定性的本质是热声耦合，声学模态提供了压力振荡的
"共振腔"。对于圆柱形燃烧室:

纵向模态 (L-mode):
    p(z,t) = p_0·cos(n·π·z/L)·cos(ω_n·t)
    ω_n = n·π·a/L,  n=1,2,3,...

径向模态 (R-mode):
    p(r,θ,t) = p_0·J_m(α_{mn}·r/R)·cos(m·θ)·cos(ω_{mn}·t)
    ω_{mn} = α_{mn}·a/R
    
    其中 α_{mn} 为J_m的第n个零点

切向模态 (T-mode):
    p(r,θ,t) = p_0·J_m(α_{m0}·r/R)·sin(m·θ)·cos(ω_{m0}·t)

Helmholtz方程 (声压p的本征值问题):
    ∇²p + k²p = 0
    k = ω/a 为波数

边界条件:
    - 壁面: ∂p/∂n = 0  (刚性壁, 零法向速度)
    - 喷注面板 (入口): 高阻抗, 近似 ∂p/∂n = 0
    - 喷管 (出口): 低阻抗, 近似 p = 0 (开放端)

有限元离散:
    ∫_Ω (∇p·∇q - k²·p·q) dΩ = 0,  ∀q
    
    => (K - k²·M)·p = 0
    
    其中 K_{ij} = ∫_Ω ∇φ_i·∇φ_j dΩ  (刚度矩阵)
          M_{ij} = ∫_Ω φ_i·φ_j dΩ    (质量矩阵)
"""

import numpy as np
from utils import cordic_cos_sin, circle_monomial_integral, safe_divide, robust_sqrt, check_finite_array


class FEMBasis2DTriangle:
    """
    二维三角形Lagrange有限元基函数。
    
    原项目映射: 371_fem_basis / fem_basis_2d.m
    
    对于d次多项式，三角形上有 N = (d+1)(d+2)/2 个基函数。
    基函数节点采用均匀分布:
        (i/d, j/d),  i+j ≤ d
    
    基函数具有Lagrange性质:
        L_{ij}(x_k, y_k) = δ_{ik}δ_{jk}
    """
    
    def __init__(self, degree: int = 2):
        self.degree = degree
        self.n_basis = (degree + 1) * (degree + 2) // 2
        
        # 生成节点坐标 (面积坐标)
        self.nodes = []
        for i in range(degree + 1):
            for j in range(degree + 1 - i):
                k = degree - i - j
                # 转换为笛卡尔坐标 (参考三角形: (0,0), (1,0), (0,1))
                x = i / degree
                y = j / degree
                self.nodes.append((x, y, i, j, k))
    
    def evaluate_basis(self, idx: int, x: float, y: float) -> float:
        """
        计算第idx个基函数在(x,y)处的值。
        
        利用Lagrange插值构造:
            L_{ijk}(x,y) = Π_{p=0}^{i-1} (d·x - p)/(i - p)
                         × Π_{p=0}^{j-1} (d·y - p)/(j - p)  
                         × Π_{p=0}^{k-1} (d·(x+y) - (d-p))/(i+j - (d-p))
        
        其中 d = degree, k = d - i - j。
        """
        node = self.nodes[idx]
        i, j, k = node[2], node[3], node[4]
        d = self.degree
        
        value = 1.0
        denom = 1.0
        
        # x方向因子
        for p in range(i):
            value *= (d * x - p)
            denom *= (i - p)
        
        # y方向因子
        for p in range(j):
            value *= (d * y - p)
            denom *= (j - p)
        
        # (x+y)方向因子
        for p in range(k):
            value *= (d * (x + y) - (d - p))
            denom *= ((i + j) - (d - p))
        
        if abs(denom) < 1e-14:
            return 0.0
        
        return value / denom
    
    def evaluate_gradient(self, idx: int, x: float, y: float) -> tuple:
        """
        计算基函数的梯度 (∂L/∂x, ∂L/∂y)。
        
        使用数值微分 (中心差分)。
        """
        h = 1e-8
        L_pdx = self.evaluate_basis(idx, x + h, y)
        L_mdx = self.evaluate_basis(idx, x - h, y)
        L_pdy = self.evaluate_basis(idx, x, y + h)
        L_mdy = self.evaluate_basis(idx, x, y - h)
        
        dLdx = (L_pdx - L_mdx) / (2 * h)
        dLdy = (L_pdy - L_mdy) / (2 * h)
        
        return dLdx, dLdy


class AcousticModeAnalyzer:
    """
    燃烧室声学模态分析器。
    
    计算纵向、径向、切向声学模态及其频率。
    """
    
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
        
        # Bessel函数零点 (J_m的第n个零点)
        # 预计算的零点表 (m=0,1,2,3; n=1,2,3)
        self.bessel_zeros = {
            (0, 1): 2.4048, (0, 2): 5.5201, (0, 3): 8.6537,
            (1, 1): 3.8317, (1, 2): 7.0156, (1, 3): 10.1735,
            (2, 1): 5.1356, (2, 2): 8.4172, (2, 3): 11.6198,
            (3, 1): 6.3802, (3, 2): 9.7610, (3, 3): 13.0152,
        }
    
    def longitudinal_modes(self) -> dict:
        """
        计算纵向声学模态。
        
        频率公式:
            f_n = (2n - 1)·a / (4·L)   (一端封闭一端开放)
            
        对于火箭燃烧室:
            - 喷注面板端 (z=0): 高阻抗, 近似速度节点 (压力腹点)
            - 喷管端 (z=L): 低阻抗, 近似压力节点
        """
        # TODO(Hole_1): 实现纵向声学模态频率计算
        # 科学知识: 一端封闭一端开放的圆柱腔体纵向模态频率公式
        # f_n = (2n - 1) * a / (4L), n = 1, 2, 3, ...
        # 其中 a 为声速, L 为燃烧室长度
        # 同时需要构造对应的模态形状函数 mode_shape(z)
        # 提示: 封闭端(z=0)为压力腹点, 开放端(z=L)为压力节点
        modes = []
        frequencies = []
        
        for n in range(1, self.nL + 1):
            f_n = 0.0  # 需要替换为正确的物理公式
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
        """
        计算径向声学模态。
        
        频率公式:
            f_{mn} = α_{mn} · a / (2π·R)
        
        其中 α_{mn} 为J_m(x)的第n个零点。
        """
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
        """
        Bessel函数J_0(x)的近似计算。
        
        对于小x: J_0(x) ≈ 1 - x^2/4 + x^4/64 - ...
        对于大x: J_0(x) ≈ √(2/(πx))·cos(x - π/4)
        """
        x = float(x)
        if x < 0:
            x = -x
        
        if x < 3.0:
            # 级数展开
            x2 = x * x
            return 1.0 - x2 / 4.0 + x2 * x2 / 64.0 - x2 * x2 * x2 / 2304.0
        else:
            # 渐近展开
            return np.sqrt(2.0 / (np.pi * x)) * np.cos(x - 0.25 * np.pi)
    
    def compute_mode_coupling_matrix(self) -> np.ndarray:
        """
        计算模态耦合矩阵。
        
        对于线性热声系统，耦合矩阵描述各模态间的能量传递:
            C_{ij} = ∫_Ω φ_i·φ_j·W(x) dΩ
        
        其中 W(x) 为火焰响应的空间分布权重。
        """
        n_total = self.nL + self.nM * self.nR
        C = np.eye(n_total)
        
        # 简化的耦合: 相邻纵向模态间有弱耦合
        for i in range(self.nL - 1):
            C[i, i+1] = 0.1
            C[i+1, i] = 0.1
        
        return C
    
    def compute_orthogonality_integrals(self, mode_type: str = "L") -> np.ndarray:
        """
        计算模态正交性积分。
        
        理论预测:
            ∫_0^L cos(mπz/L)·cos(nπz/L) dz = (L/2)·δ_{mn}
        
        使用圆积分公式验证数值正交性。
        
        原项目映射: 179_circle_integrals
        """
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
        """
        计算Rayleigh准则积分。
        
        Rayleigh准则 (热声不稳定性的必要条件):
            R = ∫_V p'(x,t)·q'(x,t) dV > 0
        
        其中:
            p': 压力脉动
            q': 热释放率脉动
        
        当热释放脉动与压力脉动同相时 (R > 0)，
        系统可能产生自激振荡。
        
        参数:
            heat_release_oscillation: 热释放率脉动分布
            pressure_mode: 压力模态形状
        
        返回:
            Rayleigh积分值
        """
        if len(heat_release_oscillation) != len(pressure_mode):
            raise ValueError("Arrays must have same length")
        
        rayleigh = np.trapezoid(pressure_mode * heat_release_oscillation)
        return float(rayleigh)
    
    def compute_damping_rate(self, mode_index: int,
                             boundary_absorption: float = 0.05,
                             viscosity_damping: float = 0.02) -> float:
        """
        计算声学模态的阻尼率。
        
        总阻尼由以下部分组成:
            1. 喷管出口辐射阻尼: α_rad
            2. 粘性边界层阻尼: α_visc
            3. 粒子阻尼 (液滴): α_part
        
        简化的总阻尼:
            α_total = α_rad + α_visc
        """
        # 喷管辐射阻尼 (与马赫数相关)
        Ma = 0.3  # 特征马赫数
        alpha_rad = boundary_absorption * Ma
        
        # 粘性阻尼
        alpha_visc = viscosity_damping
        
        return alpha_rad + alpha_visc


class FEMHelmholtzSolver:
    """
    简化的1D有限元Helmholtz方程求解器。
    
    用于验证纵向模态解析解。
    """
    
    def __init__(self, length: float = 0.60, n_elements: int = 50):
        self.L = length
        self.ne = n_elements
        self.n_nodes = n_elements + 1
        self.h = length / n_elements
        self.x = np.linspace(0, length, self.n_nodes)
    
    def solve_eigenvalue(self, n_modes: int = 5) -> dict:
        """
        求解1D Helmholtz方程的本征值问题。
        
        -u'' = k²u,  u'(0)=0, u(L)=0
        
        使用线性有限元离散:
            K_{ij} = ∫ φ_i'·φ_j' dx
            M_{ij} = ∫ φ_i·φ_j dx
        """
        # 组装刚度矩阵和质量矩阵 (三对角)
        K = np.zeros((self.n_nodes, self.n_nodes))
        M = np.zeros((self.n_nodes, self.n_nodes))
        
        for e in range(self.ne):
            i, j = e, e + 1
            # 单元刚度
            K[i, i] += 1.0 / self.h
            K[i, j] += -1.0 / self.h
            K[j, i] += -1.0 / self.h
            K[j, j] += 1.0 / self.h
            
            # 单元质量 (一致质量)
            M[i, i] += self.h / 3.0
            M[i, j] += self.h / 6.0
            M[j, i] += self.h / 6.0
            M[j, j] += self.h / 3.0
        
        # 应用边界条件
        # u(L) = 0: 移除最后一个自由度
        K_red = K[:-1, :-1]
        M_red = M[:-1, :-1]
        
        # 求解广义本征值问题 K·v = λ·M·v
        eigenvalues, eigenvectors = np.linalg.eig(np.linalg.solve(M_red, K_red))
        
        # 排序
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # 波数 k = √λ
        k = np.sqrt(np.maximum(eigenvalues, 0.0))
        
        return {
            "wave_numbers": k[:n_modes],
            "eigenvectors": eigenvectors[:, :n_modes],
            "frequencies": k[:n_modes] * 1200.0 / (2 * np.pi)  # f = k·a/(2π)
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
    
    # Rayleigh准则测试
    z = np.linspace(0, analyzer.L, 100)
    p_mode = np.cos(np.pi * z / (2 * analyzer.L))
    q_osc = np.exp(-10 * (z - analyzer.L * 0.3) ** 2)  # 热释放集中在30%位置
    rayleigh = analyzer.rayleigh_criterion(q_osc, p_mode)
    print(f"\nRayleigh criterion: {rayleigh:.4e} (positive -> unstable)")
