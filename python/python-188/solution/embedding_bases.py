"""
基于Helmholtz方程特征函数的语义嵌入正交基构建模块

原项目映射: 515_helmholtz_exact
科学背景: Helmholtz方程 Del^2 Z = -k^2 Z 在圆形膜振动问题中的特征函数
          提供了自然正交基，可用于构建高维语义嵌入空间的正交坐标系。

数学模型:
    在极坐标下，Helmholtz方程:
        Z_rr + (1/r) Z_r + (1/r^2) Z_tt + k^2 Z = 0
    
    分离变量 Z(r,theta) = R(r) T(theta) 后:
        T'' + n^2 T = 0  =>  T(theta) = alpha*cos(n*theta) + beta*sin(n*theta)
        r^2 R'' + r R' + (r^2 k^2 - n^2) R = 0  =>  R(r) = gamma * J_n(k*r)
    
    边界条件 Z(a,theta) = 0 要求:
        k(m,n) = rho(m,n) / a
    其中 rho(m,n) 是 n 阶 Bessel 函数 J_n 的第 m 个零点。
    
    正交基函数:
        Phi_{m,n}(r,theta) = J_n(rho(m,n)*r/a) * [cos(n*theta), sin(n*theta)]

在NLP语义嵌入中的应用:
    将语义向量空间映射到该物理正交基上，利用Bessel函数的正交完备性
    实现语义嵌入的谱分解与重构。
"""

import numpy as np
from scipy.special import jv, jn_zeros


class SemanticEmbeddingBases:
    """
    基于Helmholtz方程特征函数的语义嵌入正交基系统。
    
    利用圆形膜振动的Bessel特征函数构建语义空间的正交基，
    实现语义向量的物理信息谱表示。
    """

    def __init__(self, radius: float = 1.0, max_mode_m: int = 5, max_mode_n: int = 5):
        """
        初始化语义嵌入正交基系统。
        
        Parameters
        ----------
        radius : float
            圆形区域的半径，对应语义空间的归一化尺度。
        max_mode_m : int
            Bessel函数零点的最大索引（径向模式数）。
        max_mode_n : int
            Bessel函数阶数的最大索引（角向模式数）。
        """
        if radius <= 0.0:
            raise ValueError(f"radius must be positive, got {radius}")
        if max_mode_m < 1:
            raise ValueError(f"max_mode_m must be at least 1, got {max_mode_m}")
        if max_mode_n < 0:
            raise ValueError(f"max_mode_n must be non-negative, got {max_mode_n}")

        self.radius = float(radius)
        self.max_mode_m = int(max_mode_m)
        self.max_mode_n = int(max_mode_n)
        self._compute_bessel_zeros()
        self._build_basis_index_map()

    def _compute_bessel_zeros(self):
        """
        预计算各阶Bessel函数的零点。
        
        J_n(x) 的前 m 个零点 rho(m,n) 满足:
            J_n(rho(m,n)) = 0
        
        对应的波数:
            k(m,n) = rho(m,n) / a
        """
        self.bessel_zeros = {}
        self.wavenumbers = {}

        for n in range(self.max_mode_n + 1):
            # jn_zeros(n, nt) 返回 J_n 的前 nt 个正零点
            # 当 n=0 时没有零根在零点；当 n>0 时第一个零根在 x=0
            # 这与原项目 515_helmholtz_exact 中的约定一致
            if n == 0:
                # n=0: m=1,...,max_mode_m 对应 jn_zeros(0, max_mode_m)
                zeros = jn_zeros(n, self.max_mode_m)
            else:
                # n>0: m=0 对应 rho=0（特殊零点）, m=1,...,max_mode_m 对应 jn_zeros(n, max_mode_m)
                positive_zeros = jn_zeros(n, self.max_mode_m)
                zeros = np.concatenate([[0.0], positive_zeros])

            self.bessel_zeros[n] = zeros
            self.wavenumbers[n] = zeros / self.radius

    def _build_basis_index_map(self):
        """
        构建全局基函数索引映射。
        
        基函数组织方式:
            - n=0: 只有 cos 项（无 sin 项），m=1,...,max_mode_m
            - n>0: 每个 n 有 cos 和 sin 两项，m=0,...,max_mode_m
        """
        self.basis_list = []
        for n in range(self.max_mode_n + 1):
            if n == 0:
                for m in range(1, self.max_mode_m + 1):
                    self.basis_list.append((m, n, 'cos'))
            else:
                for m in range(self.max_mode_m + 1):
                    self.basis_list.append((m, n, 'cos'))
                    self.basis_list.append((m, n, 'sin'))
        self.num_bases = len(self.basis_list)

    def evaluate_basis(self, r: np.ndarray, theta: np.ndarray,
                       m: int, n: int, angular_type: str) -> np.ndarray:
        """
        计算单个正交基函数在 (r, theta) 处的值。
        
        Parameters
        ----------
        r, theta : np.ndarray
            极坐标，r 需在 [0, radius] 范围内，theta 为弧度。
        m, n : int
            径向和角向模式索引。
        angular_type : str
            'cos' 或 'sin'。
            
        Returns
        -------
        np.ndarray
            基函数值 Phi_{m,n}^{type}(r, theta)。
            
        数学公式:
            Phi_{m,n}(r,theta) = J_n(rho(m,n)*r/a) * T_n(theta)
            T_n(theta) = cos(n*theta) 或 sin(n*theta)
        """
        r = np.asarray(r, dtype=float)
        theta = np.asarray(theta, dtype=float)

        if np.any(r < 0.0) or np.any(r > self.radius):
            raise ValueError(f"r must be in [0, {self.radius}], got range [{r.min()}, {r.max()}]")

        if n == 0 and m == 0:
            raise ValueError("For n=0, m=0 is illegal (no zero at origin for J_0)")

        # n=0: bessel_zeros[0] 索引 0..max_mode_m-1 对应 m=1..max_mode_m
        # n>0: bessel_zeros[n] 索引 0..max_mode_m 对应 m=0..max_mode_m
        if n == 0:
            rho = self.bessel_zeros[n][m - 1]
        else:
            rho = self.bessel_zeros[n][m]
        k = rho / self.radius

        # 径向因子: R(r) = J_n(k*r) = J_n(rho*r/a)
        radial = jv(n, k * r)

        # 角向因子
        if angular_type == 'cos':
            angular = np.cos(n * theta)
        elif angular_type == 'sin':
            angular = np.sin(n * theta)
        else:
            raise ValueError(f"angular_type must be 'cos' or 'sin', got {angular_type}")

        return radial * angular

    def evaluate_all_bases(self, r: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        计算所有正交基函数在 (r, theta) 处的值。
        
        Returns
        -------
        np.ndarray
            形状为 (num_points, num_bases) 的矩阵。
        """
        r = np.asarray(r, dtype=float)
        theta = np.asarray(theta, dtype=float)
        num_points = r.size
        Phi = np.zeros((num_points, self.num_bases))

        for idx, (m, n, angular_type) in enumerate(self.basis_list):
            Phi[:, idx] = self.evaluate_basis(r, theta, m, n, angular_type).flatten()

        return Phi

    def project_semantic_vector(self, semantic_field: np.ndarray,
                                r: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        将语义场函数投影到Helmholtz正交基上，获得谱系数。
        
        Parameters
        ----------
        semantic_field : np.ndarray
            在 (r, theta) 网格上定义的语义场函数值。
        r, theta : np.ndarray
            极坐标网格。
            
        Returns
        -------
        np.ndarray
            谱系数 c，满足:
                semantic_field \approx \sum_j c_j * Phi_j
                
        投影公式 (L2内积):
            c_j = (f, Phi_j) / (Phi_j, Phi_j)
            (f, g) = \int_0^a \int_0^{2\pi} f(r,theta) g(r,theta) r dr d\theta
        """
        semantic_field = np.asarray(semantic_field, dtype=float).flatten()
        r = np.asarray(r, dtype=float).flatten()
        theta = np.asarray(theta, dtype=float).flatten()

        if semantic_field.size != r.size or semantic_field.size != theta.size:
            raise ValueError("semantic_field, r, theta must have the same number of points")

        Phi = self.evaluate_all_bases(r, theta)
        coeffs = np.zeros(self.num_bases)

        # 使用梯形法则近似内积积分
        # 权重包含 r (极坐标 Jacobian)
        dr = np.diff(np.sort(np.unique(r))).mean() if len(np.unique(r)) > 1 else 1.0
        dtheta = np.diff(np.sort(np.unique(theta))).mean() if len(np.unique(theta)) > 1 else 1.0
        weights = r * dr * dtheta

        for j in range(self.num_bases):
            # L2内积 (f, Phi_j)
            numerator = np.sum(semantic_field * Phi[:, j] * weights)
            # 归一化 (Phi_j, Phi_j)
            denominator = np.sum(Phi[:, j] ** 2 * weights)
            if abs(denominator) > 1e-15:
                coeffs[j] = numerator / denominator
            else:
                coeffs[j] = 0.0

        return coeffs

    def reconstruct_semantic_field(self, coeffs: np.ndarray,
                                   r: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        由谱系数重构语义场函数。
        
        重构公式:
            f(r, theta) = \sum_j c_j * Phi_j(r, theta)
        """
        coeffs = np.asarray(coeffs, dtype=float)
        if coeffs.size != self.num_bases:
            raise ValueError(f"coeffs size must be {self.num_bases}, got {coeffs.size}")

        Phi = self.evaluate_all_bases(r, theta)
        return Phi @ coeffs

    def basis_orthogonality_check(self, r: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        验证正交基的正交性。
        
        对于精确正交基，应有:
            (Phi_i, Phi_j) = delta_{ij} * N_i
            
        Returns
        -------
        np.ndarray
            归一化的Gram矩阵。
        """
        Phi = self.evaluate_all_bases(r, theta)
        dr = np.diff(np.sort(np.unique(r))).mean() if len(np.unique(r)) > 1 else 1.0
        dtheta = np.diff(np.sort(np.unique(theta))).mean() if len(np.unique(theta)) > 1 else 1.0
        weights = np.asarray(r, dtype=float).flatten() * dr * dtheta

        gram = np.zeros((self.num_bases, self.num_bases))
        for i in range(self.num_bases):
            for j in range(self.num_bases):
                gram[i, j] = np.sum(Phi[:, i] * Phi[:, j] * weights)

        # 归一化
        diag = np.sqrt(np.diag(gram))
        diag[diag < 1e-15] = 1.0
        gram_norm = gram / np.outer(diag, diag)
        return gram_norm


def demo():
    """模块功能演示"""
    print("=" * 60)
    print("Helmholtz语义嵌入正交基系统演示")
    print("=" * 60)

    bases = SemanticEmbeddingBases(radius=1.0, max_mode_m=3, max_mode_n=2)
    print(f"\n正交基总数: {bases.num_bases}")
    print(f"基函数列表: {bases.basis_list}")

    # 生成极坐标网格
    r_grid = np.linspace(0.01, 1.0, 20)
    theta_grid = np.linspace(0, 2 * np.pi, 40)
    R, T = np.meshgrid(r_grid, theta_grid)
    r_flat = R.flatten()
    theta_flat = T.flatten()

    # 构造一个测试语义场: 高斯型语义密度
    semantic_field = np.exp(-((r_flat - 0.5) ** 2 + (np.sin(theta_flat)) ** 2) / 0.1)

    # 投影到正交基
    coeffs = bases.project_semantic_vector(semantic_field, r_flat, theta_flat)
    print(f"\n谱系数 (前10个): {coeffs[:10]}")

    # 重构
    reconstructed = bases.reconstruct_semantic_field(coeffs, r_flat, theta_flat)
    error = np.linalg.norm(semantic_field - reconstructed) / np.linalg.norm(semantic_field)
    print(f"\n重构相对误差: {error:.6e}")

    # 正交性检查
    gram = bases.basis_orthogonality_check(r_flat, theta_flat)
    off_diag_max = np.max(np.abs(gram - np.eye(bases.num_bases)))
    print(f"正交性偏差 (非对角元最大绝对值): {off_diag_max:.6e}")

    print("\n模块运行完成")
    return bases, coeffs, error


if __name__ == "__main__":
    demo()
