"""
brain_field.py
脑流场与体积效应模块

融合 navier_stokes_3d_exact (三维 Navier-Stokes 精确解)
与 tetrahedron_keast_rule (四面体高斯积分规则)。

核心科学模型：
  脑血流 / 脑脊液流场对神经电磁环境的调制：
    三维不可压缩 Navier-Stokes 方程:
      du/dt + (u . nabla) u = -1/rho nabla p + nu nabla^2 u + f_EM
      div u = 0

    其中 f_EM 为电磁体积力 (Lorentz 力密度):
      f_EM = J_ion x B

    使用 Ethier 精确解作为基准流场:
      u(x,y,z,t) = -a * [ exp(a*x) * sin(a*y+d*z) + exp(a*z) * cos(a*x+d*y) ] * exp(-d^2 t)
      v(x,y,z,t) = -a * [ exp(a*y) * sin(a*z+d*x) + exp(a*x) * cos(a*y+d*z) ] * exp(-d^2 t)
      w(x,y,z,t) = -a * [ exp(a*z) * sin(a*x+d*y) + exp(a*y) * cos(a*z+d*x) ] * exp(-d^2 t)
      p(x,y,z,t) = 0.5 * a^2 * exp(-2 d^2 t) * [
                     exp(2a*x) + 2 sin(a*x+d*y) cos(a*z+d*x) exp(a*(y+z))
                   + exp(2a*y) + 2 sin(a*y+d*z) cos(a*x+d*y) exp(a*(z+x))
                   + exp(2a*z) + 2 sin(a*z+d*x) cos(a*y+d*z) exp(a*(x+y))
                   ]

    参数: a = pi/4, d = pi/2 为典型值。

  神经核团体积积分 (Keast 规则)：
    计算三维神经组织的总电活动:
      Phi_total = int_Omega rho_ion(x,y,z) dV

    将区域 Omega 剖分为 N_tet 个四面体，在每个四面体上应用 Keast 积分:
      int_{T} f(x) dV = |det(J)| * sum_{q=1}^{N_q} w_q f(xi_q)

    其中 J 为参考四面体到物理四面体的 Jacobian 矩阵:
      J = [v1-v0, v2-v0, v3-v0]
      det(J) = 6 * Volume(T)

    参考四面体顶点:
      v0 = (0,0,0), v1 = (1,0,0), v2 = (0,1,0), v3 = (0,0,1)

    Keast 规则点数 (典型值):
      Rule 1: 1 点, degree 1
      Rule 2: 4 点, degree 2
      Rule 3: 5 点, degree 3
      Rule 4: 11 点, degree 4
"""

import numpy as np


class EthierNavierStokes:
    """
    Ethier 三维 Navier-Stokes 精确解。
    融合 uvwp_ethier 的核心公式。
    """

    def __init__(self, a=np.pi / 4.0, d=np.pi / 2.0):
        self.a = a
        self.d = d

    def evaluate(self, x, y, z, t):
        """
        计算流场速度 (u,v,w) 和压力 p。
        x, y, z, t: 标量或同形状数组
        """
        a = self.a
        d = self.d

        ex = np.exp(a * x)
        ey = np.exp(a * y)
        ez = np.exp(a * z)
        e2t = np.exp(-d * d * t)

        exy = np.exp(a * (x + y))
        eyz = np.exp(a * (y + z))
        ezx = np.exp(a * (z + x))

        sxy = np.sin(a * x + d * y)
        syz = np.sin(a * y + d * z)
        szx = np.sin(a * z + d * x)

        cxy = np.cos(a * x + d * y)
        cyz = np.cos(a * y + d * z)
        czx = np.cos(a * z + d * x)

        u = -a * (ex * syz + ez * cxy) * e2t
        v = -a * (ey * szx + ex * cyz) * e2t
        w = -a * (ez * sxy + ey * czx) * e2t
        p = 0.5 * a * a * e2t * e2t * (
            ex * ex + 2.0 * sxy * czx * eyz
            + ey * ey + 2.0 * syz * cxy * ezx
            + ez * ez + 2.0 * szx * cyz * exy
        )
        return u, v, w, p

    def vorticity(self, x, y, z, t):
        """
        计算涡量 omega = curl(u)。
        使用数值差分近似。
        """
        eps = 1e-5
        u_y = (self.evaluate(x, y + eps, z, t)[0] - self.evaluate(x, y - eps, z, t)[0]) / (2 * eps)
        u_z = (self.evaluate(x, y, z + eps, t)[0] - self.evaluate(x, y, z - eps, t)[0]) / (2 * eps)
        v_x = (self.evaluate(x + eps, y, z, t)[1] - self.evaluate(x - eps, y, z, t)[1]) / (2 * eps)
        v_z = (self.evaluate(x, y, z + eps, t)[1] - self.evaluate(x, y, z - eps, t)[1]) / (2 * eps)
        w_x = (self.evaluate(x + eps, y, z, t)[2] - self.evaluate(x - eps, y, z, t)[2]) / (2 * eps)
        w_y = (self.evaluate(x, y + eps, z, t)[2] - self.evaluate(x, y - eps, z, t)[2]) / (2 * eps)

        omega_x = w_y - v_z
        omega_y = u_z - w_x
        omega_z = v_x - u_y
        return omega_x, omega_y, omega_z


class KeastTetrahedronRule:
    """
    Keast 四面体积分规则。
    融合 keast_rule / keast_subrule 的核心数据与展开算法。
    """

    # 预定义 Keast 规则 (简化版，Rule 4, 11 points, degree 4)
    _RULES = {
        1: {
            'points': np.array([[0.25, 0.25, 0.25]]),
            'weights': np.array([1.0 / 6.0])
        },
        4: {
            'points': np.array([
                [0.58541020, 0.13819660, 0.13819660],
                [0.13819660, 0.58541020, 0.13819660],
                [0.13819660, 0.13819660, 0.58541020],
                [0.13819660, 0.13819660, 0.13819660],
            ]),
            'weights': np.array([0.25, 0.25, 0.25, 0.25]) / 6.0
        }
    }

    def __init__(self, rule_id=4):
        if rule_id not in self._RULES:
            raise ValueError(f"Rule {rule_id} not available. Use 1 or 4.")
        self.rule_id = rule_id
        data = self._RULES[rule_id]
        self.points_ref = data['points']  # (Nq, 3)
        self.weights = data['weights']    # (Nq,)
        self.Nq = len(self.weights)

    def integrate(self, func, vertices):
        """
        在四面体上积分函数 func。
        vertices: (4, 3) 四面体顶点 (v0, v1, v2, v3)
        func: callable, func(x,y,z) -> scalar or array
        """
        vertices = np.asarray(vertices, dtype=float)
        if vertices.shape != (4, 3):
            raise ValueError("vertices must be (4,3).")

        v0 = vertices[0]
        J = np.column_stack([
            vertices[1] - v0,
            vertices[2] - v0,
            vertices[3] - v0
        ])
        detJ = np.linalg.det(J)
        volume = abs(detJ) / 6.0
        if volume < 1e-14:
            raise ValueError("Degenerate tetrahedron (zero volume).")

        integral = 0.0
        for q in range(self.Nq):
            xi = self.points_ref[q]
            # 参考坐标 -> 物理坐标
            x_phys = v0 + J @ xi
            val = func(x_phys[0], x_phys[1], x_phys[2])
            integral += self.weights[q] * val

        integral *= abs(detJ)
        return integral


class NeuralVolumeIntegral:
    """
    神经核团的体积积分计算。
    """

    def __init__(self, keast_rule=None):
        if keast_rule is None:
            keast_rule = KeastTetrahedronRule(rule_id=4)
        self.keast = keast_rule

    def ionic_charge_density(self, x, y, z, V_membrane=-65.0, Na_out=145.0, Na_in=15.0,
                              K_out=5.0, K_in=140.0, T=310.0):
        """
        使用 Nernst 方程和 Goldman-Hodgkin-Katz 近似计算离子电荷密度。
        rho_ion = F * ( [Na+]_in - [Na+]_out * exp(-z e V/kT) + ... )
        简化模型: 与膜电位 V 线性相关。
        """
        # 简化: rho_ion proportional to (V - V_rest) / thickness
        # 单位: C/m^3
        F_faraday = 96485.0  # C/mol
        thickness = 5e-9     # 膜厚度 ~ 5 nm
        # 简化线性关系
        rho = F_faraday * 1e3 * (V_membrane + 65.0) / thickness  # 缩放到合理范围
        # 空间调制
        modulation = 1.0 + 0.1 * np.sin(x) * np.cos(y) * np.sin(z)
        return rho * modulation

    def integrate_region(self, tetrahedra, V_membrane=-65.0):
        """
        在多个四面体构成的区域上积分总电活动。
        tetrahedra: list of (4,3) arrays
        """
        total = 0.0
        for verts in tetrahedra:
            val = self.keast.integrate(
                lambda x, y, z: self.ionic_charge_density(x, y, z, V_membrane),
                verts
            )
            total += val
        return total


def demo_navier_stokes():
    """NS 精确解 demo。"""
    ns = EthierNavierStokes()
    x, y, z, t = 0.5, 0.5, 0.5, 0.05
    u, v, w, p = ns.evaluate(x, y, z, t)
    return u, v, w, p


def demo_tetrahedron_integral():
    """四面体积分 demo。"""
    keast = KeastTetrahedronRule(rule_id=4)
    # 单位四面体
    verts = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    # 积分 f(x,y,z) = x + y + z
    result = keast.integrate(lambda x, y, z: x + y + z, verts)
    # 解析解 = 1/8
    return result


def demo_volume_integral():
    """神经核团体积积分 demo。"""
    vol = NeuralVolumeIntegral()
    # 构造两个四面体
    tet1 = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    tet2 = np.array([
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0]
    ])
    total = vol.integrate_region([tet1, tet2], V_membrane=-50.0)
    return total
