"""
deformable_mirror.py — 变形镜与快速倾斜镜动力学模型

融合原项目:
  - 709_magic4_matrix (4阶幻方矩阵 → 驱动器布局优化)
  - 858_pendulum_double_ode (双摆混沌动力学 → FSM双轴机械振动)

功能:
  - 变形镜 (DM) 驱动器影响函数 (高斯/薄板样条)
  - 驱动器布局矩阵 (幻方优化排列)
  - 快速倾斜镜 (FSM) 双轴动力学模型
  - DM响应的机械延迟与谐振

物理模型:
  1. 驱动器影响函数 (Gaussian):
       I(r) = exp( -(r/a)^2 * ln(2) )
     其中 a 为影响函数半宽.

  2. 变形镜面形:
       phi_DM(x,y) = sum_{k=1}^{N_act} v_k * I_k(x,y)
     其中 v_k 为第k个驱动器的电压/控制信号.

  3. 驱动器布局优化 (源自709_magic4_matrix):
       使用4阶幻方矩阵生成非周期性驱动器排列,
       避免共振模式简并.
       幻方矩阵 M 满足: 每行/列/对角线之和为幻常数.
       将 M 的元素映射为驱动器位置权重.

  4. 快速倾斜镜双轴动力学 (源自858_pendulum_double_ode):
       将FSM建模为双摆系统:
         theta1: x轴倾斜角
         theta2: y轴倾斜角
       运动方程:
         d^2(theta1)/dt^2 = f1(theta1, theta1_dot, theta2, theta2_dot)
         d^2(theta2)/dt^2 = f2(theta1, theta1_dot, theta2, theta2_dot)
       包含质量、刚度、阻尼和耦合项.
"""

import numpy as np


# --- 驱动器影响函数 ---

def gaussian_influence_function(x, y, x0, y0, sigma):
    """
    高斯型影响函数.

    I(x,y; x0,y0) = exp( -((x-x0)^2 + (y-y0)^2) / (2*sigma^2) )
    """
    dx = x - x0
    dy = y - y0
    return np.exp(-(dx ** 2 + dy ** 2) / (2.0 * sigma ** 2))


def thin_plate_spline_influence(r, r0, c):
    """
    薄板样条影响函数.

    phi(r) = c * r^2 * log(r/r0),  r > 0
    phi(0) = 0
    """
    if np.any(r < 0):
        r = np.clip(r, 0, None)
    phi = np.zeros_like(r, dtype=np.float64)
    mask = r > 1e-10
    phi[mask] = c * (r[mask] ** 2) * np.log(r[mask] / max(r0, 1e-10))
    return phi


# --- 幻方布局 (源自709_magic4_matrix) ---

def magic4_matrix(n):
    """
    生成阶数为4的倍数的幻方矩阵.

    算法:
      1. 按顺序填入 1..n^2
      2. 对每个 4x4 子块, 若位置在对角线上, 替换为互补值 n^2+1-k

    幻方性质:
      每行、每列、两条对角线之和 = n(n^2+1)/2
    """
    if n % 4 != 0:
        raise ValueError("n must be a multiple of 4.")
    M = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            k1 = i * n + j + 1
            m1 = abs(i - j) % 4
            m2 = (i + j + 1) % 4
            if m1 == 0 or m2 == 0:
                M[i, j] = n * n + 1 - k1
            else:
                M[i, j] = k1
    return M


def generate_actuator_layout(n_actuators, aperture_radius=1.0, use_magic_square=False):
    """
    生成驱动器布局坐标.

    若 use_magic_square=True, 使用幻方矩阵权重对极坐标位置进行微扰,
    破坏周期性排列, 减少共振简并.
    """
    if n_actuators < 1:
        raise ValueError("n_actuators must be >= 1.")

    # 极坐标均匀分布 (斐波那契螺旋)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    indices = np.arange(n_actuators)
    r = aperture_radius * np.sqrt(indices / (n_actuators - 0.5)) if n_actuators > 1 else np.array([0.0])
    theta = indices * golden_angle

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    if use_magic_square and n_actuators >= 16:
        # 使用4阶幻方对前16个驱动器施加微扰
        magic = magic4_matrix(4)
        magic_norm = (magic - np.mean(magic)) / (np.max(magic) - np.min(magic) + 1e-10)
        for idx in range(min(16, n_actuators)):
            i = idx // 4
            j = idx % 4
            perturb = magic_norm[i, j] * 0.05 * aperture_radius
            x[idx] += perturb * np.cos(theta[idx])
            y[idx] += perturb * np.sin(theta[idx])

    return x, y


# --- 变形镜模型 ---

class DeformableMirror:
    """
    变形镜模型.
    """

    def __init__(self, n_actuators, grid_size, aperture_radius=1.0,
                 influence_sigma=None, use_magic_square_layout=False):
        self.n_actuators = n_actuators
        self.grid_size = grid_size
        self.aperture_radius = aperture_radius

        if influence_sigma is None:
            influence_sigma = 0.15 * aperture_radius

        self.influence_sigma = influence_sigma
        self.act_x, self.act_y = generate_actuator_layout(
            n_actuators, aperture_radius, use_magic_square_layout
        )

        # 预计算影响函数矩阵 (Npix x N_act)
        x = np.linspace(-aperture_radius, aperture_radius, grid_size)
        y = np.linspace(-aperture_radius, aperture_radius, grid_size)
        X, Y = np.meshgrid(x, y)

        self.influence_matrix = np.zeros((grid_size, grid_size, n_actuators), dtype=np.float64)
        for k in range(n_actuators):
            self.influence_matrix[:, :, k] = gaussian_influence_function(
                X, Y, self.act_x[k], self.act_y[k], influence_sigma
            )

        # 展平
        self.influence_flat = self.influence_matrix.reshape(grid_size * grid_size, n_actuators)

    def compute_surface(self, voltages):
        """
        由驱动器电压计算镜面面形.
        """
        if len(voltages) != self.n_actuators:
            raise ValueError("voltages length must equal n_actuators.")
        surface = np.tensordot(self.influence_matrix, voltages, axes=([2], [0]))
        return surface

    def compute_surface_flat(self, voltages):
        """展平面形."""
        return self.influence_flat @ voltages

    def voltage_to_zernike_response(self, zernike_basis_flat, mask):
        """
        计算驱动器电压到Zernike系数的响应矩阵.

        R[j,k] = integral Z_j(x,y) * I_k(x,y) dA / integral Z_j^2 dA
        """
        n_modes = zernike_basis_flat.shape[1]
        R = np.zeros((n_modes, self.n_actuators), dtype=np.float64)
        mask_vec = mask.ravel()

        for k in range(self.n_actuators):
            Ik = self.influence_flat[:, k]
            for j in range(n_modes):
                Zj = zernike_basis_flat[:, j]
                num = np.sum(Zj[mask_vec] * Ik[mask_vec])
                den = np.sum(Zj[mask_vec] ** 2)
                if den > 1e-30:
                    R[j, k] = num / den

        return R


# --- 快速倾斜镜双轴动力学 (源自858_pendulum_double_ode) ---

class FastSteeringMirrorDynamics:
    """
    快速倾斜镜 (FSM) 双轴动力学模型.

    将两个倾斜轴建模为耦合双摆:
      theta1: x轴倾斜角 (rad)
      theta2: y轴倾斜角 (rad)
    """

    def __init__(self, g=9.81, m1=0.01, m2=0.01, l1=0.05, l2=0.05,
                 damping1=0.5, damping2=0.5, coupling=0.1):
        self.g = g
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.damping1 = damping1
        self.damping2 = damping2
        self.coupling = coupling

    def derivatives(self, state, t, control_torque1, control_torque2):
        """
        计算双摆ODE的右手边.

        状态: [theta1, omega1, theta2, omega2]

        运动方程:
          d(theta1)/dt = omega1
          d(omega1)/dt = [ -g*(2*m1+m2)*sin(theta1) - m2*g*sin(theta1-2*theta2)
                           - 2*sin(theta1-theta2)*m2*(omega2^2*l2 + omega1^2*l1*cos(theta1-theta2))
                           + T1 - damping1*omega1 + coupling*(theta2-theta1) ]
                         / [ 2*l1*(m1 + m2 - m2*cos^2(theta1-theta2)) ]

          d(theta2)/dt = omega2
          d(omega2)/dt = [ (m1+m2)*(l1*omega1^2 + g*cos(theta1))*sin(theta1-theta2)
                           + m2*l2*omega2^2*sin(theta1-theta2)*cos(theta1-theta2)
                           + T2 - damping2*omega2 + coupling*(theta1-theta2) ]
                         / [ l2*(m1 + m2 - m2*cos^2(theta1-theta2)) ]
        """
        theta1, omega1, theta2, omega2 = state

        delta = theta1 - theta2
        cos_delta = np.cos(delta)
        sin_delta = np.sin(delta)
        denom1 = 2.0 * self.l1 * (self.m1 + self.m2 - self.m2 * cos_delta ** 2)
        denom2 = self.l2 * (self.m1 + self.m2 - self.m2 * cos_delta ** 2)

        if abs(denom1) < 1e-10:
            denom1 = 1e-10
        if abs(denom2) < 1e-10:
            denom2 = 1e-10

        alpha1 = (-self.g * (2.0 * self.m1 + self.m2) * np.sin(theta1)
                  - self.m2 * self.g * np.sin(theta1 - 2.0 * theta2)
                  - 2.0 * sin_delta * self.m2 * (
                      omega2 ** 2 * self.l2 + omega1 ** 2 * self.l1 * cos_delta)
                  + control_torque1 - self.damping1 * omega1
                  + self.coupling * (theta2 - theta1)) / denom1

        alpha2 = ((self.m1 + self.m2) * (
                      self.l1 * omega1 ** 2 + self.g * np.cos(theta1)) * sin_delta
                  + self.m2 * self.l2 * omega2 ** 2 * sin_delta * cos_delta
                  + control_torque2 - self.damping2 * omega2
                  + self.coupling * (theta1 - theta2)) / denom2

        return np.array([omega1, alpha1, omega2, alpha2], dtype=np.float64)

    def step_rk4(self, state, dt, t, torque1, torque2):
        """RK4积分一步."""
        k1 = dt * self.derivatives(state, t, torque1, torque2)
        k2 = dt * self.derivatives(state + 0.5 * k1, t + 0.5 * dt, torque1, torque2)
        k3 = dt * self.derivatives(state + 0.5 * k2, t + 0.5 * dt, torque1, torque2)
        k4 = dt * self.derivatives(state + k3, t + dt, torque1, torque2)
        return state + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    def simulate_response(self, control_sequence_x, control_sequence_y, dt=1e-4):
        """
        模拟FSM对控制序列的响应.

        返回: theta1_history, theta2_history
        """
        n_steps = len(control_sequence_x)
        if len(control_sequence_y) != n_steps:
            raise ValueError("control sequences must have same length.")

        state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        history = np.zeros((n_steps, 4), dtype=np.float64)

        for i in range(n_steps):
            state = self.step_rk4(state, dt, i * dt,
                                  control_sequence_x[i], control_sequence_y[i])
            history[i, :] = state

        return history[:, 0], history[:, 2]
