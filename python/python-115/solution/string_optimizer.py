"""
string_optimizer.py
弦方法（String Method）反应路径优化模块

核心功能：
- 弦方法路径演化
- 样条插值参数化反应路径
- 路径能量剖面分析
- 序列文件索引管理（类比 contour_sequence4 的序列处理）
- 基于 T-puzzle 变换的路径对称性操作

科学背景：
弦方法是寻找最小能量路径（MEP）的高效算法：
    γ(s): [0,1] → R^{3N}
    s 为弧长参数

演化方程（弦方法）：
    ∂γ/∂t = -∇V^⊥(γ) = -[∇V(γ) - (∇V·t̂) t̂]

其中 t̂ = ∂γ/∂s / |∂γ/∂s| 为路径单位切向量。
垂直于路径的力分量驱动路径向 MEP 收敛。

离散化实现（N 个图像）：
    1. 演化：每个图像沿 -∇V 方向移动（攀爬图像除外）
    2. 重参数化：沿路径等弧长重新分布图像
    3. 重复直到收敛

收敛判据：
    max_i |∇V^⊥(γ_i)| < ε

样条插值用于平滑路径：
    γ(s) = Σ_j N_j(s) γ_j
    N_j 为 B-样条基函数

对称性操作（来自 T-puzzle 的旋转/反射变换）：
    在分子点群对称性下，反应路径可能具有等价副本。
    通过对称操作可减少计算量。
"""

import numpy as np
from scipy.interpolate import CubicSpline


class PathParameterization:
    """反应路径参数化工具"""

    @staticmethod
    def arc_length_parameterize(path):
        """
        计算路径的弧长参数化

        弧长元素：
            ds = |dγ/dλ| dλ
        """
        path = np.asarray(path, dtype=float)
        n_images = path.shape[0]
        s = np.zeros(n_images, dtype=float)

        for i in range(1, n_images):
            s[i] = s[i - 1] + np.linalg.norm(path[i] - path[i - 1])

        if s[-1] > 0:
            s /= s[-1]
        return s

    @staticmethod
    def reparametrize_equidistant(path, n_new=None):
        """
        将路径重新参数化为等弧长分布

        使用三次样条插值：
            γ_{new}(s_k) = CubicSpline(s, γ)(s_k)
            s_k = k / (N-1)
        """
        path = np.asarray(path, dtype=float)
        if n_new is None:
            n_new = path.shape[0]

        s_old = PathParameterization.arc_length_parameterize(path)
        s_new = np.linspace(0, 1, n_new)

        dim = path.shape[1]
        path_new = np.zeros((n_new, dim), dtype=float)

        for d in range(dim):
            cs = CubicSpline(s_old, path[:, d])
            path_new[:, d] = cs(s_new)

        return path_new


class StringMethod:
    """
    弦方法反应路径优化器
    """

    def __init__(self, energy_func, gradient_func, n_images=20,
                 spring_const=1.0, dt=0.01, max_iter=500, tol=1e-4):
        """
        参数：
            energy_func: V(x) 势能函数
            gradient_func: ∇V(x) 梯度函数
            n_images: 路径图像数
            spring_const: 弹簧常数（NEB 用）
            dt: 时间步长
            max_iter: 最大迭代数
            tol: 收敛容限
        """
        self.energy_func = energy_func
        self.gradient_func = gradient_func
        self.n_images = n_images
        self.spring_const = spring_const
        self.dt = dt
        self.max_iter = max_iter
        self.tol = tol

    def compute_tangents(self, path, energies):
        """
        计算路径切向量

        使用高低能量邻居插值：
            τ_i = (V_{i+1} - V_i) τ_i^+ + (V_i - V_{i-1}) τ_i^-
        其中 τ_i^+ = R_{i+1} - R_i, τ_i^- = R_i - R_{i-1}
        """
        path = np.asarray(path, dtype=float)
        n = path.shape[0]
        tangents = np.zeros_like(path)

        for i in range(1, n - 1):
            v_max = max(abs(energies[i + 1] - energies[i]),
                        abs(energies[i] - energies[i - 1]))
            if v_max < 1e-12:
                tau = path[i + 1] - path[i - 1]
            else:
                if energies[i + 1] > energies[i - 1]:
                    tau = (energies[i + 1] - energies[i]) * (path[i + 1] - path[i])
                    tau += (energies[i] - energies[i - 1]) * (path[i] - path[i - 1])
                else:
                    tau = (energies[i - 1] - energies[i]) * (path[i] - path[i - 1])
                    tau += (energies[i] - energies[i + 1]) * (path[i + 1] - path[i])

            norm = np.linalg.norm(tau)
            if norm > 1e-12:
                tangents[i] = tau / norm

        return tangents

    def evolve_string(self, path_init):
        """
        弦方法路径演化

        算法：
            for iteration:
                1. 计算每个图像的势能和梯度
                2. 计算路径切向量
                3. 更新图像：γ_i ← γ_i - dt * (∇V - (∇V·τ)τ)
                4. 固定端点（反应物和产物）
                5. 重新参数化（等弧长）
                6. 检查收敛
        """
        path = np.asarray(path_init, dtype=float).copy()
        n_images = path.shape[0]

        # 记录历史
        history = [path.copy()]
        energy_history = []

        for it in range(self.max_iter):
            # 计算能量和梯度
            energies = np.array([self.energy_func(path[i]) for i in range(n_images)])
            gradients = np.array([self.gradient_func(path[i]) for i in range(n_images)])
            energy_history.append(energies.copy())

            # 计算切向量
            tangents = self.compute_tangents(path, energies)

            # 垂直力
            force_perp = np.zeros_like(path)
            max_force = 0.0
            for i in range(1, n_images - 1):
                grad = gradients[i]
                tau = tangents[i]
                f_parallel = np.dot(grad, tau) * tau
                f_perp = grad - f_parallel
                force_perp[i] = -f_perp
                max_force = max(max_force, np.linalg.norm(f_perp))

            # 更新
            path[1:n_images - 1] += self.dt * force_perp[1:n_images - 1]

            # 重新参数化
            path = PathParameterization.reparametrize_equidistant(path, n_images)
            history.append(path.copy())

            if max_force < self.tol:
                break

        final_energies = np.array([self.energy_func(path[i]) for i in range(n_images)])
        return path, final_energies, history, energy_history

    def climbing_image_neb(self, path_init):
        """
        攀爬图像 Nudged Elastic Band (CI-NEB)

        在标准 NEB 中，弹簧力保持图像分布：
            F_i^s = k (|R_{i+1} - R_i| - |R_i - R_{i-1}|) τ_i

        在攀爬图像中，最高能量图像沿梯度方向攀爬：
            F_i^{climb} = ∇V - 2(∇V·τ) τ
        """
        path = np.asarray(path_init, dtype=float).copy()
        n_images = path.shape[0]

        for it in range(self.max_iter):
            energies = np.array([self.energy_func(path[i]) for i in range(n_images)])
            gradients = np.array([self.gradient_func(path[i]) for i in range(n_images)])
            tangents = self.compute_tangents(path, energies)

            # 找到最高能量图像（过渡态候选）
            ts_idx = np.argmax(energies[1:n_images - 1]) + 1

            forces = np.zeros_like(path)
            max_force = 0.0

            for i in range(1, n_images - 1):
                grad = gradients[i]
                tau = tangents[i]
                f_parallel = np.dot(grad, tau) * tau
                f_perp = grad - f_parallel

                if i == ts_idx:
                    # 攀爬图像：反转平行分量
                    forces[i] = -(f_perp - f_parallel)
                else:
                    # 标准 NEB：垂直力 + 弹簧力
                    f_spring = self.spring_const * (
                            np.linalg.norm(path[i + 1] - path[i]) -
                            np.linalg.norm(path[i] - path[i - 1])
                    ) * tau
                    forces[i] = -f_perp + f_spring

                max_force = max(max_force, np.linalg.norm(forces[i]))

            path[1:n_images - 1] += self.dt * forces[1:n_images - 1]
            path = PathParameterization.reparametrize_equidistant(path, n_images)

            if max_force < self.tol:
                break

        final_energies = np.array([self.energy_func(path[i]) for i in range(n_images)])
        return path, final_energies, ts_idx


class SequenceManager:
    """
    序列文件管理器（改编自 contour_sequence4 的序列处理）
    用于管理反应路径帧的序列索引
    """

    @staticmethod
    def increment_filename(filename):
        """
        递增文件名中的数字
        例：path_001.txt → path_002.txt
        """
        import re
        match = re.search(r'(\d+)(?!.*\d)', filename)
        if not match:
            return None
        num_str = match.group(1)
        new_num = str(int(num_str) + 1).zfill(len(num_str))
        return filename[:match.start()] + new_num + filename[match.end():]

    @staticmethod
    def generate_sequence(base_name, n_frames, extension='dat'):
        """生成序列文件名列表"""
        names = []
        for i in range(n_frames):
            names.append(f"{base_name}_{i:03d}.{extension}")
        return names


class SymmetryOperations:
    """
    分子对称性操作（改编自 T-puzzle 的旋转/反射变换）

    在过渡态搜索中，分子点群对称性可用于：
    1. 生成对称等价的反应路径
    2. 简化 Hessian 计算
    3. 验证过渡态的对称性
    """

    @staticmethod
    def rotate_coordinates(coords, axis, angle):
        """
        绕轴旋转坐标

        旋转矩阵（Rodrigues 公式）：
            R = I cosθ + (1-cosθ) nn^T + sinθ [n]_×
        """
        coords = np.asarray(coords, dtype=float)
        axis = np.asarray(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)

        c = np.cos(angle)
        s = np.sin(angle)
        n = axis

        R = np.array([
            [c + n[0] ** 2 * (1 - c), n[0] * n[1] * (1 - c) - n[2] * s,
             n[0] * n[2] * (1 - c) + n[1] * s],
            [n[1] * n[0] * (1 - c) + n[2] * s, c + n[1] ** 2 * (1 - c),
             n[1] * n[2] * (1 - c) - n[0] * s],
            [n[2] * n[0] * (1 - c) - n[1] * s, n[2] * n[1] * (1 - c) + n[0] * s,
             c + n[2] ** 2 * (1 - c)]
        ])

        return coords @ R.T

    @staticmethod
    def reflect_coordinates(coords, normal):
        """
        沿法向量反射坐标

        反射矩阵：
            R = I - 2 nn^T / |n|²
        """
        coords = np.asarray(coords, dtype=float)
        n = np.asarray(normal, dtype=float)
        n = n / np.linalg.norm(n)
        R = np.eye(3) - 2.0 * np.outer(n, n)
        return coords @ R.T

    @staticmethod
    def apply_c2v_symmetry(coords, z_axis=None):
        """
        应用 C_{2v} 点群对称操作生成等价构型
        返回 4 个对称等价构型
        """
        if z_axis is None:
            z_axis = np.array([0, 0, 1])

        configs = [coords.copy()]
        # C2 旋转
        configs.append(SymmetryOperations.rotate_coordinates(coords, z_axis, np.pi))
        # σv(xz) 反射
        configs.append(SymmetryOperations.reflect_coordinates(coords, np.array([0, 1, 0])))
        # σv'(yz) 反射
        configs.append(SymmetryOperations.reflect_coordinates(coords, np.array([1, 0, 0])))

        return configs
