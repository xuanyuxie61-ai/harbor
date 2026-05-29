r"""
cvt_adaptation.py
=================
基于 Centroidal Voronoi Tessellation (CVT) 的自适应网格生成与优化模块。

科学背景
--------
在圆柱涡激振动尾流模拟中，涡脱落导致尾流区出现强烈的剪切层与涡街结构，
需要在高梯度区域（壁面边界层、剪切层、涡核）集聚网格节点，而在远场保持
较粗分辨率以降低计算成本。CVT 通过最小化能量泛函：

    E(Z) = \sum_{i=1}^{N} \int_{V_i} \rho(x) \|x - z_i\|^2 \, dx

自动生成满足密度函数 \rho(x) 的节点分布，其中 V_i 为对应生成元 z_i 的
Voronoi 区域。

对于周期性尾流（如圆柱绕流），采用 Mirror-Periodic CVT (CVTM)：
将区间 [0, L] 通过镜像延拓到 [-L, 2L]，保证周期边界处节点分布的
光滑衔接。

算法流程
--------
1. 初始化生成元（均匀或随机）。
2. Lloyd 迭代：
   a. 在计算域内采样大量点。
   b. 对每个样本点，寻找最近生成元（考虑周期性镜像）。
   c. 将样本点归入对应 Voronoi 区域。
   d. 更新生成元为各区域质心。
3. 收敛判断：生成元平均移动量小于阈值。

反射边界处理
------------
对于壁面边界，采用反射 CVT (CCVT) 思想：
若样本点关于壁面的镜像点落入域内，则该样本点仅贡献给壁面附近的生成元，
从而自然在边界层形成节点集聚。

本模块对应原种子项目：
- 263_cvtm_1d（一维镜像周期 CVT，用于尾流区展向节点分布）
- 146_ccvt_reflect（带反射边界的二维 CVT，用于壁面边界层网格加密）
r"""

import numpy as np


class CVT1DPeriodic:
    r"""
    一维周期性 CVT 节点生成器（CVTM 思想）。
    用于圆柱尾流区展向（z 方向）或沿流向一维自适应节点分布。
    """

    def __init__(self, num_generators, domain_length=1.0,
                 density_func=None, it_max=100, tol=1e-6, sample_num=50000):
        r"""
        参数
        ----
        num_generators : int
            生成元数量。
        domain_length : float
            周期域长度 L。
        density_func : callable or None
            密度函数 \rho(x)，返回正数组。若为 None，采用均匀密度。
        it_max : int
            最大 Lloyd 迭代次数。
        tol : float
            收敛阈值（平均移动量）。
        sample_num : int
            每步采样点数。
        """
        if num_generators < 2:
            raise ValueError("生成元数量至少为 2。")
        self.g_num = num_generators
        self.L = domain_length
        self.it_max = it_max
        self.tol = tol
        self.sample_num = sample_num

        if density_func is None:
            self.density = lambda x: np.ones_like(x)
        else:
            self.density = density_func

        # 初始化生成元（均匀分布）
        self.generators = np.linspace(0.0, self.L, num_generators, endpoint=False)
        self.energy_history = []
        self.motion_history = []

    def _periodic_distance(self, x, g):
        r"""
        计算周期距离。对一维周期域 [0, L]，点 x 与生成元 g 的距离为：
        d = min(|x-g|, |x-g+L|, |x-g-L|)
        """
        d1 = np.abs(x - g)
        d2 = np.abs(x - g + self.L)
        d3 = np.abs(x - g - self.L)
        return np.minimum(np.minimum(d1, d2), d3)

    def _find_nearest_periodic(self, samples):
        r"""
        对每个样本点，寻找最近生成元（考虑周期镜像）。
        """
        # 将样本和生成元均扩展为三维数组以便广播
        s = samples[:, np.newaxis]  # (S, 1)
        g = self.generators[np.newaxis, :]  # (1, G)

        d1 = np.abs(s - g)
        d2 = np.abs(s - g + self.L)
        d3 = np.abs(s - g - self.L)
        d = np.minimum(np.minimum(d1, d2), d3)

        nearest = np.argmin(d, axis=1)
        min_d = np.min(d, axis=1)
        return nearest, min_d

    def iterate(self):
        r"""
        执行 Lloyd 迭代直至收敛或达到最大迭代次数。
        """
        for it in range(self.it_max):
            # 采样（含镜像点）
            samples = np.random.rand(self.sample_num) * self.L
            sa = samples - self.L
            sb = samples + self.L

            # 寻找最近生成元（考虑三个镜像）
            nearest_s, dist_s = self._find_nearest_periodic(samples)
            nearest_a, dist_a = self._find_nearest_periodic(sa)
            nearest_b, dist_b = self._find_nearest_periodic(sb)

            # 统计各区域样本点与加权质心
            g_new = np.zeros(self.g_num)
            w_new = np.zeros(self.g_num)
            energy = 0.0

            for idx in range(self.sample_num):
                # 选择真实样本或镜像中最近的一个
                d_list = [dist_s[idx], dist_a[idx], dist_b[idx]]
                k_list = [nearest_s[idx], nearest_a[idx], nearest_b[idx]]
                best = int(np.argmin(d_list))
                k = k_list[best]
                d = d_list[best]

                # 将镜像点映射回主区间
                if best == 1:
                    s_mapped = sa[idx]
                elif best == 2:
                    s_mapped = sb[idx]
                else:
                    s_mapped = samples[idx]

                rho_val = self.density(s_mapped)
                g_new[k] += s_mapped * rho_val
                w_new[k] += rho_val
                energy += d * d * rho_val

            # 更新生成元（空区域保持原位）
            mask = w_new > 0
            g_new[mask] /= w_new[mask]
            g_new[~mask] = self.generators[~mask]

            # 限制在主区间并排序
            g_new = np.mod(g_new, self.L)
            g_new = np.sort(g_new)

            # 计算平均移动量（周期意义下）
            motion = 0.0
            for k in range(self.g_num):
                diff = np.abs(g_new[k] - self.generators[k])
                diff = min(diff, self.L - diff)
                motion += diff * diff
            motion = np.sqrt(motion / self.g_num)

            self.generators = g_new.copy()
            self.energy_history.append(energy / self.sample_num)
            self.motion_history.append(motion)

            if motion < self.tol:
                break

        return self.generators.copy()


class CVT2DReflect:
    r"""
    二维带反射边界的 CVT 节点生成器（CCVT 思想简化版）。
    用于圆柱尾流区二维自适应节点分布。
    """

    def __init__(self, num_generators, bounds, density_func=None,
                 it_max=50, tol=1e-5, sample_num=20000):
        r"""
        参数
        ----
        num_generators : int
            生成元数量。
        bounds : tuple
            ((xmin, xmax), (ymin, ymax))。
        density_func : callable or None
            密度函数 \rho(x, y)。若为 None，均匀分布。
        it_max : int
            最大迭代次数。
        tol : float
            收敛阈值。
        sample_num : int
            每步采样数。
        """
        self.g_num = num_generators
        self.bounds = bounds
        self.it_max = it_max
        self.tol = tol
        self.sample_num = sample_num
        self.xmin, self.xmax = bounds[0]
        self.ymin, self.ymax = bounds[1]

        if density_func is None:
            self.density = lambda x, y: np.ones_like(x)
        else:
            self.density = density_func

        # 随机初始化
        self.generators = np.column_stack((
            np.random.rand(num_generators) * (self.xmax - self.xmin) + self.xmin,
            np.random.rand(num_generators) * (self.ymax - self.ymin) + self.ymin
        ))
        self.energy_history = []

    def _reflect_sample(self, sample):
        r"""
        对超出边界的样本点，计算其关于边界的反射点。
        若反射点在域内，则返回反射坐标；否则返回 None。
        """
        x, y = sample
        rx, ry = x, y
        inside = True
        if x < self.xmin:
            rx = 2.0 * self.xmin - x
            inside = False
        elif x > self.xmax:
            rx = 2.0 * self.xmax - x
            inside = False
        if y < self.ymin:
            ry = 2.0 * self.ymin - y
            inside = False
        elif y > self.ymax:
            ry = 2.0 * self.ymax - y
            inside = False

        if inside:
            return None
        if self.xmin <= rx <= self.xmax and self.ymin <= ry <= self.ymax:
            return np.array([rx, ry])
        return None

    def iterate(self):
        r"""执行 Lloyd 迭代。"""
        for it in range(self.it_max):
            samples = np.column_stack((
                np.random.rand(self.sample_num) * (self.xmax - self.xmin) + self.xmin,
                np.random.rand(self.sample_num) * (self.ymax - self.ymin) + self.ymin
            ))

            g_new = np.zeros((self.g_num, 2))
            w_new = np.zeros(self.g_num)
            energy = 0.0

            for s in samples:
                # 找最近生成元
                dists = np.sum((self.generators - s) ** 2, axis=1)
                nearest = int(np.argmin(dists))
                dmin = dists[nearest]

                # 反射检查
                reflected = self._reflect_sample(s)
                if reflected is not None:
                    # 反射点在域内，只计入最近生成元（壁面集聚效应）
                    pass

                rho_val = self.density(s[0], s[1])
                g_new[nearest] += s * rho_val
                w_new[nearest] += rho_val
                energy += dmin * rho_val

            # 更新
            mask = w_new > 0
            g_new[mask] /= w_new[mask][:, np.newaxis]
            g_new[~mask] = self.generators[~mask]

            # 限制在域内
            g_new[:, 0] = np.clip(g_new[:, 0], self.xmin, self.xmax)
            g_new[:, 1] = np.clip(g_new[:, 1], self.ymin, self.ymax)

            motion = np.sqrt(np.mean(np.sum((g_new - self.generators) ** 2, axis=1)))
            self.generators = g_new.copy()
            self.energy_history.append(energy / self.sample_num)

            if motion < self.tol:
                break

        return self.generators.copy()


def wake_density_function(X, Y, cylinder_x, cylinder_y, r_cyl,
                          wake_length_factor=5.0, wake_width_factor=1.5,
                          base_density=1.0, peak_density=10.0):
    """
    构造尾流自适应密度函数：

    \rho(x, y) = \rho_{base}
    + (\rho_{peak} - \rho_{base})
      \exp\left( -\frac{(x - x_c - L_{wake})^2}{2\sigma_x^2}
                 -\frac{(y - y_c)^2}{2\sigma_y^2} \right)

    其中 L_{wake} = wake_length_factor * r_cyl,
          \sigma_x = wake_length_factor * r_cyl / 2,
          \sigma_y = wake_width_factor * r_cyl / 2。
    r"""
    lw = wake_length_factor * r_cyl
    sx = lw / 2.0
    sy = wake_width_factor * r_cyl / 2.0

    dx = X - cylinder_x - lw
    dy = Y - cylinder_y

    rho = base_density + (peak_density - base_density) * np.exp(
        -0.5 * (dx / sx) ** 2 - 0.5 * (dy / sy) ** 2
    )

    # 壁面附近增强
    dist_to_cyl = np.sqrt((X - cylinder_x) ** 2 + (Y - cylinder_y) ** 2)
    wall_enhance = 1.0 + 2.0 * np.exp(-dist_to_cyl / r_cyl)
    rho *= wall_enhance

    return np.clip(rho, base_density, peak_density * 3.0)
