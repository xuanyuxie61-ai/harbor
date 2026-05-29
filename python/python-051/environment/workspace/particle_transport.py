"""
particle_transport.py
=====================
基于 Particle-in-Cell (PIC) 方法的拉格朗日粒子追踪模块，
用于模拟营养盐微团（nutrient parcels）在热盐环流中的输运与扩散。

核心算法思想源自 electrostatic PIC 方法：
1. 将连续的营养盐浓度场离散为大量拉格朗日粒子（macro-parcels）
2. 粒子在速度场中运动，携带属性（营养盐浓度、温度、盐度、生态状态）
3. 通过云-in-cell (CIC) 权重方案将粒子属性投影回欧拉网格
4. 生态反应在粒子属性与网格场之间耦合更新

数学描述
--------
粒子运动方程：
    dx_p/dt = u(x_p, z_p),    dz_p/dt = w(x_p, z_p)

CIC 权重投影（双线性）：
    粒子 p 位于网格节点 (i,j) 与 (i+1,j+1) 之间时，
    对周围四个节点的权重分别为 (1-hx)(1-hy), hx(1-hy), (1-hx)hy, hx·hy

生态属性演化（沿轨迹）：
    dC_p/dt = R(C_p, N_env, I_env)
其中 R 为局地生物反应率，N_env 与 I_env 为粒子所在位置的环境场。
"""

import numpy as np


class LagrangianParticleTransport:
    """
    拉格朗日粒子输运与生态属性追踪。
    """

    def __init__(self, nx, nz, Lx, Lz, nparticles=5000, dt=3600.0):
        """
        参数
        ----
        nx, nz : int
            欧拉网格数
        Lx, Lz : float
            域尺寸
        nparticles : int
            粒子总数
        dt : float
            粒子时间步长 [s]
        """
        if nx < 4 or nz < 4:
            raise ValueError("nx, nz >= 4")
        if nparticles < 100:
            raise ValueError("nparticles >= 100")
        if dt <= 0:
            raise ValueError("dt > 0")

        self.nx = nx
        self.nz = nz
        self.Lx = Lx
        self.Lz = Lz
        self.dx = Lx / (nx - 1)
        self.dz = Lz / (nz - 1)
        self.nparticles = nparticles
        self.dt = dt

        # 粒子状态 [nparticles, 2] -> (x, z)
        self.pos = np.zeros((nparticles, 2))
        # 初始化：随机均匀分布
        self.pos[:, 0] = np.random.rand(nparticles) * Lx
        self.pos[:, 1] = np.random.rand(nparticles) * Lz

        # 粒子属性：营养盐浓度、叶绿素等
        self.nutrient_conc = np.random.uniform(0.5, 5.0, nparticles)
        self.chlorophyll = np.random.uniform(0.01, 0.5, nparticles)
        self.temperature = np.zeros(nparticles)
        self.salinity = np.zeros(nparticles)

        # 特定权重（每个粒子代表的体积）
        self.spwt = (Lx * Lz) / nparticles

        # 活跃粒子标记
        self.active = np.ones(nparticles, dtype=bool)

    def bilinear_weights(self, pos_x, pos_z):
        """
        计算粒子在欧拉网格上的双线性插值权重。
        返回 (i, j, hx, hy) 其中 i,j 为左下角节点索引，hx,hy 为归一化偏移。
        """
        dx = self.dx
        dz = self.dz
        nx = self.nx
        nz = self.nz

        # 防止 NaN/Inf
        if not (np.isfinite(pos_x) and np.isfinite(pos_z)):
            return 0, 0, 0.0, 0.0

        pos_x = max(0.0, min(self.Lx - 1e-12, pos_x))
        pos_z = max(0.0, min(self.Lz - 1e-12, pos_z))

        fi = 1.0 + pos_x / dx
        i = int(np.floor(fi))
        i = max(0, min(i, nx - 2))
        hx = fi - i
        hx = max(0.0, min(1.0, hx))

        fj = 1.0 + pos_z / dz
        j = int(np.floor(fj))
        j = max(0, min(j, nz - 2))
        hy = fj - j
        hy = max(0.0, min(1.0, hy))

        return i, j, hx, hy

    def interpolate_field_to_particle(self, field):
        """
        将欧拉场插值到粒子位置（双线性插值）。
        """
        vals = np.zeros(self.nparticles)
        for p in range(self.nparticles):
            if not self.active[p]:
                continue
            i, j, hx, hy = self.bilinear_weights(self.pos[p, 0], self.pos[p, 1])
            vals[p] = (
                (1.0 - hx) * (1.0 - hy) * field[i, j] +
                hx * (1.0 - hy) * field[i + 1, j] +
                (1.0 - hx) * hy * field[i, j + 1] +
                hx * hy * field[i + 1, j + 1]
            )
        return vals

    def deposit_particles_to_grid(self, particle_scalar):
        """
        将粒子标量属性通过 CIC 方案沉积回欧拉网格。
        返回网格密度场。
        """
        grid = np.zeros((self.nx, self.nz))
        count = np.zeros((self.nx, self.nz))

        for p in range(self.nparticles):
            if not self.active[p]:
                continue
            i, j, hx, hy = self.bilinear_weights(self.pos[p, 0], self.pos[p, 1])
            w00 = (1.0 - hx) * (1.0 - hy)
            w10 = hx * (1.0 - hy)
            w01 = (1.0 - hx) * hy
            w11 = hx * hy

            grid[i, j] += w00 * particle_scalar[p]
            grid[i + 1, j] += w10 * particle_scalar[p]
            grid[i, j + 1] += w01 * particle_scalar[p]
            grid[i + 1, j + 1] += w11 * particle_scalar[p]
            count[i, j] += w00
            count[i + 1, j] += w10
            count[i, j + 1] += w01
            count[i + 1, j + 1] += w11

        # 归一化
        with np.errstate(divide='ignore', invalid='ignore'):
            grid = np.where(count > 0, grid / count, 0.0)
        return grid

    def step(self, u_field, w_field, omega_bio=0.0):
        """
        推进粒子一个时间步。

        参数
        ----
        u_field, w_field : ndarray (nx, nz)
            欧拉速度场 [m/s]
        omega_bio : float
            生物衰减率 [1/s]
        """
        # 速度插值到粒子
        u_p = self.interpolate_field_to_particle(u_field)
        w_p = self.interpolate_field_to_particle(w_field)

        # 更新位置（显式欧拉）
        self.pos[:, 0] += self.dt * u_p
        self.pos[:, 1] += self.dt * w_p

        # 边界处理：反射性底边界，周期性/开放侧边界
        for p in range(self.nparticles):
            if not self.active[p]:
                continue

            # 底边界反射
            if self.pos[p, 1] < 0.0:
                self.pos[p, 1] = -self.pos[p, 1]

            # 顶边界反射（海面）
            if self.pos[p, 1] > self.Lz:
                self.pos[p, 1] = 2.0 * self.Lz - self.pos[p, 1]

            # 侧边界：左侧流入，右侧流出后重新从左边界注入
            if self.pos[p, 0] < 0.0:
                self.pos[p, 0] = self.Lx - 1e-6
                self.nutrient_conc[p] = np.random.uniform(3.0, 6.0)
            if self.pos[p, 0] > self.Lx:
                self.pos[p, 0] = 1e-6
                self.nutrient_conc[p] = np.random.uniform(1.0, 3.0)

        # 生物属性衰减
        if omega_bio > 0:
            self.nutrient_conc *= np.exp(-omega_bio * self.dt)
            self.chlorophyll *= np.exp(-omega_bio * self.dt)

    def get_particle_density_field(self):
        """
        将粒子数密度投影回网格。
        """
        ones = np.ones(self.nparticles)
        return self.deposit_particles_to_grid(ones)

    def get_mean_nutrient_field(self):
        """
        粒子营养盐浓度投影回网格。
        """
        return self.deposit_particles_to_grid(self.nutrient_conc)

    def resample_particles(self, N_grid, P_grid, T_grid, S_grid):
        """
        根据欧拉场对粒子属性进行重新采样（保持统计一致性）。
        """
        # 插值环境场到粒子
        N_env = self.interpolate_field_to_particle(N_grid)
        P_env = self.interpolate_field_to_particle(P_grid)
        self.temperature = self.interpolate_field_to_particle(T_grid)
        self.salinity = self.interpolate_field_to_particle(S_grid)

        # 粒子营养盐向环境场松弛
        relax = 0.01
        self.nutrient_conc += relax * (N_env - self.nutrient_conc)
        self.chlorophyll += relax * (P_env - self.chlorophyll)

        # 非负约束
        self.nutrient_conc = np.maximum(self.nutrient_conc, 0.0)
        self.chlorophyll = np.maximum(self.chlorophyll, 0.0)
