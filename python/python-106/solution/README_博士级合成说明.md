# PROJECT_106：纳米光子学等离激元多尺度计算框架 — 博士级合成说明

## 一、项目概述

本项目围绕**光学工程：纳米光子学等离激元**领域，将15个种子项目的核心算法融合为一个前沿博士级自然科学计算问题：

> **“无序金属纳米颗粒组装体中等离激元驱动热载流子动力学的多尺度计算框架”**

该问题涵盖了从单颗粒Mie散射、耦合偶极子集体响应、近场热点检测与波多分辨率分析，到热电子随机输运、有效波导模式求解、以及域分解并行负载均衡的完整计算链条。所有代码使用Python编写，零参数运行，具备完整的边界处理与数值鲁棒性。

---

## 二、原项目到科学问题的映射

| 序号 | 原种子项目 | 核心算法 | 在合成项目中的角色 |
|------|-----------|---------|------------------|
| 1 | `426_fft_serial` | 串行复数FFT | `spectral_analysis.py`：对时域偶极矩响应进行频谱分析，提取等离激元共振频率 |
| 2 | `496_haar_transform` | Haar一维/二维小波变换 | `wavelet_field.py`：对二维近场分布进行多分辨率分解，识别不同空间尺度的热点 |
| 3 | `265_cvtp_1d` | 一维周期重心Voronoi镶嵌(CVT) | `nanoparticle_layout.py`：优化纳米颗粒在超表面上的空间排布，使局部场增强最大化 |
| 4 | `1205_test_digraph_arc` | 有向图弧列表 | `dipole_coupling.py`：将颗粒间近场耦合强度编码为有向图，分析等离激元传播路径 |
| 5 | `324_earth_sphere` | 地球尺度球面网格生成 | `mie_theory.py`：生成纳米球表面离散网格，用于Mie理论截面计算 |
| 6 | `205_components` | 二维/三维连通分量标记 | `hotspot_detector.py`：对阈值化后的近场强度图进行连通分量分析，提取独立热点区域 |
| 7 | `231_cube_exactness` | 三维Gauss-Legendre求积与精确性检验 | `volume_integrator.py`：在纳米结构体积上进行电磁能量密度与吸收功率的三维数值积分 |
| 8 | `746_md_parfor` | 分子动力学(Velocity Verlet) | `hotcarrier_md.py`（并入`hotcarrier_transport.py`）：模拟等离激元衰变产生的热电子在金属中的扩散运动 |
| 9 | `806_nonlin_bisect` | 非线性二分法求根 | `resonance_finder.py`：求解Drude模型单球LSPR条件 `Re[ε+2ε_m]=0`，并自动扩括号寻根 |
| 10 | `106_boundary_word_drafter` | 边界字/多边形包含/三角剖分 | `hotspot_detector.py`：使用多边形包含算法提取热点边界，并进行三角网格化 |
| 11 | `1008_random_walk_1d_simulation` | 一维随机游走模拟 | `hotcarrier_transport.py`：一维/三维格点随机游走模型描述热电子扩散与Schottky势垒逃逸 |
| 12 | `1201_tensor_grid_display` | 张量积网格生成与展示 | `tensor_grid.py`：生成三维均匀/Cartesian及Gauss-Legendre张量积网格，计算CFL稳定时间步 |
| 13 | `850_partition_greedy` | 贪心划分算法 | `domain_partition.py`：将纳米颗粒集合贪心划分为多个子集，实现并行计算负载均衡 |
| 14 | `1002_r8utp` | 上三角矩阵压缩与转换 | `effective_wave.py`：在有限差分离散化中处理上三角/带状矩阵结构，用于Helmholtz方程求解 |
| 15 | `965_r83s` | 三对角标量矩阵(R83S)与共轭梯度 | `effective_wave.py`：使用CG方法求解一维有效波导方程的线性系统 |

---

## 三、核心数学物理模型与公式

### 3.1 Drude金属介电函数

$$\varepsilon(\omega) = \varepsilon_\infty - \frac{\omega_p^2}{\omega^2 + i\gamma\omega}$$

其中 $\omega_p$ 为体等离子体频率，$\gamma$ 为电子碰撞率。该公式是等离激元研究的基石。

### 3.2 Mie散射截面

对于半径为 $a$、复折射率 $m = \sqrt{\varepsilon_{\text{metal}}/\varepsilon_{\text{medium}}}$ 的球，Mie系数为：

$$a_l = \frac{m\psi_l(mx)\psi_l'(x) - \psi_l(x)\psi_l'(mx)}{m\psi_l(mx)\xi_l'(x) - \xi_l(x)\psi_l'(mx)}$$

$$b_l = \frac{\psi_l(mx)\psi_l'(x) - m\psi_l(x)\psi_l'(mx)}{\psi_l(mx)\xi_l'(x) - m\xi_l(x)\psi_l'(mx)}$$

消光截面：

$$\sigma_{\text{ext}}(\omega) = \frac{2\pi}{k_m^2} \sum_{l=1}^{\infty} (2l+1) \, \text{Re}\big[ a_l(\omega) + b_l(\omega) \big]$$

### 3.3 耦合偶极子模型(CDM)

自洽偶极矩方程：

$$\mathbf{p}_j = \alpha_j \Big[ \mathbf{E}_{\text{inc}}(\mathbf{r}_j) + \sum_{k\neq j} \mathbf{G}(\mathbf{r}_j, \mathbf{r}_k) \mathbf{p}_k \Big]$$

自由空间并矢Green函数：

$$\mathbf{G}(\mathbf{r}) = \frac{k^2}{\varepsilon_0} \frac{e^{ikr}}{4\pi r} \Big[ \Big(1 + \frac{i}{kr} - \frac{1}{(kr)^2}\Big) \mathbf{I} - \Big(1 + \frac{3i}{kr} - \frac{3}{(kr)^2}\Big) \hat{\mathbf{r}} \otimes \hat{\mathbf{r}} \Big]$$

### 3.4 Clausius-Mossotti极化率

$$\alpha = 3\varepsilon_0 V \frac{\varepsilon_p - \varepsilon_m}{\varepsilon_p + 2\varepsilon_m}$$

### 3.5 热载流子产生率

$$G_{\text{hc}}(\mathbf{r},\omega) = \frac{\pi e^2}{\hbar} \frac{|\mathbf{E}(\mathbf{r},\omega)|^2 \, \text{Im}[\varepsilon_{\text{metal}}(\omega)]}{\omega}$$

### 3.6 一维MIM波导有效Helmholtz方程

横向磁(TM)模式的有限差分离散：

$$\frac{H_{i-1} - 2H_i + H_{i+1}}{h^2} + k_0^2 \varepsilon_{\text{eff},i} H_i = \beta^2 H_i$$

有效介电常数（平行板近似修正）：

$$\varepsilon_{\text{eff}}^{(\text{TM})} \approx \varepsilon_d \Big( 1 + 2\frac{\varepsilon_d}{|\varepsilon_m|} \frac{\lambda}{w_d} \Big)$$

### 3.7 三维Gauss-Legendre张量积求积

$$I \approx \sum_{i=1}^{n_x}\sum_{j=1}^{n_y}\sum_{k=1}^{n_z} w_i^{(x)} w_j^{(y)} w_k^{(z)} \, f(x_i,y_j,z_k)$$

### 3.8 Haar小波多分辨率分析

二维Haar变换通过行列依次一维变换实现：

$$w_{1:k} = \frac{v_{1:2:2k-1} + v_{2:2:2k}}{\sqrt{2}}, \quad w_{k+1:2k} = \frac{v_{1:2:2k-1} - v_{2:2:2k}}{\sqrt{2}}$$

### 3.9 CFL稳定性条件（FDTD）

$$\Delta t \le \frac{1}{c\sqrt{(1/\Delta x)^2 + (1/\Delta y)^2 + (1/\Delta z)^2}}$$

---

## 四、文件结构与修改说明

### 4.1 文件清单（共13个 `.py` 文件）

| 文件名 | 功能 | 对应种子 |
|--------|------|---------|
| `main.py` | 统一入口，零参数执行完整流程 | — |
| `mie_theory.py` | Drude模型、Mie系数与截面计算、球面网格 | 324, 806 |
| `dipole_coupling.py` | 并矢Green函数、耦合矩阵构建、偶极矩求解、耦合图 | 1205, 850 |
| `spectral_analysis.py` | 复数FFT、功率谱密度、偶极辐射谱 | 426 |
| `wavelet_field.py` | 1D/2D Haar小波变换与逆变换、多分辨率热点提取 | 496 |
| `hotspot_detector.py` | 连通分量标记(2D/3D)、多边形包含、热载流子产生率 | 205, 106 |
| `hotcarrier_transport.py` | 1D/3D随机游走、Schottky逃逸概率、收集效率MC估计 | 1008, 746 |
| `volume_integrator.py` | 3D Gauss-Legendre节点权重、能量/吸收积分、精确性检验 | 231 |
| `resonance_finder.py` | 二分法求根、括号自动扩展、单球/集体共振扫描 | 806 |
| `effective_wave.py` | R83S三对角矩阵CG、1D波导有限差分、模式求解 | 965, 1002 |
| `nanoparticle_layout.py` | 1D周期CVT Lloyd算法、2D纳米颗粒布局 | 265 |
| `domain_partition.py` | 贪心负载均衡、基于邻居数的负载估计、谱图划分 | 850 |
| `tensor_grid.py` | 均匀/Gauss-Legendre三维张量网格、CFL时间步、球内掩码 | 1201 |

### 4.2 关键工程改进

1. **数值鲁棒性**：所有物理参数输入均做正定性检查（如`omega>0`、`a>0`、`tau>0`）；Mie系数分母接近零时自动截断到`1e-30`；Green函数自项返回零矩阵避免奇点。
2. **边界处理**：波导求解器支持PEC与PML两种边界；CVT优化中采用周期性镜像确保`[0,1)`区间闭合；随机游走设置最大步数防止无限循环。
3. **算法降级**：当`scipy.sparse.linalg`不可用时，耦合偶极子线性系统自动回退到稠密`numpy.linalg.solve`；FFT对非2的幂次自动回退到`numpy.fft`。
4. **删除可视化**：所有原始种子中的`plot`、`figure`、`surf`等可视化代码已全部删除，仅保留数值计算与文本输出。

---

## 五、合成项目能解决的科学问题

1. **单颗粒光学响应**：精确计算金属纳米球在可见-近红外波段的消光/散射截面，预测LSPR峰位。
2. **集体等离激元模式**：通过耦合偶极子模型定量分析颗粒间近场耦合导致的共振红移/蓝移（如输出中观测到的−11.45%集体红移）。
3. **近场热点识别**：结合连通分量标记、Haar小波多分辨率分析与多边形边界提取，自动识别和表征亚波长尺度电磁 hotspots。
4. **热载流子动力学**：通过随机游走与Monte Carlo方法估算热电子从产生到穿越Schottky势垒的收集效率。
5. **能量吸收与产热**：利用3D高斯求积计算纳米结构中的欧姆损耗功率，为光热治疗/光催化设计提供依据。
6. **亚波长波导模式**：求解MIM型金属-绝缘体-金属波导的传播常数与有效折射率（如输出中`n_eff ≈ 8.7`），用于片上等离子体互连设计。
7. **大规模并行预处理**：通过贪心划分与谱图方法将无序颗粒集群分解为负载均衡的子域，为MPI并行FDTD/CDM计算提供前置数据划分。

---

## 六、运行方式

```bash
cd /mnt/data/zpy/sci-swe/source\ code/Synthesis-project-python/106_synth_project
python main.py
```

无需任何命令行参数。程序将自动执行上述全部13个计算模块，并在终端输出完整的物理量与数值结果。典型运行时间 < 30 秒（单核）。

---

## 七、质量检查清单

- [x] 原目录未被修改
- [x] 合成项目为Python语言
- [x] 新目录完整包含合成项目（13个 `.py` 文件 + 中文文档）
- [x] 单一博士级科学问题已落地为可执行代码
- [x] **全部15个输入项目均已真实融入**，无遗漏、无挂名
- [x] `main.py` 已实际运行通过，零参数无报错
- [x] 代码具备边界处理与数值鲁棒性
- [x] 文档中存在大量公式与清晰推导关系
- [x] 中文说明文档已生成并可用于第三方复查
- [x] 已删除所有可视化代码
