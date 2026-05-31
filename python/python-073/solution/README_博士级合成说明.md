# 高超声速边界层转捩预测 — 博士级科研代码合成说明

## 一、项目概述

本项目围绕**计算流体力学：高超声速边界层转捩**这一前沿科学问题，将 15 个原始科研代码项目的核心算法融合重构为一个完整的博士级科研计算流程。

**科学问题定位**：
高超声速飞行器（再入舱、高超声速巡航导弹等）在大气层内飞行时，边界层从层流转变为湍流（转捩）会急剧增加热流与摩阻，准确预测转捩位置是气动热设计的关键瓶颈。本项目实现了从边界层网格生成、基流求解、线性稳定性分析（LST）、$e^N$ 转捩预测到不确定性量化的端到端计算流程。

---

## 二、原项目到科学问题的映射

| 原项目 | 核心算法 | 合成后角色 | 融入文件 |
|--------|----------|-----------|---------|
| 1330_triangulation | Delaunay 三角剖分、邻居搜索、边界检测、网格质量评估 | 高超声速边界层计算域的曲面三角化与壁面法向网格加密 | `mesh_generator.py` |
| 1123_sphere_llt_grid | 球面 LLT 经纬度网格生成 | 扰动波矢方向在球面上的离散化（用于三维稳定性分析） | `mesh_generator.py` |
| 512_heated_plate | 稳态热方程 Jacobi 迭代求解 | 可压缩边界层能量方程的迭代松弛求解（含粘性耗散源项与迎风离散） | `thermal_solver.py` |
| 1130_sphere_triangle_quad | 球面三角形数值积分（多种求积法则） | 波矢空间方向积分的球面求积 | `spectral_integrator.py` |
| 911_prime_factors | 整数质因数分解 | 最优 Chebyshev 展开阶数选取（FFT 友好长度） | `utils.py` |
| 1199_tec_to_vtk | TECPLOT ↔ VTK 数据格式转换 | CFD 计算结果的格式转换与报告生成 | `data_io.py` |
| 587_imshow_numeric | 数值归一化与边界处理 | 数值场归一化、安全除法、异常值处理 | `utils.py` |
| 164_chebyshev1_exactness | Gauss-Chebyshev 求积精确度验证 | 边界层剖面积分、$e^N$ 放大因子积分的高精度数值验证 | `spectral_integrator.py` |
| 610_jordan_matrix | 随机 Jordan 矩阵生成与特征值分析 | 稳定性算子的 Jordan 分解与非模态增长分析 | `stability_analysis.py` |
| 1420_xy_io | XY 坐标数据读写 | 基流剖面、转捩位置的二维数据 I/O | `data_io.py`、`mesh_generator.py` |
| 417_fem3d_pack | TET4 四面体线性基函数、Jacobian、坐标变换 | 三维扰动方程的有限元离散与质量/刚度矩阵构造 | `fem_basis.py` |
| 555_hyperball_positive_distance | 高维超球随机采样与距离统计 | 多维参数空间的均匀性评估与采样验证 | `monte_carlo_sampler.py` |
| 534_high_card_simulation | 最优停止问题（秘书问题）蒙特卡洛模拟 | 参数空间中序贯最优采样策略 | `monte_carlo_sampler.py` |
| 1366_tsp_moler | 旅行商问题启发式求解（2-opt、插入法） | 多展向站位转捩前沿曲线的光滑性优化 | `transition_predictor.py` |
| 502_hand_data | 轮廓数据提取与坐标记录 | 转捩前沿几何轮廓的数据提取 | `data_io.py` |

---

## 三、新增数学物理模型与核心公式

### 3.1 可压缩边界层自相似能量方程

在相似坐标 $\eta = y \sqrt{u_e / (\nu_e x)}$ 下，稳态能量方程为：

$$
\frac{d}{d\eta}\left(\frac{\mu}{\Pr} \frac{dT}{d\eta}\right) + \frac{1}{2} f \frac{dT}{d\eta} + (\gamma-1) M_a^2 \mu \left(\frac{du}{d\eta}\right)^2 = 0
$$

边界条件：
- 壁面：$T(0) = T_w / T_e$
- 远场：$T(\infty) = 1$

### 3.2 Blasius 相似方程

不可压缩边界层相似解满足：

$$
f''' + \frac{1}{2} f f'' = 0, \quad f(0)=f'(0)=0, \; f'(\infty)=1
$$

壁面曲率 $f''(0) \approx 0.332057$。

### 3.3 Sutherland 粘性定律

$$
\frac{\mu}{\mu_{\text{ref}}} = \left(\frac{T}{T_{\text{ref}}}\right)^{3/2} \frac{T_{\text{ref}} + S}{T + S}
$$

其中 $S = 110.4\,\text{K}$ 为 Sutherland 常数。

### 3.4 线性稳定性理论 (LST)

对平行流假设下的小扰动 $\tilde{q} = \hat{q}(y) \exp[i(\alpha x + \beta z - \omega t)]$，可压缩 Orr-Sommerfeld 方程组离散为广义特征值问题：

$$
\mathbf{A} \, \hat{\mathbf{q}} = \omega \, \mathbf{B} \, \hat{\mathbf{q}}
$$

其中状态向量 $\hat{\mathbf{q}} = [\hat{u}, \hat{v}, \hat{\theta}, \hat{p}]^T$。

### 3.5 Jordan 分解与非模态增长

对离散算子进行 Jordan 分解：

$$
\mathbf{M} = \mathbf{P} \mathbf{J} \mathbf{P}^{-1}
$$

瞬态增长上界：

$$
G(t) \le \|\mathbf{P}\| \cdot \|\mathbf{P}^{-1}\| \cdot \exp[\text{Re}(\omega_{\max}) t]
$$

### 3.6 $e^N$ 转捩预测方法

扰动放大因子：

$$
N(x) = \ln\frac{A}{A_0} = -\int_{x_0}^{x} \alpha_i(s) \, ds
$$

当 $N(x_t) = N_{\text{cr}}$（典型值 $7 \sim 11$）时判定转捩。

### 3.7 球面三角形求积

对球面三角形 $\Delta_S$ 上的积分，采用基于 L'Huilier 定理的面积计算：

$$
\tan\frac{E}{4} = \sqrt{\tan\frac{s}{2} \tan\frac{s-a}{2} \tan\frac{s-b}{2} \tan\frac{s-c}{2}}
$$

其中 $E$ 为球面角盈，面积 $= E \cdot R^2$。

### 3.8 Chebyshev 谱微分矩阵

Gauss-Lobatto 节点 $x_j = \cos(j\pi/N)$ 上的微分矩阵：

$$
D_{ij} = \frac{c_i}{c_j} \frac{(-1)^{i+j}}{x_i - x_j}, \quad i \ne j
$$

$$
D_{ii} = -\frac{x_i}{2(1-x_i^2)}, \quad 1 \le i \le N-1
$$

其中 $c_0 = c_N = 2$，$c_j = 1$（$1 \le j \le N-1$）。

---

## 四、文件结构说明

```
073_synth_project/
├── main.py                      # 统一入口，零参数运行
├── mesh_generator.py            # 边界层网格生成 + 球面波矢离散
├── fem_basis.py                 # 3D TET4 有限元基函数与矩阵组装
├── thermal_solver.py            # 可压缩边界层能量方程求解
├── stability_analysis.py        # LST 稳定性算子 + Jordan 分析
├── spectral_integrator.py       # Chebyshev 求积 + 球面积分 + e^N 积分
├── transition_predictor.py      # e^N 转捩预测 + 多站位优化
├── monte_carlo_sampler.py       # LHS 采样 + 序贯采样 + 不确定性传播
├── data_io.py                   # 数据读写 + 报告生成
├── utils.py                     # 工具函数（Blasius、Sutherland、Chebyshev 节点等）
└── README_博士级合成说明.md     # 本文档
```

---

## 五、如何运行

### 环境要求
- Python 3.8+
- NumPy

### 运行方式

```bash
cd 073_synth_project
python main.py
```

程序将自动执行以下流程：
1. 生成边界层结构化网格（3000 节点，5782 三角形）
2. 求解可压缩边界层温度场与速度剖面
3. 验证有限元基函数 partition of unity 与矩阵条件数
4. 构建 Chebyshev 谱稳定性算子并求解特征值
5. 执行 Jordan 分解评估非模态增长上界
6. 沿流向积分 $e^N$ 放大因子
7. 预测单站位与多展向站位转捩位置
8. 执行蒙特卡洛不确定性量化（300 样本）
9. 输出计算报告与数据文件

### 输出文件
- `baseflow_profile.xy`：基流速度剖面
- `temperature_profile.xy`：温度剖面
- `eigenvalue_spectrum.dat`：特征值谱数据
- `transition_report.txt`：综合计算报告
- `wavevectors.dat`：球面波矢方向离散点

---

## 六、关键数值方法与鲁棒性设计

1. **网格拉伸**：壁面法向几何拉伸，第一层高度满足 $y^+ < 1$ 约束。
2. **迎风离散**：能量方程对流项采用一阶迎风格式，保证 Jacobi 迭代的数值稳定性。
3. **安全除法**：`safe_divide` 函数避免被零除，广播处理标量与数组混用。
4. **Chebyshev 阶数优化**：基于质因数分解选取 FFT 友好的截断阶数。
5. **边界条件硬编码**：通过行替换直接施加 Dirichlet/Neumann 条件。
6. **广义特征值奇异处理**：当 $\mathbf{B}$ 奇异时，采用正则化伪逆替代。

---

## 七、科学难度说明

本项目涵盖以下博士级前沿内容：
- **可压缩边界层理论**：Crocco-Busemann 能量积分、Sutherland 粘性定律、自相似变换
- **线性稳定性理论**：可压缩 Orr-Sommerfeld 方程组、时间/空间模式转换、特征值追踪
- **非模态稳定性**：Jordan 块结构分析、瞬态增长上界估计、条件数与伪谱理论
- **转捩预测工程方法**：$e^N$ 方法、Mack 第二模态、感受性系数、壁面粗糙度修正
- **不确定性量化**：拉丁超立方采样、序贯最优采样、高维参数空间统计传播
- **谱方法**：Chebyshev-Gauss-Lobatto 节点、谱微分矩阵、快速变换优化
- **有限元方法**：TET4 等参映射、一致质量矩阵、刚度矩阵组装

---

*项目合成日期：2026-05-04*
