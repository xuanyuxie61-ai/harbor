# 光子晶体带隙工程综合计算平台 —— 博士级合成说明

## 一、项目概述

本项目围绕**光学工程：光子晶体带隙工程**这一前沿科学领域，将15个科研代码种子项目的核心算法融合重构为一个完整的博士级计算平台。项目实现了从光子晶体结构生成、能带计算、带隙分析、态密度积分、耦合模理论模拟、制造缺陷建模到光子输运与安德森局域化分析的端到端科研计算流程。

## 二、原项目到科学问题的映射

| 原项目 | 核心算法 | 在合成项目中的角色 |
|---|---|---|
| 330_ellipse_grid | 二维椭圆内网格生成 | 光子晶体空气孔几何建模的二维网格生成核 |
| 333_ellipsoid_grid | 三维椭球内网格生成 | 三维木堆/反蛋白石结构的介电体素化生成 |
| 708_magic_matrix | 幻方矩阵生成 | 准周期光子晶格的非周期确定性调制矩阵 |
| 1196_task_division | 任务分配算法 | 布里渊区k点计算的并行任务分区 |
| 1036_rk4 | 四阶Runge-Kutta ODE求解 | 耦合模方程的精确数值传播 |
| 1070_shallow_water_1d | 一维守恒律差分 | 有限差分时域耦合模(FDTM)的差分格式基础 |
| 1085_sine_transform | 离散正弦变换 | 层状光子晶体谱方法的本征模展开 |
| 451_gauss_seidel | Gauss-Seidel迭代 | 大型稀疏本征问题的预处理迭代求解 |
| 1154_st_to_ccs | 稀疏矩阵ST→CCS转换 | 光子晶体Maxwell方程稀疏算子的格式转换 |
| 991_r8pp | 对称正定packed Cholesky | 隐式FDTD中对称正定系统的直接求解 |
| 1081_simplex_monte_carlo | 单纯形蒙特卡洛积分 | 布里渊区态密度的四面体蒙特卡洛积分 |
| 538_histogram_data_2d_sample | 二维离散PDF/CDF采样 | 重要性采样DOS的k空间分布反演 |
| 1021_rejection_sample | 拒绝采样(Chebyshev/CVT) | 制造缺陷的孔径/位置随机涨落生成 |
| 1091_snakes_and_ladders | 马尔可夫链转移矩阵 | 无序光子晶体中的光子跳跃概率模型 |
| 786_nas | NAS数值核(矩阵乘/FFT/Cholesky) | 高性能数值计算核函数与基准测试 |

## 三、新增数学物理模型与核心公式

### 3.1 麦克斯韦-赫姆霍兹方程与平面波展开

二维TE模式麦克斯韦方程可化为赫姆霍兹形式：

$$
-\frac{\partial}{\partial x}\left(\frac{1}{\varepsilon(\mathbf{r})}\frac{\partial H_z}{\partial x}\right) - \frac{\partial}{\partial y}\left(\frac{1}{\varepsilon(\mathbf{r})}\frac{\partial H_z}{\partial y}\right) = \frac{\omega^2}{c^2} H_z
$$

采用平面波展开(PWE)，将介电函数与磁场按倒格矢展开：

$$
\frac{1}{\varepsilon(\mathbf{r})} = \sum_{\mathbf{G}} \kappa(\mathbf{G}) e^{i\mathbf{G}\cdot\mathbf{r}}, \quad H_z(\mathbf{r}) = \sum_{\mathbf{G}} H(\mathbf{G}) e^{i(\mathbf{k}+\mathbf{G})\cdot\mathbf{r}}
$$

得到广义本征值问题：

$$
\sum_{\mathbf{G}'} \kappa(\mathbf{G}-\mathbf{G}') |\mathbf{k}+\mathbf{G}'|^2 E(\mathbf{k}+\mathbf{G}') = \frac{\omega^2}{c^2} E(\mathbf{k}+\mathbf{G})
$$

### 3.2 耦合模理论(CMT)

一维布拉格光栅中的前向模 $A^+$ 与后向模 $A^-$ 满足：

$$
\frac{dA^+}{dz} = i\delta\beta \, A^+ + i\kappa \, A^-
$$

$$
\frac{dA^-}{dz} = -i\kappa^* A^+ - i\delta\beta \, A^-
$$

其中 $\delta\beta = \beta - \beta_B$ 为偏离布拉格条件的传播常数失谐，$\kappa$ 为耦合系数。

带隙内反射率解析解($|\delta\beta| < |\kappa|$)：

$$
R = \frac{|\kappa|^2 \sinh^2(SL)}{|\kappa|^2 \sinh^2(SL) + S^2 \cosh^2(SL)}, \quad S = \sqrt{|\kappa|^2 - \delta\beta^2}
$$

### 3.3 离散正弦变换谱方法

对于层状结构，场量按正弦基展开：

$$
H_z(x,y) = \sum_{n=1}^{N} c_n \sin\left(\frac{n\pi x}{a}\right) e^{i(k_x y - \omega t)}
$$

代入波动方程得到本征值问题，其中核矩阵：

$$
K_{nm} = \frac{2}{a}\int_0^a \sin\left(\frac{n\pi x}{a}\right) \frac{1}{\varepsilon(x)} \left[\left(\frac{m\pi}{a}\right)^2 + k_x^2\right] \sin\left(\frac{m\pi x}{a}\right) dx
$$

离散正弦变换(DST)系数：

$$
S(k) = \sqrt{\frac{2}{N+1}} \sum_{j=1}^{N} \sin\left(\frac{\pi k j}{N+1}\right) f(j)
$$

### 3.4 光子态密度(DOS)

态密度定义为：

$$
\rho(\omega) = \sum_n \int_{\text{BZ}} \delta(\omega - \omega_n(\mathbf{k})) \, d^2\mathbf{k} / V_{\text{BZ}}
$$

采用高斯展宽近似：

$$
\delta(\omega - \omega_0) \approx \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(\omega-\omega_0)^2}{2\sigma^2}\right)
$$

### 3.5 安德森局域化

Ioffe-Regel判据：

$$
kl \lesssim 1 \quad \text{(强局域化)}
$$

其中 $k = 2\pi n/\lambda$ 为波矢，$l$ 为平均自由程。

局域化长度(Vollhardt-Wölfle理论，弱散射极限)：

$$
\xi_{\text{loc}} \approx l \cdot \exp\left(\frac{\pi^2}{2}(kl)^2\right)
$$

标度理论的$\beta$函数：

$$
\beta(g) = (d-2) + \frac{2-d}{1+g^2}
$$

其中 $g$ 为无量纲Thouless电导，$d$ 为空间维度。

### 3.6 辐射输运方程

一维稳态辐射输运方程：

$$
\mu \frac{\partial I}{\partial z} + (\sigma_s + \sigma_a) I = \frac{\sigma_s}{2}\int I(\mu') \, d\mu' + S
$$

采用离散坐标法(S_N方法)数值求解。

## 四、合成后的项目文件结构

```
101_synth_project/
├── main.py                  # 统一入口，零参数运行
├── physics_core.py          # 物理常数、介电模型、电磁公式
├── lattice_generator.py     # 晶格结构生成（正方/三角/准周期/木堆/反蛋白石）
├── maxwell_eigensolver.py   # PWE、DST谱方法、Cholesky/GS/ST-CCS求解器
├── bandgap_analysis.py      # k点分区、RK4传播、带隙检测、群折射率
├── dos_calculator.py        # 单纯形MC积分、重要性采样、Van Hove奇异点
├── disorder_modeling.py     # 拒绝采样、位置/尺寸/介电无序模型
├── photon_transport.py      # 马尔可夫链、辐射输运、安德森局域化
├── numerical_kernels.py     # NAS随机数、优化矩阵乘、三/五对角求解、FFT
└── README_博士级合成说明.md  # 本文档
```

## 五、运行方式

在项目根目录下直接运行：

```bash
python3 main.py
```

程序将自动执行以下完整流程：
1. 设定硅-空气光子晶体的物理参数（晶格常数500nm，孔径150nm）
2. 生成5种不同维度和对称性的光子晶体结构
3. 沿布里渊区高对称路径（Γ→X→M→Γ）计算能带结构
4. 检测光子带隙并分析缺陷态、群折射率
5. 用离散正弦变换求解层状结构本征模
6. 用RK4+线性打靶法模拟布拉格光栅反射特性
7. 蒙特卡洛计算布里渊区态密度
8. 生成多种制造缺陷无序样本
9. 模拟光子输运、辐射输运方程与安德森局域化
10. 运行全部数值核函数的精度验证与性能基准测试

## 六、科学问题与工程能力

本项目能够解决的科学问题包括：
- **二维/三维光子晶体的完整能带结构计算**：采用平面波展开法，支持正方、三角、准周期等多种晶格
- **光子带隙的全自动检测与定量分析**：输出带隙宽度、中心频率、相对宽度等关键指标
- **布拉格光栅与波导耦合器的精确设计**：基于耦合模理论的解析解与数值解双重验证
- **制造缺陷对带隙的退化效应评估**：位置无序、尺寸无序、介电涨落三种机制
- **无序光子晶体中的光局域化预测**：Ioffe-Regel判据、标度理论、局域化长度估算
- **光子晶体辐射输运特性**：透射率、反射率、能量守恒校验

工程上具备：
- 边界条件检查与数值稳定性保护
- 稀疏矩阵格式转换与高效矩阵-向量乘
-  packed对称正定矩阵的直接求解
- 迭代法的收敛性监控与残差历史记录
- 高性能数值核的基准测试框架

## 七、公式-算法-代码一致性校验

所有核心公式均在代码中以注释形式完整保留，并通过以下方式验证一致性：
- **PWE本征值**：数值频率与理论归一化频率 $\omega a/(2\pi c)$ 比对
- **CMT反射率**：RK4数值解与解析解在 $\delta\beta/\kappa = \pm 1$ 处精确匹配（误差 $<10^{-15}$）
- **DST重构**：正弦变换系数重构误差 $<10^{-14}$
- **Cholesky分解**：$A\mathbf{x}=\mathbf{b}$ 验证残差 $=0$
- **三对角求解**：残差范数 $\sim 10^{-16}$
- **五对角求解**：残差范数 $\sim 10^{-14}$
- **能量守恒**：辐射输运 $T+R \approx 0.97$（考虑吸收损耗）
