# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

```markdown
# 引力波数值相对论模拟与贝叶斯参数推断系统

## 项目概述
本项目实现一个面向双黑洞并合事件的引力波信号模拟与参数推断框架。系统包含七个主要功能模块，覆盖从初始数据构造、轨道动力学演化、波形生成、探测器响应、贝叶斯后验估计到数值稳定性验证的完整流程。所有模块通过 `main.py` 统一调度，其余 `.py` 文件提供底层实现，后续将只保留 `main.py`，需要根据本描述重新实现各缺失模块。

## 文件与模块职责

### `main.py`
项目入口，编排整个模拟流水线。按顺序调用各模块执行以下阶段：
1. 双黑洞初始数据构造（Brill‑Lindquist + 共形平坦近似）
2. 轨道动力学数值演化（后牛顿辐射反作用）
3. 引力波波形生成（inspiral‑merger‑ringdown）
4. 多探测器网络响应与天球定位
5. 贝叶斯参数估计（MCMC + 方形求积）
6. 数值稳定性测试套件
7. 辅助科学计算（Legendre 多项式、有理近似、有效自旋等）

### `binary_black_hole.py`
双黑洞系统物理模型。定义 `BinaryBlackHole` 类，封装质量、自旋、光度距离、倾角等天体物理参数，提供单位转换（从太阳质量/秒等物理单位到几何单位制），计算啁啾质量、对称质量比、ISCO 频率、旋进时间、轨道分离、轨道速度、能量通量和应变振幅等物理量。此外包含辅助函数：
- `mass_ratio_to_components`：由质量比和总质量分解分量质量
- `effective_spin`：计算有效自旋参数
- `precession_spin`：计算进动有效自旋

### `numerical_relativity.py`
数值相对论时空演化模块。提供：
- 双星轨道导数（含 2.5PN 辐射反作用）和轨道演化函数 `evolve_binary_orbit`
- 共形因子解析解 `conformal_factor_brill_lindquist` 及 ADM 度规组件
- 最大切片条件下的时移函数求解器
- 规范波传播测试
- 数值稳定性测试套件 `run_stability_tests`，包含 Robertson 刚性系统、Burgers 激波、锯齿波振子、规范波精度四项测试
- 并合产物质量与自旋的拟合估算 `final_mass_spin`

### `sparse_solver.py`
稀疏矩阵与线性系统求解，用于求解 ADM 方程离散化后的约束方程。主要内容：
- `SparseMatrixHB` 类：Harwell‑Boeing 风格稀疏矩阵存储，支持从 CSR 格式构建、矩阵向量乘法
- 二维双调和算子 13 点有限差分模板及矩阵构建函数 `build_biharmonic_matrix`
- `solve_initial_data_brill_lindquist`：基于双调和正则化求解共形平坦初始数据，返回共形因子和 ADM 质量
- `compute_extrinsic_curvature`：计算共形平坦度规下的外曲率分量

### `teukolsky.py`
Teukolsky 方程与黑洞微扰理论。提供：
- Weierstrass‑Durand‑Kerner (WDK) 多项式求根算法 `wdk_roots`
- 准正规模特征多项式构造与频率求解器 `solve_qnm_frequencies`
- Teukolsky 径向方程有效势与数值积分函数 `teukolsky_radial_integration`
- 引力波辐射光度估算 `gravitational_wave_luminosity`

### `waveform.py`
引力波波形生成模块。包含：
- 移位 Legendre 多项式 `shifted_legendre_polynomial` 及自旋权重‑2 球谐函数近似
- Gauss‑Chebyshev Type 2 求积规则及基于该规则的波形内积计算 `waveform_inner_product_chebyshev`
- Gauss‑Patterson 嵌套求积规则（支持自适应积分） `patterson_quadrature`, `adaptive_patterson_integral`
- 后牛顿 inspiral 波形 `post_newtonian_waveform`
- Ringdown 波形 `ringdown_waveform`
- 完整的 IMR 风格波形 `full_imrphenom_waveform`（通过窗口函数平滑拼接）
- 匹配滤波信噪比计算 `matched_filter_snr`

### `detector.py`
引力波探测器响应与天球定位。包含：
- 探测器响应张量 `detector_tensor` 和天线方向函数 `antenna_pattern_functions`
-
