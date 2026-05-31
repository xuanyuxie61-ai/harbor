# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 项目描述：光纤非线性脉冲传输（python-103）

本项目是一个面向光纤超短脉冲非线性传输的数值仿真与分析框架，涵盖几何建模、频域/时域谱方法、稀疏线性求解、蒙特卡洛不确定性量化、MCMC参数反演、根查找与相位编码等多个计算模块。所有模块由 `main.py` 统一调度，通过零参数运行完成全流程演示。

## 文件清单与职责

### `main.py`
主入口文件。整合所有子模块，依次运行 Jacobi 谱方法、脉冲重叠积分、光纤几何与有效模场面积、蒙特卡洛采样与球面积分、噪声模型、GMRES 稀疏求解、DREAM MCMC 参数反演、Laguerre 根查找、相位编码与整数搜索、以及 GNLSE 脉冲传输演示。不包含独有的业务逻辑，主要负责调用各模块的公开接口、打印结果。

### `fiber_geometry.py`
负责光纤截面的几何建模与有效模场面积、非线性系数的计算。
- 使用 Wandzura 5 阶三角形高斯积分规则对三角形区域进行数值积分（`triangle_integrand_gauss`）。
- `create_fiber_triangulation`：生成阶跃型光纤的环形三角形网格，返回节点、三角形和边界标记。
- `identify_boundary_nodes`：基于边共享统计识别仅被一个三角形使用的边界边，得到边界节点标记。
- `compute_effective_area`：对网格上的模式场进行强度积分，计算有效模场面积 A_eff。
- `compute_nonlinear_coefficient`：由 A_eff 计算非线性系数 γ。
- 辅助函数 `triangle_area` 计算三角形面积。
- 模块内嵌 Wandzura 节点与权重的常量数组。

### `gnlse_solver.py`
核心求解器，实现广义非线性薛定谔方程（GNLSE）的对称分步傅里叶法（SSFM）数值解。
- `raman_response_function`：生成归一化 Raman 响应函数 h_R(t)。
- `dispersion_operator`：构造频域色散算子 D̂(ω)，支持 α, β₂, β₃, β₄。
- `nonlinear_operator`：计算非线性算子作用结果，包含 Raman 卷积（通过 FFT）和自陡峭项。
- `ssfm_solve`：主传播函数，接收初始脉冲、时间网格、色散/非线性参数及可选的 ASE 噪声和隐式步进标志，返回最终脉冲与历史记录。
- `soliton_order`：计算孤子阶数、色散长度和非线性长度。
- `spectral_width` / `temporal_width`：计算脉冲的 3dB 光谱宽度和时域 FWHM 宽度。

### `jacobi_spectral.py`
基于 Jacobi 多项式的谱方法模块，用于脉冲包络的正交展开与色散算子施加。
- `jacobi_polynomial`：递推计算前 n+1 阶 Jacobi 多项式在给定点的值。
- `jacobi_quadrature_rule`：通过构造 Jacobi 矩阵求特征值/特征向量得到 Gauss-Jacobi 节点和权重。
- `spectral_expand_pulse`：将时域脉冲映射到 [-1,1] 区间，用 Jacobi 求积节点展开，返回展开系数与重构脉冲。
- `dispersion_operator_spectral`：在谱系数空间中利用微分矩阵近似施加色散（β₂, β₃ 对应的二阶、三阶导数作用）。

### `pulse_overlap.py`
处理分段线性函数乘积积分及脉冲重叠相关计算。
- `r8vec_bracket3`：在有序数组中二分定位求值区间。
- `pwl_product_integral`：计算两个分段线性函数在给定区间上的乘积积分（基于 Simpson 法则）。
- `pulse_nonlinear_overlap`：计算双脉冲强度乘积的时域重叠积分（XPM 作用强度）。
- `pulse_inner_product`：计算复值脉冲的内积，实虚部分别调用 PWL 积分。
- `raman_response_convolution`：通过 FFT 计算 Raman 响应函数与脉冲强度的因果卷积。

### `monte_carlo_sampler.py`
高维蒙特卡洛采样与球面积分模块，用于不确定性量化和远场积分。
- `hyperball01_sample`：在单位 m 维超球内均匀采样 n 个点。
- `tp_to_xyz`：经纬度到单位球面直角坐标转换。
- `sphere01_triangle_vertices_to_area`：利用 L’Huilier 定理计算球面三角形面积。
- `sphere01_quad_llm`：经纬度网格中点规则球面积分，支持极帽和中间区域。
- `sphere_cvt_step`：球面 Centroidal Voronoi Tessellation 一步迭代（基于蒙特卡洛近似）。
- `monte_carlo_uncertainty_quantification`：从参数中心和标准差生成多维正态扰动样本。

### `noise_model.py`
噪声模型与统计模块，模拟 ASE 噪声、光子统计及 Parrondo 启发式耦合。
- `brownian_motion_simulation`：模拟 m 维布朗运动轨迹。
- `generate_ase_noise`：根据自发辐射因子、增益、光子能量等生成复高斯 ASE 噪声。
- `bose_einstein_distribution`：计算给定平均光子数的玻色-爱因斯坦概率分布。
- `photon_number_fluctuation`：基于光子数分布的量子涨落估计。
- `parrondo_inspired_noise_coupling`：交替施加两种随机相位扰动，模拟两个“不利”过程交替可能产生的有益效果。

### `sparse_solver.py`
重启 GMRES 稀疏线性求解器，并支持色散矩阵的 CRS 格式构建。
- `mult_givens`：应用 Givens 旋转更新向量。
- `ax_crs`：CRS 格式的稀疏矩阵-向量乘法。
- `mgmres`：重启 GMRES 算法，实现 Arnoldi 过程、重新正交化、Givens 旋转消去上 Hessenberg 矩阵、回代求解。
- `build_dispersion_matrix_crs`：利用中心有限差分构建色散算子时域离散矩阵的 CRS 表示，支持 β₂, β₃, β₄。

### `mcmc_inversion.py`
DREAM MCMC 参数反演模块，用于从观测数据推断光纤物理参数。
- 管理交叉概率 CR 的初始化、选择、更新和概率分布调整。
- `sample_candidate`：利用差分进化和 CR 交叉生成候选参数，支持长跳。
- `gr_compute`：计算 Gelman-Rubin R 统计量以评估链收敛。
- `dream_mcmc`：主函数，管理多条链的进化，接受/拒绝基于似然与先验，输出后验样本与接受率。

### `rootfinder.py`
Laguerre 根查找与阶跃光纤模式特征方程求解。
- `zero_laguerre`：通用 Laguerre 方法求根，利用函数值及一、二阶导数。
- `fiber_mode_characteristic_eq` / `fiber_mode_characteristic_eq_scalar`：构建阶跃光纤弱导近似下的特征方程 u * J'_l / J_l - w * K'_l / K_l = 0，并提供数值导数。
- `find_fiber_mode_roots`：扫描符号变化并用二分法精化，得到模式 u 值和对应的传播常数 β。
- `degree_estimation`：粗略估计特征方程的多项式次数。

### `phase_coding.py`
相位编码、幻方矩阵与整数搜索模块，模拟频谱操作和 WDM 信道选择。
- `caesar_shift_phase`：对频谱进行循环移位（Caesar 密码启发的相位编码）。
- `magic_matrix`：用 Siamese 方法生成奇数阶幻方矩阵。
- `magic_phase_mask`：由幻方生成 [0, 2π) 的二维相位掩码。
- `apply_phase_mask_to_pulse`：通过频域施加幻方相位调制来编码脉冲。
- `four_fifths_search`：搜索满足 a^p + b^p +
