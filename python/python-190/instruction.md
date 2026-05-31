# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

```markdown
# 项目描述：物理信息生成对抗网络（PI-GAN）三维流场合成

本项目基于纯 NumPy 实现了一个物理信息生成对抗网络，用于三维不可压缩 Navier‑Stokes 方程流场的生成。系统包含多个独立模块，分别负责精确解生成、几何处理、采样、网络构建、训练、数值积分、特殊函数、网格生成、文件 I/O、等变性验证及超参数优化。项目入口为 `main.py`，其他文件均被 `main.py` 导入使用，需逐一实现。

## 文件清单与职责

### 1. `navier_stokes_exact.py` —— 精确解与残差计算
- 提供三维不可压缩 Navier‑Stokes 方程的 Ethier 精确解函数 `uvwp_ethier`，用于生成真实流场数据。
- 提供基于中心差分的 Navier‑Stokes 残差计算函数 `ns_residual`，可处理规则网格或展平数据。
- 提供 `generate_training_data` 函数，在结构化网格上调用 `uvwp_ethier` 生成训练样本（坐标和状态）。

### 2. `gan_numpy.py` —— 纯 NumPy 生成对抗网络
- 实现全连接层 `DenseLayer`（含 He 初始化、反向传播、带动量的 SGD 更新）。
- 实现激活层 `ReLU`、`LeakyReLU`、`Sigmoid`（前向与反向传播）。
- 实现损失函数 `MSELoss`、`BCELoss`（数值稳定版）。
- 实现 `Generator` 类：输入为隐向量与坐标的拼接，输出速度场和压力。
- 实现 `Discriminator` 类：输入为坐标与物理量的拼接，输出真假概率。
- 提供 `compute_physics_loss_batch` 函数，用于在子网格上计算生成场的 NS 残差损失。

### 3. `training_engine.py` —— 训练循环与评估
- 负责数据准备、训练循环（对抗训练 + 物理损失 + 等变损失）。
- 提供 `prepare_training_data` 和 `train_pigan` 函数，实现完整的训练流程。
- 提供 `evaluate_with_geometry` 函数，调用几何统计、球面积分等模块进行评估。

### 4. `complex_geometry.py` —— 多边形几何处理
- 实现射线法判断点是否在多边形内 `point_in_polygon`。
- 实现多边形包围盒计算 `polygon_bounding_box`。
- 实现 Monte Carlo 面积估计 `polygon_area_mc`。
- 生成简化人体轮廓多边形 `human_outline_polygon`（利用参数曲线）。
- 多边形内部均匀采样 `sample_in_polygon`（拒绝采样，失败时回退至网格采样）。
- 在复杂多边形域内生成带物理精确解的速度场样本 `velocity_field_in_complex_domain`。

### 5. `cvt_sampler.py` —— Centroidal Voronoi Tessellation 采样
- 实现 CVT 能量计算 `cvt_energy`。
- 实现单步 Lloyd 迭代 `lloyd_step`（将生成元更新为 Voronoi 单元的质心）。
- 实现二维矩形域 CVT 采样 `cvt_2d_sampling`。
- 实现高维隐空间 CVT 采样 `cvt_latent_samples`。

### 6. `geometric_stats.py` —— 几何统计
- 实现三角形内均匀采样（Turk's rule） `triangle_sample`。
- 计算三角形内随机点对距离的统计量 `triangle_distance_stats`。
- 提供等边三角形内点对距离的精确 PDF `equilateral_distance_pdf`。
- 基于 Monte Carlo 距离统计近似 Wasserstein‑1 距离 `wasserstein_approx_mc`。
- 在整个三角网格上估计随机点对距离分布 `mesh_distance_distribution`。

### 7. `hooke_jeeves.py` —— 直接搜索优化
- 实现 Hooke‑Jeeves 模式搜索算法 `hooke_jeeves`（探测移动、模式移动、步长收缩）。
- 提供 `optimize_gan_hyperparams` 函数，用于对 GAN 超参数进行后验微调。

### 8. `latent_path.py` —— 隐空间路径优化
- 实现 TSP 穷举求解 `tsp_brute`，使用 Trotter 排列生成算法。
- 实现隐向量集的 TSP 最优排序 `latent_space_tsp_path`（小规模穷举，大规模贪心）。
- 实现球面线性插值 `slerp`。
- 提供隐空间插值序列生成 `latent_interpolation_sequence`。

### 9. `mesh_generator.py` —— 三角网格生成
- 实现 Bowyer‑Watson 增量 Delaunay 三角剖分 `bowyer_watson`。
- 在封闭多边形内部生成带内部点的 Delaunay 网格 `generate_mesh_from_boundary`。
- 生成简化人体轮廓边界 `human_outline_boundary`。
- 提供网格质量统计 `mesh_quality_stats`（最小角、面积比等）。

### 10. `normal_approx.py` —— 正态分布近似
- 实现三种标准正态 CDF 计算方法：AS 66 (`alnorm`)、Algorithm 5666 (`normp`)、Algorithm 39 (`nprob`)。
- 提供向量化 CDF 接口 `standard_normal_cdf`。
- 实现 Box‑Muller 变换 `box_muller_transform` 生成标准正态样本。
- 实现重参数化采样 `reparameterized_gaussian_sample`。
- 计算两个一元高斯分布的 KL 散度 `gaussian_kl_divergence`。

### 11. `obj_io.py` —— Wavefront OBJ 文件 I/O
- 读取 OBJ 文件 `obj_read`，返回顶点、法线、面片。
- 写入 OBJ 文件 `obj_write`。
- 计算面片法线 `compute_face_normals` 和顶点法线 `compute_vertex_normals`（面积加权平均）。
- 生成 icosphere 网格 `icosphere_obj`。

### 12. `quaternion_equivariance.py` —— 四元数旋转等变性
- 实现四元数乘法、共轭、范数、归一化、逆、指数等基本运算。
- 将旋转轴与角度转换为单位四元数 `rotation_axis_to_quat`。
- 使用四元数旋转向量 `rotate_vector_by_quat`。
- 同时旋转坐标与速度场 `rotate_velocity_field`。
- 计算旋转等变性损失 `equivariance_loss`。

### 13. `special_functions.py` —— 特殊函数
- 实现 Clausen 函数 `clausen`（级数求和）及向量化版本
