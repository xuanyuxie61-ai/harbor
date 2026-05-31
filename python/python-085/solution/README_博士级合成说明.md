# 博士级科研代码合成说明

## 项目概述

**科学领域**：结构力学 —— 接触问题与摩擦算法

**合成目标**：基于15个种子科研项目的核心算法，融合构建一个面向前沿博士级科学计算的 Python 项目，求解二维弹性体的 Signorini-Coulomb 接触问题，并集成磨损演化、不确定性分析、摩擦参数优化、粗糙度随机场与复频域稳定性分析。

---

## 一、原项目到科学问题的映射

| 序号 | 种子项目 | 核心算法 | 在合成项目中的角色 |
|------|---------|---------|------------------|
| 1 | 1340_triangulation_node_to_element | 三角网格节点到单元映射与平均 | `mesh_generator.py`：有限元网格生成、节点-单元邻接表、面积加权平均 |
| 2 | 828_ode_midpoint | 中点法隐式ODE积分 | `wear_ode.py`：Archard磨损定律的时间积分（不动点迭代中点法） |
| 3 | 288_diophantine | 非负丢番图方程整数解 | `diophantine_utils.py`：Hermite标准型整数规划、节点重编号优化 |
| 4 | 131_c8lib | 复数矩阵运算库 | `complex_analysis.py`：复模态分析、复矩阵Frobenius范数、幂迭代 |
| 5 | 888_polygon_monte_carlo | 多边形区域蒙特卡洛采样 | `monte_carlo_contact.py`：接触区域多边形均匀采样与压力统计 |
| 6 | 1266_toms178 | Hooke-Jeeves直接搜索优化 | `friction_optimization.py`：摩擦系数反演优化 |
| 7 | 856_peaks_movie | 峰值测试函数 | `friction_optimization.py`：peaks函数作为非线性接触势能测试曲面 |
| 8 | 1379_usa_matrix | 稀疏矩阵构造 | `fem_assembler.py`：全局刚度矩阵稀疏组装模式 |
| 9 | 973_r8cb | 紧凑带状矩阵LU分解 | `banded_solver.py`：无选主元紧凑带状分解与求解 |
| 10 | 935_pyramid_monte_carlo | 三维金字塔区域采样 | `monte_carlo_contact.py`：金字塔区域体积积分采样函数 |
| 11 | 979_r8gb | 一般带状矩阵PLU分解 | `banded_solver.py`：带选主元的一般带状PLU分解与求解 |
| 12 | 1167_stla_to_tri_surface | STL三角表面解析 | `mesh_generator.py`：ASCII STL文件读取与三角表面拓扑处理 |
| 13 | 871_plasma_matrix | 非线性Jacobian矩阵组装 | `fem_assembler.py`：弹性刚度矩阵与残差向量组装模式 |
| 14 | 1092_snakes_and_ladders_simulation | 蒙特卡洛统计框架 | `monte_carlo_contact.py`：批量蒙特卡洛统计与敏感性分析 |
| 15 | 870_pink_noise | 1/f噪声生成与自相关 | `roughness_field.py`：接触面分形粗糙度随机场、自相关函数 |

---

## 二、新增数学物理模型与核心公式

### 2.1 线弹性本构（平面应变）

应力-应变关系（Hooke定律）：

$$
\sigma = D : \varepsilon, \quad
D = \frac{E(1-\nu)}{(1+\nu)(1-2\nu)}
\begin{bmatrix}
1 & \frac{\nu}{1-\nu} & 0 \\
\frac{\nu}{1-\nu} & 1 & 0 \\
0 & 0 & \frac{1-2\nu}{2(1-\nu)}
\end{bmatrix}
$$

### 2.2 Signorini接触条件（法向）

对于接触节点 $i \in \mathcal{C}$：

$$
g_n^{(i)} \ge 0, \quad p_n^{(i)} \ge 0, \quad g_n^{(i)} \cdot p_n^{(i)} = 0
$$

其中 $g_n^{(i)} = (y_i + u_y^{(i)}) - y_{\text{rigid}}$ 为法向间隙。

### 2.3 Coulomb摩擦定律（切向）

$$
\|p_t^{(i)}\| \le \mu \, p_n^{(i)}
$$

粘着状态：$\|p_t\| < \mu p_n \Rightarrow \dot{u}_t = 0$

滑动状态：$\|p_t\| = \mu p_n \Rightarrow p_t = -\mu p_n \frac{\dot{u}_t}{\|\dot{u}_t\|}$

### 2.4 增广Lagrange泛函

$$
\mathcal{L}(u, \lambda_n, \lambda_t) = \frac{1}{2} u^T K u - u^T f_{ext}
+ \sum_{i \in \mathcal{C}} \left[
\lambda_n^{(i)} g_n^{(i)} + \frac{c_n}{2} (g_n^{(i)})^2
+ \lambda_t^{(i)} g_t^{(i)} + \frac{c_t}{2} (g_t^{(i)})^2
\right]
$$

乘子更新（Uzawa型）：

$$
\lambda_n^{new} = \max(\lambda_n - c_n g_n, 0), \quad
\lambda_t^{new} = \text{proj}_{[-\mu\lambda_n, \mu\lambda_n]}(\lambda_t + c_t g_t)
$$

### 2.5 Archard磨损定律

$$
\frac{dh}{dt} = k_w \cdot p_n(t) \cdot |v_t(t)|
$$

其中 $v_t(t) = v_0 \sin(\omega t)$ 为往复切向速度。

### 2.6 复模态稳定性分析

状态空间方程：

$$
\dot{z} = A z, \quad A = \begin{bmatrix} 0 & I \\ -M^{-1}K & -M^{-1}C \end{bmatrix}
$$

稳定性判据：若 $\max_i \text{Re}(\lambda_i) > 0$，则系统存在摩擦诱发颤振（sprag-slip）。

### 2.7 分形粗糙度功率谱

$$
S(k_x, k_y) = C \cdot (k_x^2 + k_y^2)^{-(H+1)}
$$

Hurst指数 $H \in [0.5, 0.9]$，分形维数 $D = 3 - H$。

### 2.8 Hooke-Jeeves优化

用于摩擦系数校准：

$$
J(\mu) = (Q_{\text{sim}}(\mu) - Q_{\text{target}})^2 \rightarrow \min
$$

---

## 三、修改了哪些文件来实现合成

本合成项目共包含 **12个Python文件**：

| 文件 | 功能 | 融入的种子项目 |
|------|------|--------------|
| `main.py` | 统一入口，零参数运行全部流程 | 全部15个项目 |
| `mesh_generator.py` | 网格生成、节点-单元映射、STL解析 | 1340, 1167 |
| `fem_assembler.py` | 线弹性FEM刚度矩阵组装、残差计算 | 871, 1379 |
| `banded_solver.py` | 紧凑/一般带状矩阵LU/PLU分解 | 973, 979 |
| `contact_solver.py` | Signorini-Coulomb增广Lagrange求解 | 核心科学算法 |
| `wear_ode.py` | Archard磨损ODE中点法/RK4积分 | 828 |
| `monte_carlo_contact.py` | 多边形/金字塔MC采样、统计框架 | 888, 935, 1092 |
| `friction_optimization.py` | Hooke-Jeeves优化、peaks测试函数 | 1266, 856 |
| `roughness_field.py` | 1/f噪声粗糙度场、自相关分析 | 870 |
| `diophantine_utils.py` | 丢番图整数解、节点重编号 | 288 |
| `complex_analysis.py` | 复模态稳定性、频响函数 | 131 |
| `utils.py` | 通用数学工具、安全除法、Macaulay括号 | 973, 979, 131 |

---

## 四、合成后的项目能够解决什么科学问题

1. **二维弹性Signorini接触问题**：计算刚性基础与可变形体之间的接触压力分布，满足非穿透条件。
2. **Coulomb摩擦接触状态判定**：区分粘着（sticking）与滑动（slipping）状态，计算切向摩擦牵引力。
3. **磨损演化预测**：基于Archard定律，积分计算往复滑动条件下的累积磨损深度。
4. **接触压力不确定性量化**：通过蒙特卡洛采样评估接触压力的统计分布（均值、方差、极值）。
5. **摩擦系数反演校准**：利用Hooke-Jeeves无梯度优化，从实验观测值反演材料摩擦系数。
6. **接触面粗糙度建模**：生成符合分形统计（1/f^β功率谱）的随机粗糙表面，叠加到接触几何上。
7. **摩擦诱发颤振稳定性分析**：通过复特征值分析判定接触系统是否存在动力不稳定模态。
8. **带状矩阵快速求解**：针对接触问题产生的大型稀疏带状线性系统，提供高效的LU/PLU分解算法。

---

## 五、合成后的项目如何运行

### 环境要求
- Python 3.8+
- NumPy

### 运行方式
```bash
cd Synthesis-project-python/085_synth_project
python main.py
```

程序将自动执行以下10个阶段：
1. 网格生成与三角剖分
2. 线弹性有限元刚度矩阵组装
3. Signorini-Coulomb接触问题求解（增广Lagrange）
4. 带状矩阵求解器验证
5. Archard磨损演化分析
6. 接触压力蒙特卡洛不确定性分析
7. 摩擦系数Hooke-Jeeves优化校准
8. 接触面1/f粗糙度随机场生成
9. 丢番图节点编号整数规划
10. 复特征值稳定性分析

最终输出计算结果汇总，零参数、无需任何外部输入文件。

---

## 六、质量检查清单

- [x] 原目录未被修改
- [x] 合成后项目为Python语言
- [x] 新目录完整包含合成后的项目（12个.py文件 + 1个.md文档）
- [x] 只有一个博士级科学计算问题已落地为可执行代码
- [x] 每个输入项目都已真实融入合成项目，无遗漏、无挂名
- [x] `main.py`已实际运行通过，零参数可运行且无报错
- [x] 代码具备边界处理与数值鲁棒性（安全除法、除零检查、奇异矩阵保护）
- [x] 文档中存在大量公式与清晰推导关系
- [x] 中文说明文档已生成并可用于第三方复查
- [x] 原始未合成的中间文件已被删除
