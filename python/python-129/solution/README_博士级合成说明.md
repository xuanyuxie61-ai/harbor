# 血凝级联反应网络多尺度动力学建模系统

## 项目概述

本项目围绕 **生物医学：血凝级联反应网络（Coagulation Cascade）**，融合15个种子项目的核心算法，构建了一个面向前沿生物医学计算问题的博士级多尺度动力学建模框架。

血凝级联是维持止血与血栓平衡的核心生化网络，涉及数十种凝血因子、血小板、抑制剂在血管损伤处的复杂时空相互作用。本系统从分子反应动力学（ODE）、空间扩散-反应（PDE）、网络拓扑分析、蒙特卡洛随机模拟、参数敏感性降维、数值稳定性诊断、基因多态性编码、 clot 微观结构表征、患者表型聚类等**9个维度**对血栓形成过程进行全链条建模，具备极高的科学计算复杂度与工程鲁棒性。

---

## 科学问题与核心模型

### 1. 血凝级联ODE反应网络模型

系统状态向量 $\mathbf{y} = [y_1, y_2, \dots, y_{12}]^T$ 包含 12 种关键分子/复合物：

$$
\mathbf{y} = [\text{TF-VIIa}, \text{IXa}, \text{Xa}, \text{Va}, \text{IIa}, \text{Fibrin}, \text{APC}, \text{ATIII-Xa}, \text{TFPI}, \text{Plasmin}, \text{tPA}, \text{PLT}_{\text{act}}]^T
$$

外源性途径催化：
$$v_{\text{IXa}} = 2 k_{\text{cat}}^{\text{TF}} \cdot [\text{TF-VIIa}] \cdot \frac{[\text{IX}]}{K_M^{\text{IX}} + [\text{IX}]}$$

$$v_{\text{Xa}}^{\text{(TF)}} = 2 k_{\text{cat}}^{\text{TF}} \cdot [\text{TF-VIIa}] \cdot \frac{[\text{X}]}{K_M^{\text{X}} + [\text{X}]}$$

内源性途径放大：
$$v_{\text{Xa}}^{\text{(IXa)}} = 2 k_{\text{cat}}^{\text{IXa}} \cdot [\text{IXa}] \cdot \frac{[\text{X}]}{K_M^{\text{X}} + [\text{X}]}$$

凝血酶爆发（正反馈核心）：
$$v_{\text{IIa}} = 3 k_{\text{cat}}^{\text{Xa}} \cdot [\text{Xa}] \cdot [\text{Va}] \cdot \frac{[\text{II}]}{K_M^{\text{II}} + [\text{II}]}$$

$$v_{\text{Va}}^{\text{反馈}} = 2 [\text{IIa}] \cdot \frac{\max(5.0 - [\text{Va}], 0)}{5.0 + [\text{Va}]}$$

纤维蛋白聚合（Hill动力学）：
$$\frac{d[\text{Fibrin}]}{dt} = k_{\text{poly}} \cdot \frac{[\text{IIa}]^n}{K_{\text{poly}}^n + [\text{IIa}]^n} - k_{\text{lysis}} \cdot [\text{Plasmin}] \cdot [\text{Fibrin}]$$

抑制系统：
$$\text{inact}_{\text{ATIII}} = k_{\text{inact}}^{\text{ATIII}} \cdot [\text{ATIII}] \cdot [\text{enzyme}]$$

整体向量形式：
$$\frac{d\mathbf{y}}{dt} = \mathbf{R}(\mathbf{y}; \mathbf{p}), \quad \mathbf{y}(0) = \mathbf{y}_0$$

### 2. 血管截面反应-扩散-源项PDE模型

在柱坐标 $(r, \theta)$ 下，凝血因子浓度 $c(r,\theta,t)$ 满足：

$$\frac{\partial c}{\partial t} = D \cdot \frac{1}{r} \frac{\partial}{\partial r}\left(r \frac{\partial c}{\partial r}\right) + \frac{D}{r^2} \frac{\partial^2 c}{\partial \theta^2} + R(c) + S(r,\theta,t)$$

其中 $S(r,\theta)$ 为血管壁损伤处的高斯释放源：
$$S(r,\theta) = S_0 \exp\left(-\frac{(r-r_w)^2}{2\sigma_r^2} - \frac{(\theta-\theta_w)^2}{2\sigma_\theta^2}\right)$$

### 3. 修正Bessel函数的Clot径向分布

血栓内部纤维蛋白浓度的稳态柱坐标解：
$$D_{\text{fibrin}} \cdot \frac{1}{r} \frac{d}{dr}\left(r \frac{dC}{dr}\right) - k_{\text{poly}} \cdot C = 0$$

通解为修正Bessel函数组合：
$$C(r) = A \, I_\nu(\alpha r) + B \, K_\nu(\alpha r), \quad \alpha = \sqrt{\frac{k_{\text{poly}}}{D_{\text{fibrin}}}}$$

在 clot 中心有界条件要求 $B=0$，边缘归一化后：
$$\tilde{C}(r) = \frac{I_\nu(\alpha r)}{I_\nu(\alpha r_0)}$$

### 4. 网络PageRank重要性分析

将血凝级联建模为有向图 $G=(V,E)$，定义随机冲浪矩阵 $\mathbf{S}$：
$$S_{ij} = \frac{A_{ij}}{\sum_k A_{ik}}$$

PageRank向量满足：
$$\boldsymbol{\pi}^T = \alpha \, \boldsymbol{\pi}^T \mathbf{S} + \frac{1-\alpha}{|V|} \mathbf{1}^T$$

通过幂迭代求解，识别网络关键Hub节点（如 IIa、Fibrin、Xa）。

### 5. Jaccard距离与层次聚类

节点间功能距离定义为Jaccard距离：
$$d(i,j) = 1 - \frac{|N(i) \cap N(j)|}{|N(i) \cup N(j)|}$$

采用单连接（single linkage）层次聚类：
$$d(C_i, C_j) = \min_{u \in C_i, v \in C_j} d(u,v)$$

### 6. SVD参数敏感性降维

敏感性矩阵 $\mathbf{S} \in \mathbb{R}^{m \times p}$ 的SVD分解：
$$\mathbf{S} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^T$$

在低维子空间中进行参数优化：
$$\mathbf{p} = \mathbf{p}_0 + \mathbf{V}_r \boldsymbol{\alpha}, \quad \boldsymbol{\alpha} = \boldsymbol{\Sigma}_r^{-1} \mathbf{U}_r^T (\mathbf{R}_{\text{target}} - \mathbf{S}\mathbf{p}_0)$$

### 7. 最优停止策略与蒙特卡洛模拟

血小板粘附的最优停止问题：观察前 $k$ 个血小板后，选择第一个超过阈值的。

理论最优 $k^* \approx N/e$，其中 $e$ 为自然对数底。

clot稳定性评分：
$$S = \bar{X} \sqrt{N_{\text{adhered}}} - \lambda \left(N_{\text{adhered}} - \frac{N}{3}\right)^2$$

### 8. 六边形数值积分与三角形重心坐标求积

正六边形区域上的矩：
$$M_{p,q} = \int_{\text{hex}} x^p y^q \, dx \, dy$$

三角形上的对称求积规则：
$$\int_T f(x,y) \, dx \, dy = |T| \sum_i w_i f(\lambda_1^{(i)}, \lambda_2^{(i)}, \lambda_3^{(i)})$$

Arrhenius速率的高斯-埃尔米特积分：
$$k_{\text{eff}}(T) = A \int_0^\infty e^{-E_a/(RT)} \cdot \mathcal{N}(E_a; \mu, \sigma^2) \, dE_a$$

### 9. 数值稳定性分析

隐式梯形法稳定性函数：
$$R(z) = \frac{1 + z/2}{1 - z/2}$$

刚性比：
$$S = \frac{|\text{Re}(\lambda_{\max})|}{|\text{Re}(\lambda_{\min})|}$$

本系统刚性比 $S \sim 6 \times 10^4$，属于严重刚性，必须使用A-稳定隐式方法。

### 10. Gray码基因多态性编码

Gray码转换：
$$G(n) = n \oplus (n \gg 1)$$

Hamming距离：
$$H(\mathbf{a}, \mathbf{b}) = \sum_i |a_i - b_i| = \text{popcount}(\mathbf{a} \oplus \mathbf{b})$$

遗传风险评分：
$$\text{Risk} = \sum_j w_j \cdot g_j$$

### 11. 交换优化聚类

类内离散度准则：
$$J = \sum_{k=1}^K \sum_{i \in C_k} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2$$

通过对象类别交换优化，每次交换计算 $\Delta J$，若 $\Delta J < 0$ 则执行。

---

## 15个种子项目映射关系

| 序号 | 种子项目 | 核心算法/思想 | 合成项目中的角色 |
|------|---------|-------------|----------------|
| 1 | **079_besselj** | 非整数阶Bessel函数数值计算 | `special_functions.py`: 修正Bessel函数 $I_\nu, K_\nu$ 用于clot径向浓度分布 |
| 2 | **154_chain_letter_tree** | 距离矩阵构建与层次聚类 | `network_pagerank.py`: Jaccard距离矩阵 + 单连接层次聚类分析凝血因子功能模块 |
| 3 | **534_high_card_simulation** | 最优停止策略蒙特卡洛 | `monte_carlo_platelet.py`: 血小板最优粘附时机选择的 $1/e$ 法则模拟 |
| 4 | **1170_stochastic_heat2d** | 2D随机热方程有限差分 | `reaction_diffusion_pde.py`: 环形区域反应-扩散方程的稀疏矩阵有限差分求解 |
| 5 | **527_hexagon_integrals** | 六边形区域数值积分 | `quadrature_integrals.py`: 正六边形矩计算用于clot微观结构渗透率估算 |
| 6 | **485_gray_code_display** | Gray码与Hamming距离 | `gray_code_genetics.py`: 凝血因子基因多态性的Gray码编码与Hamming距离风险度量 |
| 7 | **845_pagerank2** | 稀疏矩阵图网络构造 | `network_pagerank.py`: 血凝级联有向图的稀疏邻接矩阵 + PageRank重要性排序 |
| 8 | **011_annulus_rule** | 环形区域数值积分 | `reaction_diffusion_pde.py`: 血管横截面环形区域上的面积积分（梯形法则权重） |
| 9 | **105_boundary_locus2** | ODE稳定性区域分析 | `stability_analysis.py`: 隐式梯形法稳定性函数 $R(z)$ 与系统刚性比诊断 |
| 10 | **039_asa113** | 交换优化聚类 | `clustering_phenotypes.py`: 患者凝血表型的swap-based非层次聚类优化 |
| 11 | **1389_variomino** | 变体多格矩阵操作 | `variomino_clot.py`: 纤维蛋白clot的密度矩阵凝聚、嵌入、旋转、孔隙连通性分析 |
| 12 | **1313_triangle_quadrature_symmetry** | 三角形重心坐标求积 | `quadrature_integrals.py`: 任意三角形上的3点/4点对称高斯求积规则 |
| 13 | **1190_svd_powers** | SVD分解与降维 | `svd_sensitivity.py`: 参数敏感性矩阵的SVD分解 + 低维子空间投影优化 |
| 14 | **692_llsq** | 线性最小二乘拟合 | `svd_sensitivity.py`: 参数子空间中的最小二乘拟合（隐含在降维投影中） |
| 15 | **831_ode_trapezoidal** | 梯形法ODE求解 | `coagulation_ode.py`: 隐式梯形法思想扩展为BDF刚性求解器处理血凝级联ODE |

---

## 文件清单

| 文件 | 功能描述 |
|------|---------|
| `main.py` | **统一入口**，零参数运行，顺序调用所有11个模块完成全流程模拟 |
| `special_functions.py` | Bessel函数库与clot径向分布模型 |
| `coagulation_ode.py` | 12维血凝级联ODE核心模型 + BDF刚性求解器 |
| `reaction_diffusion_pde.py` | 环形区域反应-扩散-源项PDE求解器 |
| `network_pagerank.py` | 血凝网络图构造、PageRank、层次聚类 |
| `monte_carlo_platelet.py` | 血小板最优停止蒙特卡洛模拟 |
| `quadrature_integrals.py` | 六边形/三角形数值积分 + Arrhenius速率计算 |
| `svd_sensitivity.py` | SVD参数敏感性分析与降维优化 |
| `stability_analysis.py` | 数值方法稳定性函数与刚性诊断 |
| `gray_code_genetics.py` | Gray码基因型编码与Hamming距离分析 |
| `variomino_clot.py` | clot微观结构多格矩阵表示与拓扑分析 |
| `clustering_phenotypes.py` | 患者凝血表型交换优化聚类 |

---

## 运行方式

```bash
cd Synthesis-project-python/129_synth_project
python main.py
```

程序将自动执行以下完整流程：
1. Bessel函数精度验证与clot径向分布计算
2. 血凝级联ODE系统600秒动力学模拟
3. 血管截面反应-扩散稳态求解
4. 血凝网络PageRank与层次聚类
5. 血小板最优停止蒙特卡洛模拟
6. 六边形/三角形数值积分与Arrhenius速率
7. SVD参数敏感性降维分析
8. ODE求解器稳定性与刚性诊断
9. 凝血因子基因Gray码编码
10. clot多格结构孔隙与各向异性分析
11. 患者表型交换优化聚类

---

## 数值鲁棒性设计

- **非负约束**：所有生化浓度变量在ODE/PDE求解中强制执行 $y \geq 0$
- **边界处理**：Bessel函数在 $r>r_0$ 时指数衰减；PDE浓度截断在 $[0, 10^6]$；Newton迭代失败后回退到最小二乘
- **刚性处理**：系统自动检测Jacobian刚性比（$S \sim 6 \times 10^4$），采用A-稳定的BDF方法
- **参数防除零**：所有Michaelis-Menten项分母加入 $10^{-12}$ 安全余量
- **空类处理**：聚类算法中若出现空类，自动随机选择样本填充

---

## 科学计算复杂度说明

本项目的博士级难度体现在：

1. **多尺度耦合**：从分子反应（ms）到 clot 结构（min）跨越5个时间数量级
2. **高维刚性ODE**：12维状态空间，Jacobian条件数极高
3. **空间-反应耦合**：2D柱坐标PDE与ODE源项的迭代求解
4. **网络拓扑分析**：28节点有向图的PageRank + 层次聚类
5. **参数降维**：12参数空间的SVD主成分提取与低维优化
6. **多物理场公式**：生化反应动力学 + 扩散输运 + 特殊函数 + 遗传编码 + 蒙特卡洛随机过程
