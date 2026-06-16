# PROJECT 266 — 计算凝聚态: 密度泛函理论能带计算
## 高阶有限差分与稳定性分析（小规模可复现实验）

---

## 一、项目概述

本项目融合 **15 个种子项目** 的核心数值算法，构建了一个面向 **计算凝聚态物理** 的完整 DFT 能带计算框架。
在 1D 周期晶格（Mathieu 势模型）上，实现了从实空间离散化、Kohn-Sham 方程求解、自洽场迭代、
能带分析到贝叶斯反演的完整计算物理流程。

### 核心科学问题
**在 1D 周期势中，如何用高阶有限差分方法精确求解 Kohn-Sham 方程的能带结构，
并系统分析数值稳定性与收敛性？**

---

## 二、种子项目 → 科学问题映射

| 序号 | 种子项目 | 核心算法 | 在本项目中的角色 |
|:---:|---|---|---|
| 1 | `681_line_integrals` | 1D 线段单式积分 `1/(e+1)` | Brillouin 区精确积分参考值 |
| 2 | `040_asa121` | Trigamma 函数 ψ'(x) 渐近展开 | Fermi-Dirac 热力学、电子比热 |
| 3 | `350_fd_predator_prey` | 前向 Euler ODE 步进 | SCF 动力学稳定性类比 |
| 4 | `302_disk01_rule` | Gauss-Legendre/Jacobi 矩阵求积 | Fourier 系数高精度求积 |
| 5 | `549_humps` | Lorentzian 峰及其解析导数 | 赝势构造、解析导数验证 |
| 6 | `683_line_monte_carlo` | 随机/遍历（黄金比例）采样 | BZ Monte Carlo 积分 |
| 7 | `559_hypercube_integrals` | M 维超立方体单式积分 | 高维 BZ 积分推广接口 |
| 8 | `271_dg1d_advection` | DG 方法、Vandermonde、Jacobi 多项式 | 谱 FD 算子、GL 节点构建 |
| 9 | `742_mcnuggets` | Diophantine 枚举、DP 计数 | 能带占据组合计数 |
| 10 | `596_interp_trig` | 三角基数插值 | 能带平滑加密、周期势重建 |
| 11 | `225_cpr` | Chebyshev 伴随矩阵求根 | van Hove 奇点精确定位 |
| 12 | `362_fd1d_heat_steady` | 三对角 FD、Dirichlet BC | Hartree 势 Poisson 求解 |
| 13 | `1415_will_you_be_alive` | Monte Carlo 概率模拟 | SCF 收敛概率分析 |
| 14 | `1247_vcasasmo_BayRad3D` | 贝叶斯变分推断 | 从能带数据反演势场参数 |
| 15 | `054_asa299` | Simplex 格点枚举 | k 点星轨道分类 |

---

## 三、物理模型与核心公式

### 3.1 Mathieu 型周期势

$$V_{\text{ext}}(x) = V_1 \cos(Gx) + V_2 \cos(2Gx) + V_3 \cos(3Gx)$$

其中 $G = 2\pi/a$ 为第一倒格矢。默认参数：
- $a = 10$ Bohr, $V_1 = 0.5$ Ha, $V_2 = 0.15$ Ha, $V_3 = 0.03$ Ha

### 3.2 Kohn-Sham 方程

$$\left[-\frac{1}{2}\frac{d^2}{dx^2} + V_{\text{eff}}(x)\right]\phi_{nk}(x) = \varepsilon_{nk}\phi_{nk}(x)$$

其中 $V_{\text{eff}} = V_{\text{ext}} + V_H + V_{xc}$（原子单位 $\hbar = m_e = e = 1$）。

### 3.3 高阶有限差分 Laplacian

$$\left(\frac{d^2\psi}{dx^2}\right)_i \approx \frac{1}{dx^2}\sum_{j=-p}^{p} c_j\,\psi_{i+j}$$

$2p$ 阶精度的系数 $c_j$ 通过求解 Vandermonde 系统获得：

$$\sum_{j=1}^p c_j \cdot j^{2m} = \delta_{m,0}, \quad m = 0, \ldots, p-1$$

### 3.4 Bloch 边界条件

$$\psi(x+a) = e^{ika}\psi(x) \quad \Rightarrow \quad \psi_{i+N} = e^{ika}\psi_i$$

### 3.5 修正波数（色散关系）

$$k^2_{\text{mod}}(k) = -\frac{1}{dx^2}\left[c_0 + 2\sum_{j=1}^{p} c_j\cos(jk\,dx)\right]$$

相对误差 $\delta(k) = |k^2_{\text{mod}}/k^2 - 1| \sim O((kdx)^{2p})$。

### 3.6 交换关联泛函（1D LDA）

交换能：$\varepsilon_x(n) = -\alpha \cdot n$, $\quad V_x(n) = -2\alpha \cdot n$, $\quad \alpha = 1/4$

关联能（Casula 参数化）：$\varepsilon_c(r_s) = \frac{A\ln(1+Dr_s)}{1+Br_s+Cr_s^2}$

### 3.7 自洽场迭代

$$n^{\text{new}} = \alpha\, n^{\text{out}} + (1-\alpha)\, n^{\text{in}}$$

收敛判据：$\max_x |n^{\text{out}}(x) - n^{\text{in}}(x)| < \text{tol}$

### 3.8 Trigamma 函数（源自 AS 121）

$$\psi^{(1)}(x) = \frac{1}{x} + \frac{1}{2x^2} + \sum_{k=1}^{4} \frac{B_{2k}}{x^{2k+1}}$$

用于有限温度 DFT 中的电子热力学。

### 3.9 Chebyshev 伴随矩阵求根（van Hove 检测）

将 $d\varepsilon/dk$ 在 Chebyshev 节点上展开，构建伴随矩阵 $C$：
其特征值即为导数零点（van Hove 奇点）。

### 3.10 贝叶斯反演

$$p(\theta|D) \propto \exp\left(-\frac{\chi^2}{2\sigma^2}\right) \cdot p(\theta)$$
$$\chi^2 = \sum_{n,k} \frac{[\varepsilon_{nk}(\theta) - \varepsilon_{nk}^{\text{obs}}]^2}{\sigma^2}$$

---

## 四、项目结构

```
266_synth_project_Advanced/
├── main.py                    # 统一入口（零参数运行）
├── physical_constants.py      # 物理常数、原子单位、FD 系数
├── lattice.py                 # 晶格几何、k 点网格、占据计数
├── potential.py               # 周期势场构造（Mathieu/赝势/三角插值）
├── fd_operators.py            # FD Laplacian 矩阵、Bloch BC、Vandermonde
├── xc_functionals.py          # 交换关联泛函、Fermi-Dirac、trigamma
├── kohn_sham.py               # KS 哈密顿量、特征值求解、电子密度
├── band_structure.py          # 能带分析、三角插值、有效质量
├── scf_solver.py              # 自洽场迭代（线性/Pulay 混合）
├── dos.py                     # 态密度（Gaussian 展宽、Monte Carlo）
├── stability_analysis.py      # 数值稳定性、网格收敛、条件数
├── monte_carlo_bz.py          # BZ Monte Carlo 积分（随机/遍历/Halton）
├── chebyshev_rootfinder.py    # Chebyshev 代理求根（van Hove 检测）
├── bayesian_inverse.py        # 贝叶斯反演（MLE + Laplace 后验）
└── README_博士级合成说明.md     # 本文档
```

---

## 五、运行方法

```bash
cd 266_synth_project_Advanced
python main.py
```

**零参数运行**，自动执行全部 10 个计算阶段，输出完整分析结果。

---

## 六、计算阶段说明

| 阶段 | 内容 | 涉及种子项目 |
|:---:|---|---|
| 1 | 物理常数、晶格设置、k 点网格、k 点星枚举 | 054, 742 |
| 2 | 高阶 FD 算子、色散关系、GL 节点 | 271, 302 |
| 3 | Mathieu 势构造、Fourier 分析、赝势 | 549, 596 |
| 4 | KS 方程直接求解、能带结构 | 362 |
| 5 | SCF 迭代（线性混合 + LDA-XC） | 350 |
| 6 | 能带分析、态密度、三角插值 | 596, 681 |
| 7 | van Hove 奇点检测（CPR） | 225 |
| 8 | Monte Carlo BZ 积分 | 683, 559, 681 |
| 9 | 稳定性分析（FD 误差、网格收敛、条件数） | 350, 1415 |
| 10 | 贝叶斯反演（MLE + Laplace 近似） | 1247, 040 |

---

## 七、关键输出示例

```
Phase 4: 能带结构
  能带 0→1 能隙 (X 点): 284 meV
  能带 1→2 能隙 (X 点): 357 meV

Phase 9: 稳定性分析
  4 阶 FD: kdx_1% = 0.997, Nyquist 误差 = 4.60e-01
  Richardson 外推: E_∞ = -0.24647820 Ha
  哈密顿量条件数: κ(H) = 4.44e+02

Phase 10: 贝叶斯反演
  真实参数: a=10.00, V₁=0.500, V₂=0.150
  反演参数: a=10.94, V₁=0.425, V₂=-0.008
  后验标准差: [0.11, 0.008, 0.018, 0.015]
```

---

## 八、科学创新点

1. **统一 FD 精度框架**：实现 2~10 阶精度的 Laplacian 模板，
   系统分析色散误差、Nyquist 极限、稳定性边界。

2. **多尺度 BZ 采样**：对比 Monkhorst-Pack、随机、遍历（黄金比例）、
   Halton 序列的收敛行为。

3. **CPR 精确定位 van Hove 奇点**：利用 Chebyshev 伴随矩阵的
   特征值分解，实现指数级收敛的奇点定位。

4. **贝叶斯势场反演**：从有限噪声能带数据重建晶体势参数，
   量化后验不确定性。

5. **概率稳定性分析**：将 SCF 收敛视为随机事件，
   通过 Monte Carlo 估计收敛概率。

---

## 九、依赖

- Python ≥ 3.8
- NumPy
- SciPy

无需额外安装包，无需可视化库。

---

## 十、参考文献

1. Monkhorst & Pack, PRB **13**, 5188 (1976) — k 点网格
2. Hesthaven & Warburton, *Nodal DG Methods* (Springer, 2007) — 谱 FD
3. Casula et al., PRB **74**, 245416 (2006) — 1D LDA 关联能
4. Boyd, *Chebyshev and Fourier Spectral Methods* (Dover, 2001) — CPR
5. B.E. Schneider, AS 121: Trigamma function (1978)
6. Chasalow & Brand, AS 299: Simplex lattice enumeration (1995)
