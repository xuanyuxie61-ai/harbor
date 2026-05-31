"""
main.py
=======
博士级利率期限结构模型：多因子 HJM 框架综合计算平台

本程序为零参数可运行入口，执行以下完整流程:
  1. 初始化多因子 HJM 模型参数
  2. 生成二维有限元空间网格（期限 × 时间域）
  3. 市场收益率曲线校准（Shepard 插值 + Horner 多项式）
  4. 初始化前向利率曲线
  5. 运行 Lorenz-96 / Duffing / Oregonator 多因子随机动力学
  6. 求解 HJM 前向利率 PDE（输运-扩散方程）
  7. 计算零息债券价格与零息收益率
  8. 多项式混沌不确定性量化
  9. 稀疏矩阵分析与存储
 10. 输出结果与性能指标

金融背景
--------
利率期限结构（Term Structure of Interest Rates）描述了不同期限的无风险利率
之间的关系，是金融工程的核心理论对象。本程序基于 Heath-Jarrow-Morton (HJM)
一般无套利框架，引入多因子随机波动率结构：

    因子 1: Vasicek 型指数衰减波动率 σ_1(t,s) = σ_0 exp(-κ_1 s)
    因子 2: 斜率型波动率 σ_2(t,s) = σ_0 s exp(-κ_2 s)
    因子 3: 混沌耦合波动率 σ_3(t,s) = σ_chaos(t) exp(-κ_3 s)

其中 σ_chaos(t) 由 Lorenz-96 混沌系统（市场微观噪声）、Duffing 振子
（利率周期性波动）和 Oregonator 化学反应系统（流动性冲击）通过非线性
耦合矩阵投影得到。

核心 PDE（Musiela 参数化）:
    ∂r/∂t = -∂r/∂s + ν ∂²r/∂s² + μ(s) ∂r/∂s + α(t,s) + F(t,s)
    r(t,0) = r_0(t)   （短期利率边界）
    r(t,s_max) = r_∞  （长期利率渐近值）
    r(0,s) = r_init(s) （初始期限结构）

科学公式
--------
1. HJM 无套利漂移限制:
   α(t,T) = Σ_{i=1}^d σ_i(t,T) ∫_t^T σ_i(t,u) du

2. 债券定价公式:
   P(t,T) = exp(-∫_t^T f(t,s) ds)

3. 零息收益率:
   y(t,T) = -ln P(t,T) / (T - t)

4. 多项式混沌展开:
   r_t(s;ξ) = Σ_{|α|≤p} r_{t,α}(s) He_α(ξ)

5. Hermite 多项式递推:
   He_{n+1}(x) = x He_n(x) - n He_{n-1}(x)

6. Lambert W 函数（闭式解辅助）:
   W(z) e^{W(z)} = z

7. 对数正态分布（利率 positivity 约束）:
   X ~ LogNormal(μ,σ²):  f(x) = exp(-(ln x - μ)²/(2σ²)) / (x σ √(2π))

8. 后向 Euler 离散:
   (I - dt A) u^{n+1} = u^n + dt f^n

9. RK3 时间推进:
   k1 = dt f(t,u)
   k2 = dt f(t+dt, u+k1)
   k3 = dt f(t+dt/2, u+(k1+k2)/4)
   u_new = u + (k1 + k2 + 4k3)/6

10. Shepard 插值:
    w_j = ||x - x_j||^{-p}
    f(x) = Σ w_j z_j / Σ w_j
"""

import os
import sys
import time
