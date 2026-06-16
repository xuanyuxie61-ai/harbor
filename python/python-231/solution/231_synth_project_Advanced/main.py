"""
main.py — PDF 全局拟合与误差传播的统一入口
=============================================================
计算高能物理：PDF 全局拟合与误差传播
—— 高阶有限差分与稳定性分析（小规模可复现实验）

本项目融合 15 个种子项目的核心算法, 实现:
  - LO DGLAP 演化方程的 IMEX 高阶求解
  - PDF 参数化的 χ² 全局拟合
  - Hessian 与 Monte Carlo 误差传播
  - 有限差分格式的稳定性分析
  - 特征向量退化度分析

运行方式:
    python main.py

无需任何参数, 直接运行即可.

映射自 15 个种子项目:
  1.  757_mesh2d              → 自适应 x 网格加密
  2.  140_caustic              → Mellin-N 模块化算术
  3.  621_kmeans_fast          → MC 副本 k-means 聚类
  4.  976_r8ci                 → 循环矩阵 FD 稳定性
  5.  108_boundary_word_hexagon→ 约束词合法性
  6.  1164_jenzenho_flame-ai   → 扰动传播追踪
  7.  409_fem2d_poisson        → FEM 弱形式验证
  8.  1020_lx913_SoleFlip      → 单参数翻转敏感度
  9.  1428_zero_chandrupatla   → Chandrupatla 求根
  10. 928_pwl_interp_2d_scattered → PWL 插值
  11. 1148_agrimUT_SLAI        → 流水线编排与计时
  12. 1159_KadelkaLab_canalization → 退化度计数
  13. 1377_usa_box_plot        → 矩阵文本热图
  14. 1184_Secure-Data-Reconstruction → 稀疏异常恢复
  15. 226_craps_simulation     → 蒙特卡洛概率估计
"""
from __future__ import annotations
import sys
import os

# 确保可以导入同目录模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import PDFGlobalFitPipeline


def main():
    """
    PDF 全局拟合流水线主入口.

    配置参数:
        nx:         x 网格点数
        nq:         Q² 网格点数
        q02:        初始演化尺度 Q0² (GeV²)
        qmax2:      最大演化尺度 Qmax² (GeV²)
        n_replicas: Monte Carlo 副本数
    """
    config = {
        'nx': 24,           # x 网格点 (对数均匀)
        'nq': 12,           # Q² 网格点
        'q02': 2.0,         # Q0² = 2 GeV²
        'qmax2': 1.0e4,     # Qmax² = 10000 GeV²
        'n_replicas': 3,    # MC 副本数 (小规模实验)
    }

    pipeline = PDFGlobalFitPipeline(config)
    results = pipeline.run(verbose=True)

    print("\n[完成] PDF 全局拟合与误差传播流水线已成功执行.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
