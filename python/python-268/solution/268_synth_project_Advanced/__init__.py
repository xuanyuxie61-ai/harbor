"""
268_synth_project_Advanced
===========================

计算凝聚态: 强关联 Hubbard 模型量子蒙特卡洛
高阶有限差分与稳定性分析 (小规模可复现实验)

子模块:
    lattice_geometry          - 晶格几何与布里渊区离散化
    finite_difference         - 高阶有限差分模板与虚时导数
    hubbard_stratonovich      - HS 变换与辅助场采样
    greens_function           - 虚时格林函数
    determinant_qmc           - DQMC 核心采样引擎
    stability_analysis        - 数值稳定性分析
    parameter_optimizer       - 参数空间搜索优化
    thermalization_diagnostics - 热化诊断
    observable_estimator      - 物理可观测量估计
"""

__version__ = '1.0.0'
__author__ = 'DA Project Synthesis'
