"""
PROJECT_275: 计算凝聚态 - 非厄米体系谱结构与例外点
====================================================

本包实现了非厄米 SSH/Hatano-Nelson 模型的高阶有限差分分析、
例外点定位、谱稳定性、自适应网格等博士级数值工具.

模块:
  - nonhermitian_hamiltonian: 哈密顿量构造 (graph_adj, blend, polynomial)
  - high_order_fd: 高阶有限差分与稳定性 (burgers_leap, allen_cahn)
  - spectral_solver: 谱分解与双正交基
  - exceptional_point_locator: EP 定位 (cauchy, cvt, polynomial)
  - stability_analysis: 时间演化稳定性 (leapfrog, Cauchy theta)
  - brillouin_quadrature: BZ 积分 (witherden, wedge, triangle)
  - parameter_continuation: 参数延拓 (cauchy continuation, blend)
  - spectral_statistics: 谱统计 (asa005, box_plot, digital_dice)
  - phase_classifier: 谱相分类 (seizyml features, graph_adj)
  - adaptive_ep_mesh: 自适应网格 (LPFC adaptation, cvt)
"""
from __future__ import annotations

__version__ = "1.0.0"
__all__ = [
    "nonhermitian_hamiltonian",
    "high_order_fd",
    "spectral_solver",
    "exceptional_point_locator",
    "stability_analysis",
    "brillouin_quadrature",
    "parameter_continuation",
    "spectral_statistics",
    "phase_classifier",
    "adaptive_ep_mesh",
]
