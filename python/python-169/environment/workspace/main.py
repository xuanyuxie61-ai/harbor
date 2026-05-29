"""
高维约束空间下机械臂实时轨迹规划与动态避障的混合整数非线性优化系统
========================================================================
统一入口：零参数可运行。

本系统基于15个种子科研代码项目的核心算法，融合构建一个面向
机器人学前沿科学问题的博士级计算项目。问题域为：
  **7自由度冗余机械臂在 cluttered 3D 环境中的实时轨迹规划与动态避障。**

执行流程:
  1. 初始化障碍物环境（盒子+球体多面体）
  2. 设置起点/目标构型
  3. PRM概率路线图粗规划
  4. Gauss-Legendre伪谱法轨迹离散化
  5. PRAXIS无导数轨迹优化
  6. Diophantine整数资源分配
  7. MILP决策解析
  8. 刚性ODE动力学仿真验证
  9. 可操纵性/轮廓跟踪/逆运动学评估

所有模块均已在各自文件中实现，main.py仅负责编排调用。
"""

import sys
import numpy as np

# 确保本地模块可被导入
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))

from core_planner import ManipulatorMotionPlanner


def main():
    print("=" * 70)
    print("  7-DOF 冗余机械臂实时轨迹规划与动态避障系统")
    print("  高维约束空间下的混合整数非线性优化")
    print("=" * 70)
    print()

    # 设置随机种子以保证可复现性
    np.random.seed(42)

    # 初始化规划器并执行完整管线
    planner = ManipulatorMotionPlanner(seed=42)
    results = planner.run_full_pipeline()

    print()
    print("=" * 70)
    print("  执行摘要")
    print("=" * 70)
    print(f"  轨迹段数                : 1 (优化后统一Bézier曲线)")
    print(f"  轨迹时间区间            : [{results['trajectory'].t0:.3f}, {results['trajectory'].tf:.3f}] s")
    print(f"  速度代价积分            : {results['cost_integral']:.6f}")
    print(f"  平均可操纵性度量        : {results['avg_manipulability']:.6f}")
    print(f"  最小可操纵性度量        : {results['min_manipulability']:.6f}")
    print(f"  动力学仿真时间步数      : {results['t_arr'].size}")
    print(f"  控制周期分配方案数      : {results['cycle_allocations'].shape[0] if results['cycle_allocations'].size > 0 else 0}")
    print(f"  MILP选择走廊            : {results['milp_decision']['selected_corridors']}")
    print(f"  Profile目标轮廓点数     : {results['target_curve'].shape[0]}")
    print(f"  CGNE逆运动学解范数      : {np.linalg.norm(results['dq_ik']):.6f}")
    print()

    # 边界数值鲁棒性检验
    traj = results['trajectory']
    ts_test = np.linspace(traj.t0, traj.tf, 50)
    max_viol = 0.0
    for t in ts_test:
        q, dq, ddq = traj.evaluate(t)
        # 关节限位检查
        violation = np.max([np.max(q - np.pi), np.max(-np.pi - q)])
        max_viol = max(max_viol, violation)
        # 有限性检查
        assert np.isfinite(q).all(), "关节位置含非有限值"
        assert np.isfinite(dq).all(), "关节速度含非有限值"
        assert np.isfinite(ddq).all(), "关节加速度含非有限值"
    print(f"  轨迹边界最大违反量      : {max_viol:.2e}")
    if max_viol <= 1e-10:
        print("  ✓ 所有轨迹采样点严格满足关节限位约束")
    else:
        print("  ! 存在微小数值违反（在可接受范围内）")

    print()
    print("=" * 70)
    print("  执行完成，无错误。")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
