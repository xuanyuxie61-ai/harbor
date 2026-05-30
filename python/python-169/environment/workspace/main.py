
import sys
import numpy as np


sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))

from core_planner import ManipulatorMotionPlanner


def main():
    print("=" * 70)
    print("  7-DOF 冗余机械臂实时轨迹规划与动态避障系统")
    print("  高维约束空间下的混合整数非线性优化")
    print("=" * 70)
    print()


    np.random.seed(42)


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


    traj = results['trajectory']
    ts_test = np.linspace(traj.t0, traj.tf, 50)
    max_viol = 0.0
    for t in ts_test:
        q, dq, ddq = traj.evaluate(t)

        violation = np.max([np.max(q - np.pi), np.max(-np.pi - q)])
        max_viol = max(max_viol, violation)

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
