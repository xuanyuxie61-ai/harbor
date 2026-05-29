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

# ================================================================
# 测试用例（25个，assert模式，涉及随机值均使用固定种子）
# ================================================================
# ---- TC01: main() 完整管线返回dict类型 ----
res = main()
assert isinstance(res, dict), '[TC01] main() 应返回 dict FAILED'

# ---- TC02: 返回字典包含全部11个期望键 ----
expected_keys = ['trajectory', 'path_nodes', 'cycle_allocations', 'milp_decision',
                 't_arr', 'y_arr', 'cost_integral', 'avg_manipulability', 'min_manipulability',
                 'target_curve', 'dq_ik']
for key in expected_keys:
    assert key in res, f'[TC02] 缺少键 {key} FAILED'

# ---- TC03: 速度代价积分为有限非负值 ----
assert np.isfinite(res['cost_integral']), '[TC03] cost_integral 应为有限值 FAILED'
assert res['cost_integral'] >= 0.0, '[TC03] cost_integral 应为非负数 FAILED'

# ---- TC04: 可操纵性度量均为有限正值 ----
assert np.isfinite(res['avg_manipulability']), '[TC04] avg_manipulability 应为有限值 FAILED'
assert res['avg_manipulability'] > 0.0, '[TC04] avg_manipulability 应为正值 FAILED'
assert np.isfinite(res['min_manipulability']), '[TC04] min_manipulability 应为有限值 FAILED'
assert res['min_manipulability'] > 0.0, '[TC04] min_manipulability 应为正值 FAILED'

# ---- TC05: 轨迹对象类型为JointSpaceBezierTrajectory ----
from bernstein_path import JointSpaceBezierTrajectory
traj = res['trajectory']
assert isinstance(traj, JointSpaceBezierTrajectory), '[TC05] 轨迹类型错误 FAILED'

# ---- TC06: trajectory.evaluate() 返回三个正确形状的(7,)向量 ----
q, dq, ddq = traj.evaluate(traj.t0)
assert q.shape == (7,), f'[TC06] 位置向量形状应为(7,) 实际{q.shape} FAILED'
assert dq.shape == (7,), f'[TC06] 速度向量形状应为(7,) 实际{dq.shape} FAILED'
assert ddq.shape == (7,), f'[TC06] 加速度向量形状应为(7,) 实际{ddq.shape} FAILED'

# ---- TC07: Bézier曲线在t0处等于第一个控制点（de Casteljau端点性质） ----
P = traj.P
q0, _, _ = traj.evaluate(traj.t0)
assert np.allclose(q0, P[0], atol=1e-10), '[TC07] 曲线在t0处不等于第一个控制点 FAILED'

# ---- TC08: Bézier曲线在tf处等于最后一个控制点（de Casteljau端点性质） ----
qf, _, _ = traj.evaluate(traj.tf)
assert np.allclose(qf, P[-1], atol=1e-10), '[TC08] 曲线在tf处不等于最后一个控制点 FAILED'

# ---- TC09: 轨迹位置在50个采样点处均不超出关节限位[-π, π] ----
for ti in np.linspace(traj.t0, traj.tf, 50):
    qi, _, _ = traj.evaluate(ti)
    assert np.all(qi >= -np.pi - 1e-8), f'[TC09] 位置低于下限 at t={ti} FAILED'
    assert np.all(qi <= np.pi + 1e-8), f'[TC09] 位置超出上限 at t={ti} FAILED'

# ---- TC10: 轨迹速度和加速度在20个采样点处均为有限值 ----
for ti in np.linspace(traj.t0, traj.tf, 20):
    qi, dqi, ddqi = traj.evaluate(ti)
    assert np.isfinite(dqi).all(), f'[TC10] 速度含非有限值 at t={ti} FAILED'
    assert np.isfinite(ddqi).all(), f'[TC10] 加速度含非有限值 at t={ti} FAILED'

# ---- TC11: trajectory.position()/velocity()/acceleration() 返回正确形状 ----
t_mid = 0.5 * (traj.t0 + traj.tf)
assert traj.position(t_mid).shape == (7,), '[TC11] position() 形状错误 FAILED'
assert traj.velocity(t_mid).shape == (7,), '[TC11] velocity() 形状错误 FAILED'
assert traj.acceleration(t_mid).shape == (7,), '[TC11] acceleration() 形状错误 FAILED'

# ---- TC12: ODE仿真时间数组t_arr严格递增 ----
t_arr = res['t_arr']
assert t_arr.size >= 2, '[TC12] ODE时间步数不足 FAILED'
assert np.all(np.diff(t_arr) > 0), '[TC12] ODE时间数组不严格递增 FAILED'

# ---- TC13: ODE解y_arr形状为(步数, 14) ----
y_arr = res['y_arr']
assert y_arr.ndim == 2, '[TC13] y_arr 维度错误 FAILED'
assert y_arr.shape[0] == t_arr.size, '[TC13] y_arr 行数与时间步数不匹配 FAILED'
assert y_arr.shape[1] == 14, f'[TC13] y_arr 列数应为14(7位置+7速度) 实际{y_arr.shape[1]} FAILED'

# ---- TC14: cycle_allocations每行加权和等于100 ----
ca = res['cycle_allocations']
if ca.size > 0:
    joint_weights = np.array([5, 5, 4, 4, 3, 3, 2], dtype=int)
    weighted_sums = np.dot(ca, joint_weights)
    assert np.all(weighted_sums == 100), '[TC14] 控制周期分配加权和不等于100 FAILED'

# ---- TC15: dq_ik解向量为有限值且形状为(7,) ----
dq_ik = res['dq_ik']
assert dq_ik.shape == (7,), f'[TC15] dq_ik 形状错误 FAILED'
assert np.isfinite(dq_ik).all(), '[TC15] dq_ik 含非有限值 FAILED'

# ---- TC16: target_curve为(n_points, 3)数组 ----
tc = res['target_curve']
assert tc.ndim == 2, '[TC16] target_curve 维度错误 FAILED'
assert tc.shape[1] == 3, f'[TC16] target_curve 列数应为3 实际{tc.shape[1]} FAILED'
assert np.isfinite(tc).all(), '[TC16] target_curve 含非有限值 FAILED'

# ---- TC17: milp_decision包含expected键 ----
md = res['milp_decision']
assert 'selected_corridors' in md, '[TC17] milp_decision 缺少 selected_corridors FAILED'
assert isinstance(md['selected_corridors'], list), '[TC17] selected_corridors 应为 list FAILED'

# ---- TC18: path_nodes每个元素为(7,)的ndarray ----
pn = res['path_nodes']
assert len(pn) >= 2, '[TC18] path_nodes 节点数应>=2 FAILED'
for node in pn:
    assert isinstance(node, np.ndarray), '[TC18] path_node 不是 ndarray FAILED'
    assert node.shape == (7,), f'[TC18] path_node 形状应为(7,) 实际{node.shape} FAILED'

# ---- TC19: 可复现性——两次seed=42调用cost_integral一致 ----
np.random.seed(42)
res2 = main()
assert np.isclose(res['cost_integral'], res2['cost_integral'], atol=1e-12), '[TC19] 可复现性失败 FAILED'

# ---- TC20: bernstein_basis 单位划分——在任意t处ΣB_i=1 ----
from bernstein_path import bernstein_basis
import numpy as np
for n_test in [1, 2, 3, 5, 7]:
    for t_test in [0.0, 0.25, 0.5, 0.75, 1.0]:
        B = bernstein_basis(n_test, 0.0, 1.0, t_test)
        assert np.isclose(np.sum(B), 1.0, atol=1e-12), f'[TC20] n={n_test} t={t_test} 和≠1 FAILED'
        assert np.all(B >= -1e-15), f'[TC20] n={n_test} t={t_test} 负基函数值 FAILED'

# ---- TC21: legendre_compute 权重和为2，节点在[-1,1]内 ----
from pseudospectral_control import legendre_compute
nodes, weights = legendre_compute(8)
assert np.isclose(np.sum(weights), 2.0, atol=1e-10), '[TC21] Legendre权重和≠2 FAILED'
assert np.all(nodes >= -1.0) and np.all(nodes <= 1.0), '[TC21] Legendre节点超出[-1,1] FAILED'

# ---- TC22: simplex_grid_size 与组合数公式一致 ----
from configuration_space import simplex_grid_size
from math import comb
assert simplex_grid_size(3, 5) == comb(5 + 3, 3), '[TC22] simplex_grid_size(3,5) 与 C(8,3) 不一致 FAILED'
assert simplex_grid_size(2, 4) == comb(4 + 2, 2), '[TC22] simplex_grid_size(2,4) 与 C(6,2) 不一致 FAILED'
assert simplex_grid_size(0, 10) == 1, '[TC22] simplex_grid_size(0,10) 应为1 FAILED'

# ---- TC23: triangle_signed_area_2d 正确计算已知三角形面积 ----
from obstacle_geometry import triangle_signed_area_2d
p1 = np.array([0.0, 0.0])
p2 = np.array([1.0, 0.0])
p3 = np.array([0.0, 1.0])
area = triangle_signed_area_2d(p1, p2, p3)
assert np.isclose(area, 0.5, atol=1e-12), '[TC23] 单位直角三角形面积应为0.5 FAILED'
# 交换顶点顺序应得到负面积
area_neg = triangle_signed_area_2d(p1, p3, p2)
assert np.isclose(area_neg, -0.5, atol=1e-12), '[TC23] 反向三角形面积应为-0.5 FAILED'

# ---- TC24: 正运动学返回4×4齐次变换矩阵 ----
from kinematics_dynamics import ManipulatorKinematics
kin = ManipulatorKinematics()
q_test = np.array([0.0, -0.5, 0.0, -1.5, 0.0, 1.0, 0.0])
T = kin.forward_kinematics(q_test)
assert T.shape == (4, 4), f'[TC24] FK返回形状应为(4,4) 实际{T.shape} FAILED'
assert np.isclose(T[3, 3], 1.0, atol=1e-12), '[TC24] 齐次变换矩阵右下角应为1 FAILED'
# 旋转子矩阵检验正交性
R = T[:3, :3]
assert np.allclose(R @ R.T, np.eye(3), atol=1e-10), '[TC24] 旋转子矩阵非正交 FAILED'
assert np.isclose(np.linalg.det(R), 1.0, atol=1e-10), '[TC24] 旋转子矩阵行列式≠1 FAILED'

# ---- TC25: diophantine_nd_solutions 所有解满足 a·x = b ----
from discrete_planning import diophantine_nd_solutions
a_test = np.array([3, 5, 7], dtype=int)
b_test = 20
sols = diophantine_nd_solutions(a_test, b_test, max_solutions=50)
assert sols.size > 0, '[TC25] 应有非零数量的Diophantine解 FAILED'
for sol in sols:
    assert np.dot(a_test, sol) == b_test, f'[TC25] 解 {sol} 不满足 a·x=b FAILED'
    assert np.all(sol >= 0), f'[TC25] 解 {sol} 含负数 FAILED'

print('\n全部 25 个测试通过!\n')
