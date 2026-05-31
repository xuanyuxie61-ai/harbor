# ---- TC01: circle_arc_grid 返回形状正确 (n, 2) ----
pts = circle_arc_grid(0.0, 0.0, 5.0, 0.0, 90.0, 5)
assert pts.shape == (5, 2), '[TC01] circle_arc_grid shape FAILED'

# ---- TC02: circle_arc_grid 所有点到圆心距离为 r ----
import numpy as np
pts = circle_arc_grid(1.0, 2.0, 3.0, 0.0, 360.0, 10)
dists = np.sqrt((pts[:, 0] - 1.0)**2 + (pts[:, 1] - 2.0)**2)
assert np.allclose(dists, 3.0, atol=1e-12), '[TC02] circle_arc_grid radius FAILED'

# ---- TC03: polynomial_multiply 卷积正确性 ----
p = np.array([1.0, 2.0, 3.0])
q = np.array([4.0, 5.0])
result = polynomial_multiply(p, q)
expected = np.convolve(p, q)
assert np.allclose(result, expected), '[TC03] polynomial_multiply FAILED'

# ---- TC04: diff2_center sin''(x) = -sin(x) 数值验证 ----
x_val = np.pi / 6.0
d2 = diff2_center(np.sin, x_val, h=1e-4)
assert abs(d2 + np.sin(x_val)) < 1e-6, '[TC04] diff2_center FAILED'

# ---- TC05: triangle_area_2d 直角三角形面积 ----
from utils import triangle_area_2d
a = np.array([0.0, 0.0])
b = np.array([3.0, 0.0])
c = np.array([0.0, 4.0])
area = triangle_area_2d(a, b, c)
assert abs(area - 6.0) < 1e-12, '[TC05] triangle_area_2d FAILED'

# ---- TC06: triangle_angles 内角和为 π ----
from utils import triangle_angles
a = np.array([0.0, 0.0])
b = np.array([1.0, 0.0])
c = np.array([0.5, 0.866])
angles = triangle_angles(a, b, c)
assert abs(np.sum(angles) - np.pi) < 1e-12, '[TC06] triangle_angles sum FAILED'

# ---- TC07: rk4_step 求解 y'=y, y(0)=1 精度验证 ----
from utils import rk4_step
def exp_ode(t, y):
    return np.array([y[0]])
y0 = np.array([1.0])
y1 = rk4_step(exp_ode, 0.0, y0, 0.1)
expected_y1 = np.exp(0.1)
assert abs(y1[0] - expected_y1) < 1e-6, '[TC07] rk4_step FAILED'

# ---- TC08: i4mat_rref 单位矩阵保持不变 ----
from utils import i4mat_rref
I = np.eye(4, dtype=np.int64)
rref = i4mat_rref(I.copy())
assert np.allclose(rref, I), '[TC08] i4mat_rref identity FAILED'

# ---- TC09: GridTopology.generate_ring_radial_topology 确定性输出 ----
import numpy as np
from grid_topology import GridTopology
np.random.seed(42)
g1 = GridTopology.generate_ring_radial_topology(n_ring=4, n_radial=1, r_inner=3.0, r_outer=6.0)
np.random.seed(42)
g2 = GridTopology.generate_ring_radial_topology(n_ring=4, n_radial=1, r_inner=3.0, r_outer=6.0)
assert np.allclose(g1.nodes, g2.nodes), '[TC09] GridTopology deterministic FAILED'

# ---- TC10: GridTopology 边数至少为节点数-1（连通性） ----
from grid_topology import GridTopology
import numpy as np
np.random.seed(123)
grid = GridTopology.generate_ring_radial_topology(n_ring=5, n_radial=1, r_inner=4.0, r_outer=8.0)
edges = grid.get_edge_list()
assert len(edges) >= grid.n_nodes - 1, '[TC10] GridTopology connectivity FAILED'

# ---- TC11: GridTopology 网格质量指标非负 ----
from grid_topology import GridTopology
import numpy as np
np.random.seed(99)
grid = GridTopology.generate_ring_radial_topology(n_ring=4, n_radial=1, r_inner=5.0, r_outer=10.0)
quality = grid.compute_mesh_quality()
assert quality['min_angle_deg'] >= 0.0, '[TC11] GridTopology min_angle_deg FAILED'
assert quality['mean_area'] >= 0.0, '[TC11] GridTopology mean_area FAILED'

# ---- TC12: SparseMatrix add + mv 与 to_dense 一致性 ----
from sparse_matrix import SparseMatrix
import numpy as np
sm = SparseMatrix(3)
sm.add(0, 0, 4.0); sm.add(0, 1, -1.0); sm.add(0, 2, -2.0)
sm.add(1, 0, -1.0); sm.add(1, 1, 4.0); sm.add(1, 2, -1.0)
sm.add(2, 0, -2.0); sm.add(2, 1, -1.0); sm.add(2, 2, 4.0)
x = np.array([1.0, 2.0, 3.0])
y_sparse = sm.mv(x)
y_dense = sm.to_dense() @ x
assert np.allclose(y_sparse, y_dense), '[TC12] SparseMatrix mv vs dense FAILED'

# ---- TC13: conjugate_gradient 求解 SPD 系统 Ax = b ----
from sparse_matrix import SparseMatrix, conjugate_gradient
import numpy as np
sm = SparseMatrix(3)
sm.add(0, 0, 4.0); sm.add(0, 1, 1.0)
sm.add(1, 0, 1.0); sm.add(1, 1, 4.0); sm.add(1, 2, 1.0)
sm.add(2, 1, 1.0); sm.add(2, 2, 4.0)
b = np.array([1.0, 2.0, 3.0])
x_cg = conjugate_gradient(sm, b, tol=1e-12)
assert np.allclose(sm.mv(x_cg), b, atol=1e-8), '[TC13] conjugate_gradient FAILED'

# ---- TC14: build_y_bus 2-bus 导纳矩阵验证 ----
from power_flow import build_y_bus
import numpy as np
edges = np.array([[0, 1]], dtype=np.int32)
r_line = np.array([0.01])
x_line = np.array([0.1])
y_bus = build_y_bus(2, edges, r_line, x_line, None)
y_series = 1.0 / complex(0.01, 0.1)
assert abs(y_bus[0, 0] - y_series) < 1e-12, '[TC14] Y_bus diagonal FAILED'
assert abs(y_bus[0, 1] + y_series) < 1e-12, '[TC14] Y_bus off-diagonal FAILED'

# ---- TC15: EconomicDispatch solve_lambda 出力总和等于需求 ----
from optimal_dispatch import EconomicDispatch
import numpy as np
a = np.array([0.02, 0.015, 0.025])
b = np.array([10.0, 12.0, 8.0])
c = np.array([100.0, 120.0, 90.0])
p_min = np.array([10.0, 10.0, 5.0])
p_max = np.array([100.0, 120.0, 80.0])
ed = EconomicDispatch(a, b, c, p_min, p_max)
res = ed.solve_lambda(150.0)
assert abs(res['total_generation'] - 150.0) < 1e-4, '[TC15] EconomicDispatch total FAILED'

# ---- TC16: EconomicDispatch incremental_cost 公式验证 ----
from optimal_dispatch import EconomicDispatch
import numpy as np
a = np.array([0.01, 0.02])
b = np.array([5.0, 8.0])
c = np.array([50.0, 60.0])
p_min = np.array([0.0, 0.0])
p_max = np.array([200.0, 200.0])
ed = EconomicDispatch(a, b, c, p_min, p_max)
p_test = np.array([100.0, 80.0])
ic = ed.incremental_cost(p_test)
assert abs(ic[0] - (2.0 * 0.01 * 100.0 + 5.0)) < 1e-10, '[TC16] incremental_cost FAILED'
assert abs(ic[1] - (2.0 * 0.02 * 80.0 + 8.0)) < 1e-10, '[TC16] incremental_cost FAILED'

# ---- TC17: LoadMarkovModel fit 转移矩阵行和为 1 ----
import numpy as np
from load_markov import LoadMarkovModel
np.random.seed(42)
load_data = np.random.default_rng(42).normal(100, 20, 200)
model = LoadMarkovModel(n_states=5)
model.fit(load_data)
row_sums = model.P.sum(axis=1)
assert np.allclose(row_sums, 1.0, atol=1e-9), '[TC17] Markov row sums FAILED'

# ---- TC18: LoadMarkovModel predict 概率分布和为 1 ----
import numpy as np
from load_markov import LoadMarkovModel
np.random.seed(42)
load_data = np.random.default_rng(42).normal(100, 20, 200)
model = LoadMarkovModel(n_states=5)
model.fit(load_data)
pred = model.predict(current_state=2, n_steps=3)
assert abs(pred.sum() - 1.0) < 1e-9, '[TC18] Markov predict sum FAILED'

# ---- TC19: LoadMarkovModel entropy_rate 非负 ----
import numpy as np
from load_markov import LoadMarkovModel
np.random.seed(42)
load_data = np.random.default_rng(42).normal(100, 20, 200)
model = LoadMarkovModel(n_states=5)
model.fit(load_data)
H = model.entropy_rate()
assert H >= 0.0, '[TC19] entropy_rate non-negative FAILED'

# ---- TC20: SwingEquation electrical_power 已知角度验证 ----
from transient_stability import SwingEquation
import numpy as np
swing = SwingEquation(H=5.0, D=2.0, E_prime=1.1, V_inf=1.0, X=0.5)
Pe_90 = swing.electrical_power(np.pi / 2.0)
assert abs(Pe_90 - 1.1 * 1.0 / 0.5 * 1.0) < 1e-12, '[TC20] electrical_power FAILED'

# ---- TC21: SwingEquation critical_clearing_angle 在 δ_0 与 δ_max 之间 ----
from transient_stability import SwingEquation
import numpy as np
swing = SwingEquation(H=5.0, D=2.0, E_prime=1.1, V_inf=1.0, X=0.5)
P_m = 0.8
P_max = 1.1 * 1.0 / 0.5
delta_0 = np.arcsin(P_m / P_max)
delta_cr = swing.critical_clearing_angle(P_m)
assert delta_cr is not None, '[TC21] critical_clearing_angle None FAILED'
assert delta_0 < delta_cr < np.pi - delta_0, '[TC21] critical_clearing_angle range FAILED'

# ---- TC22: SwingEquation simulate 无故障仿真稳定 ----
from transient_stability import SwingEquation
import numpy as np
swing = SwingEquation(H=5.0, D=2.0, E_prime=1.1, V_inf=1.0, X=0.5)
P_m = 0.5
delta_0 = np.arcsin(P_m * swing.X / (swing.E_prime * swing.V_inf))
omega_0 = swing.omega_s
res = swing.simulate(t_span=(0.0, 2.0), dt=0.01, P_m=P_m, delta0=delta_0, omega0=omega_0)
assert res['stable'], '[TC22] SwingEquation stable FAILED'

# ---- TC23: MultiMachineStability simulate 返回正确形状 ----
from transient_stability import MultiMachineStability
import numpy as np
n_gen = 3
H_mm = np.array([5.0, 4.0, 6.0])
D_mm = np.array([2.0, 1.5, 2.5])
E_mm = np.array([1.1, 1.05, 1.08])
Y_red = np.array([
    [0.5 - 5.0j, 0.2 + 1.0j, 0.1 + 0.5j],
    [0.2 + 1.0j, 0.6 - 6.0j, 0.15 + 0.8j],
    [0.1 + 0.5j, 0.15 + 0.8j, 0.55 - 5.5j]
])
mm = MultiMachineStability(n_gen, H_mm, D_mm, E_mm, Y_red)
P_m_mm = np.array([0.8, 0.6, 0.7])
delta0_mm = np.array([0.3, 0.2, 0.25])
omega0_mm = np.full(n_gen, mm.omega_s)
res_mm = mm.simulate(t_span=(0.0, 1.0), dt=0.05, P_m=P_m_mm, delta0=delta0_mm, omega0=omega0_mm)
assert len(res_mm['t']) == 21, '[TC23] MultiMachine t length FAILED'
assert res_mm['delta'].shape[1] == 3, '[TC23] MultiMachine delta shape FAILED'
assert res_mm['omega'].shape[1] == 3, '[TC23] MultiMachine omega shape FAILED'

# ---- TC24: LineReliability 可用率在 [0,1] 范围内 ----
from reliability import LineReliability
import numpy as np
line_lengths = np.array([5.0, 10.0, 15.0])
rel = LineReliability(line_lengths, lambda0=0.1, mu=8760.0)
assert np.all(rel.availability >= 0.0), '[TC24] availability >= 0 FAILED'
assert np.all(rel.availability <= 1.0), '[TC24] availability <= 1 FAILED'

# ---- TC25: LineReliability series_reliability ≤ 1 ----
from reliability import LineReliability
import numpy as np
line_lengths = np.array([5.0, 5.0, 5.0, 5.0])
rel = LineReliability(line_lengths, lambda0=0.05, mu=8760.0)
sr = rel.series_reliability([0, 1, 2, 3])
assert 0.0 < sr <= 1.0, '[TC25] series_reliability range FAILED'

# ---- TC26: VoltageStabilityMargin max_power_limit 为正 ----
from reliability import VoltageStabilityMargin
vsm = VoltageStabilityMargin(E=1.1, X=0.5, Q=0.2)
p_max = vsm.max_power_limit()
assert p_max > 0.0, '[TC26] max_power_limit positive FAILED'

# ---- TC27: VoltageStabilityMargin voltage_margin 返回 [0,1] ----
from reliability import VoltageStabilityMargin
vsm = VoltageStabilityMargin(E=1.1, X=0.5, Q=0.2)
margin = vsm.voltage_margin(P_operating=0.5)
assert 0.0 <= margin <= 1.0, '[TC27] voltage_margin range FAILED'

# ---- TC28: ThreePhaseUnbalance 平衡三相系统不平衡度 ≈ 0 ----
from reliability import ThreePhaseUnbalance
import numpy as np
alpha = np.exp(1j * 2.0 * np.pi / 3.0)
Va_bal = complex(1.0, 0.0)
Vb_bal = Va_bal * alpha**2
Vc_bal = Va_bal * alpha
tpu_bal = ThreePhaseUnbalance(Va_bal, Vb_bal, Vc_bal)
uf_bal = tpu_bal.unbalance_factor()
assert uf_bal < 1e-6, '[TC28] balanced unbalance ≈ 0 FAILED'

# ---- TC29: ObservabilityAnalysis 满秩矩阵可观测 ----
from state_estimation import ObservabilityAnalysis
import numpy as np
obs = ObservabilityAnalysis(n_bus=4)
H_full = np.random.default_rng(42).normal(0, 1, (8, 8))
obs_res = obs.check_observability(H_full)
assert obs_res['observable'], '[TC29] observability FAILED'
assert obs_res['deficiency'] == 0, '[TC29] observability deficiency FAILED'

# ---- TC30: diff2_center 边界检查 h 必须为正 ----
import numpy as np
try:
    diff2_center(np.sin, 0.0, h=0.0)
    assert False, '[TC30] diff2_center h=0 should raise FAILED'
except ValueError:
    pass

# ---- TC31: UnitCommitmentDP solve_single_unit_dp 返回6时段调度 ----
from optimal_dispatch import UnitCommitmentDP
import numpy as np
uc = UnitCommitmentDP(
    n_gen=3, T=6,
    startup_cost=np.array([50.0, 60.0, 40.0]),
    shutdown_cost=np.array([20.0, 25.0, 15.0]),
    min_up=np.array([2, 2, 1]),
    min_down=np.array([2, 2, 1])
)
ed_cost_on = np.array([500.0, 450.0, 480.0, 520.0, 550.0, 510.0])
res_uc = uc.solve_single_unit_dp(gen_idx=0, ed_cost_on=ed_cost_on, ed_cost_off=0.0)
assert len(res_uc['schedule']) == 6, '[TC31] UC schedule length FAILED'
assert res_uc['total_cost'] > 0.0, '[TC31] UC total_cost positive FAILED'
