import os

base_dir = "/mnt/data/zpy/sci-swe/source code/Synthesis-project-python/104_synth_project/104_synth_project_Advanced"
out_dir = os.path.join(base_dir, "104_synth_project_Advanced_test_benchmark_kimi-code_kimi-for-coding")

main_py_path = os.path.join(base_dir, "main.py")
with open(main_py_path, "r", encoding="utf-8") as f:
    main_content = f.read()

test_cases_inline = '''# ---- TC01: adaptive_sampling.triangle_area 直角三角形面积正确性 ----
area = adaptive_sampling.triangle_area([0,0], [3,0], [0,4])
assert abs(area - 6.0) < 1e-10, '[TC01] adaptive_sampling.triangle_area 直角三角形面积正确性 FAILED'

# ---- TC02: adaptive_sampling.sample_triangle_uniform 输出形状与凸包范围 ----
pts = adaptive_sampling.sample_triangle_uniform([0,0], [1,0], [0,1], 200, seed=42)
assert pts.shape == (200, 2), '[TC02] adaptive_sampling.sample_triangle_uniform 输出形状与凸包范围 FAILED'
assert np.all(pts >= -1e-9), '[TC02] adaptive_sampling.sample_triangle_uniform 输出形状与凸包范围 FAILED'
assert np.all(pts[:,0] + pts[:,1] <= 1.0001), '[TC02] adaptive_sampling.sample_triangle_uniform 输出形状与凸包范围 FAILED'

# ---- TC03: adaptive_sampling.cvt_disk_uniform 生成器数量与圆盘约束 ----
gens = adaptive_sampling.cvt_disk_uniform(16, radius=1.0, n_iterations=5, seed=42)
assert 0 < len(gens) <= 16, '[TC03] adaptive_sampling.cvt_disk_uniform 生成器数量与圆盘约束 FAILED'
r_gens = np.linalg.norm(gens, axis=1)
assert np.all(r_gens <= 1.0001), '[TC03] adaptive_sampling.cvt_disk_uniform 生成器数量与圆盘约束 FAILED'

# ---- TC04: atmosphere_turbulence.fried_parameter 公式数值验证与零Cn2极限 ----
r0 = atmosphere_turbulence.fried_parameter(500e-9, 1e-14)
assert r0 > 0 and np.isfinite(r0), '[TC04] atmosphere_turbulence.fried_parameter 公式数值验证与零Cn2极限 FAILED'
r0_tiny = atmosphere_turbulence.fried_parameter(500e-9, 1e-30)
assert r0_tiny > 1e6, '[TC04] atmosphere_turbulence.fried_parameter 公式数值验证与零Cn2极限 FAILED'

# ---- TC05: atmosphere_turbulence.generate_phase_screen 输出形状与可复现性 ----
ph_a, m_a = atmosphere_turbulence.generate_phase_screen(64, 0.01, 0.1, seed=42)
ph_b, m_b = atmosphere_turbulence.generate_phase_screen(64, 0.01, 0.1, seed=42)
assert ph_a.shape == (64, 64), '[TC05] atmosphere_turbulence.generate_phase_screen 输出形状与可复现性 FAILED'
assert np.allclose(ph_a, ph_b), '[TC05] atmosphere_turbulence.generate_phase_screen 输出形状与可复现性 FAILED'
assert np.any(m_a), '[TC05] atmosphere_turbulence.generate_phase_screen 输出形状与可复现性 FAILED'

# ---- TC06: atmosphere_turbulence.barenblatt_pme_solution 零边界行为 ----
u_pme = atmosphere_turbulence.barenblatt_pme_solution(np.array([2.0, 3.0]), 1.0, m=3.0, c=0.01, delta=1.0)
assert np.all(u_pme >= 0), '[TC06] atmosphere_turbulence.barenblatt_pme_solution 零边界行为 FAILED'
assert u_pme[0] == 0.0, '[TC06] atmosphere_turbulence.barenblatt_pme_solution 零边界行为 FAILED'

# ---- TC07: closed_loop_control.PIController 积分 windup 限幅 ----
ctrl = closed_loop_control.PIController(Kp=1.0, Ki=10.0, dt=0.1, integral_limit=1.0)
for _ in range(100):
    u = ctrl.update(1.0)
assert abs(ctrl.integral) <= 1.0001, '[TC07] closed_loop_control.PIController 积分 windup 限幅 FAILED'
assert u <= 11.0, '[TC07] closed_loop_control.PIController 积分 windup 限幅 FAILED'

# ---- TC08: closed_loop_control.BandwidthLimitedActuator 阶跃响应稳态 ----
act = closed_loop_control.BandwidthLimitedActuator(100.0, 1e-3)
for _ in range(5000):
    out = act.step(1.0)
assert abs(out - 1.0) < 0.01, '[TC08] closed_loop_control.BandwidthLimitedActuator 阶跃响应稳态 FAILED'

# ---- TC09: closed_loop_control.HHControlCircuit 状态有界性 ----
hh = closed_loop_control.HHControlCircuit(C=1.0, dt=1e-5)
for _ in range(2000):
    v = hh.step(10.0)
assert np.isfinite(v), '[TC09] closed_loop_control.HHControlCircuit 状态有界性 FAILED'
assert -100 < v < 100, '[TC09] closed_loop_control.HHControlCircuit 状态有界性 FAILED'

# ---- TC10: deformable_mirror.magic4_matrix 幻方常数验证 ----
M = deformable_mirror.magic4_matrix(4)
magic_sum = np.sum(M[0,:])
assert np.all(np.sum(M, axis=1) == magic_sum), '[TC10] deformable_mirror.magic4_matrix 幻方常数验证 FAILED'
assert np.all(np.sum(M, axis=0) == magic_sum), '[TC10] deformable_mirror.magic4_matrix 幻方常数验证 FAILED'
assert np.sum(np.diag(M)) == magic_sum, '[TC10] deformable_mirror.magic4_matrix 幻方常数验证 FAILED'

# ---- TC11: deformable_mirror.DeformableMirror compute_surface 零电压输出 ----
dm = deformable_mirror.DeformableMirror(16, 32, aperture_radius=0.5, influence_sigma=0.05, use_magic_square_layout=True)
surf = dm.compute_surface(np.zeros(16))
assert surf.shape == (32, 32), '[TC11] deformable_mirror.DeformableMirror compute_surface 零电压输出 FAILED'
assert np.allclose(surf, 0.0), '[TC11] deformable_mirror.DeformableMirror compute_surface 零电压输出 FAILED'

# ---- TC12: deformable_mirror.FastSteeringMirrorDynamics simulate_response 长度匹配 ----
fsm = deformable_mirror.FastSteeringMirrorDynamics()
t1, t2 = fsm.simulate_response(np.ones(100)*0.001, np.ones(100)*0.001, dt=1e-4)
assert len(t1) == 100 and len(t2) == 100, '[TC12] deformable_mirror.FastSteeringMirrorDynamics simulate_response 长度匹配 FAILED'

# ---- TC13: zernike_modes.compute_zernike_basis 输出维度与掩码 ----
basis13, mask13, xv13, yv13 = zernike_modes.compute_zernike_basis(32, 6)
assert basis13.shape == (32*32, 6), '[TC13] zernike_modes.compute_zernike_basis 输出维度与掩码 FAILED'
assert mask13.shape == (32, 32), '[TC13] zernike_modes.compute_zernike_basis 输出维度与掩码 FAILED'
assert len(xv13) == 32 and len(yv13) == 32, '[TC13] zernike_modes.compute_zernike_basis 输出维度与掩码 FAILED'

# ---- TC14: zernike_modes.zernike_decompose_reconstruct 精确一致性 ----
basis14, mask14, _, _ = zernike_modes.compute_zernike_basis(32, 6)
phase14 = np.zeros((32, 32))
phase14[mask14] = basis14[mask14.ravel(), 2]
coeffs14 = zernike_modes.zernike_decompose(phase14, mask14, basis14)
recon14 = zernike_modes.zernike_reconstruct(coeffs14, basis14, mask14)
assert np.allclose(phase14[mask14], recon14[mask14], atol=1e-6), '[TC14] zernike_modes.zernike_decompose_reconstruct 精确一致性 FAILED'

# ---- TC15: zernike_modes.kolmogorov_zernike_covariance 对角正定性 ----
cov15 = zernike_modes.kolmogorov_zernike_covariance(10, 2.0)
assert np.all(np.diag(cov15) > 0), '[TC15] zernike_modes.kolmogorov_zernike_covariance 对角正定性 FAILED'
assert np.all(cov15 == cov15.T), '[TC15] zernike_modes.kolmogorov_zernike_covariance 对角正定性 FAILED'

# ---- TC16: zernike_modes.simplex_lattice_enum 组合数公式验证 ----
lattice16 = zernike_modes.simplex_lattice_enum(3, 4)
expected16 = 15
assert len(lattice16) == expected16, '[TC16] zernike_modes.simplex_lattice_enum 组合数公式验证 FAILED'
assert np.all(np.sum(lattice16, axis=1) <= 4), '[TC16] zernike_modes.simplex_lattice_enum 组合数公式验证 FAILED'

# ---- TC17: optical_transfer.tetrahedron01_monomial_integral 体积验证 ----
vol17 = optical_transfer.tetrahedron01_monomial_integral(0, 0, 0)
assert abs(vol17 - 1.0/6.0) < 1e-14, '[TC17] optical_transfer.tetrahedron01_monomial_integral 体积验证 FAILED'

# ---- TC18: wavefront_propagation.compute_strehl_ratio 平坦波前为1 ----
N18 = 64
x18 = np.linspace(-1, 1, N18)
X18, Y18 = np.meshgrid(x18, x18)
mask18 = (X18**2 + Y18**2) <= 1.0
flat18 = np.zeros((N18, N18))
S18 = wavefront_propagation.compute_strehl_ratio(flat18, 500e-9, mask18)
assert abs(S18 - 1.0) < 1e-6, '[TC18] wavefront_propagation.compute_strehl_ratio 平坦波前为1 FAILED'

# ---- TC19: wavefront_propagation.compute_wavefront_curvature_hessian 输出形状 ----
Wxx, Wyy, Wxy = wavefront_propagation.compute_wavefront_curvature_hessian(np.zeros((16,16)), 0.01)
assert Wxx.shape == (16, 16) and Wyy.shape == (16, 16) and Wxy.shape == (16, 16), '[TC19] wavefront_propagation.compute_wavefront_curvature_hessian 输出形状 FAILED'

# ---- TC20: iterative_utils.collatz_step_size 残差单调递减性 ----
s_large = iterative_utils.collatz_step_size(1e6, base_step=1.0)
s_small = iterative_utils.collatz_step_size(1e-12, base_step=1.0)
assert s_large < s_small, '[TC20] iterative_utils.collatz_step_size 残差单调递减性 FAILED'
assert abs(s_small - 0.5) < 1e-10, '[TC20] iterative_utils.collatz_step_size 残差单调递减性 FAILED'

# ---- TC21: wavefront_reconstruction.build_southwell_matrix_1d 矩阵维度与有界性 ----
A21 = wavefront_reconstruction.build_southwell_matrix_1d(5, 0.1)
x21 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y21 = A21.matvec(x21)
assert len(y21) == 5, '[TC21] wavefront_reconstruction.build_southwell_matrix_1d 矩阵维度与有界性 FAILED'
assert np.isfinite(y21).all(), '[TC21] wavefront_reconstruction.build_southwell_matrix_1d 矩阵维度与有界性 FAILED'

# ---- TC22: wavefront_reconstruction.reconstruct_wavefront_1d 零斜率输出为常数 ----
slopes22 = np.zeros(5)
recon22 = wavefront_reconstruction.reconstruct_wavefront_1d(slopes22, 0.1, method='cg')
assert len(recon22) == 6, '[TC22] wavefront_reconstruction.reconstruct_wavefront_1d 零斜率输出为常数 FAILED'
assert np.allclose(np.diff(recon22), 0.0, atol=1e-8), '[TC22] wavefront_reconstruction.reconstruct_wavefront_1d 零斜率输出为常数 FAILED'

# ---- TC23: shack_hartmann_sensor.generate_subaperture_grid 方形网格数量 ----
subs23 = shack_hartmann_sensor.generate_subaperture_grid(64, 4, geometry='square')
assert len(subs23) == 16, '[TC23] shack_hartmann_sensor.generate_subaperture_grid 方形网格数量 FAILED'

# ---- TC24: shack_hartmann_sensor.slopes_to_vector 双向转换一致性 ----
sx24 = np.array([1.0, 2.0, 3.0])
sy24 = np.array([4.0, 5.0, 6.0])
svec24 = shack_hartmann_sensor.slopes_to_vector(sx24, sy24)
sx24b, sy24b = shack_hartmann_sensor.vector_to_slopes(svec24)
assert np.allclose(sx24, sx24b) and np.allclose(sy24, sy24b), '[TC24] shack_hartmann_sensor.slopes_to_vector 双向转换一致性 FAILED'

# ---- TC25: data_io.write_xy_data / read_xy_data 往返一致性 ----
tmp25 = '/tmp/test_xy_104.txt'
data_io.write_xy_data(tmp25, [0.0, 1.0, 2.0], [3.0, 4.0, 5.0])
x25, y25 = data_io.read_xy_data(tmp25)
assert len(x25) == 3 and np.allclose(x25, [0.0, 1.0, 2.0]) and np.allclose(y25, [3.0, 4.0, 5.0]), '[TC25] data_io.write_xy_data / read_xy_data 往返一致性 FAILED'
import os as _os25
_os25.remove(tmp25)

# ---- TC26: data_io.write_zernike_coefficients / read_zernike_coefficients 往返一致性 ----
tmp26 = '/tmp/test_zern_104.txt'
data_io.write_zernike_coefficients(tmp26, np.array([0.5, -0.3, 0.1]))
c26 = data_io.read_zernike_coefficients(tmp26)
assert len(c26) == 3 and np.allclose(c26, [0.5, -0.3, 0.1]), '[TC26] data_io.write_zernike_coefficients / read_zernike_coefficients 往返一致性 FAILED'
import os as _os26
_os26.remove(tmp26)

# ---- TC27: data_io.write_subaperture_slopes / read_subaperture_slopes 往返一致性 ----
tmp27 = '/tmp/test_slope_104.txt'
data_io.write_subaperture_slopes(tmp27, np.array([0.1, 0.2]), np.array([0.3, 0.4]))
i27, sx27, sy27 = data_io.read_subaperture_slopes(tmp27)
assert len(i27) == 2 and np.allclose(sx27, [0.1, 0.2]) and np.allclose(sy27, [0.3, 0.4]), '[TC27] data_io.write_subaperture_slopes / read_subaperture_slopes 往返一致性 FAILED'
import os as _os27
_os27.remove(tmp27)

# ---- TC28: closed_loop_control.simulate_modal_control_loop 输出结构正确性 ----
np.random.seed(123)
cov28 = zernike_modes.kolmogorov_zernike_covariance(5, 1.0)
fs28, ms28, rh28, sh28 = closed_loop_control.simulate_modal_control_loop(
    n_modes=5, turb_covariance=cov28, Kp=0.5, Ki=0.1, bandwidth_hz=100.0,
    dt=1e-3, n_steps=10, noise_std=0.01
)
assert len(rh28) == 10 and len(sh28) == 10, '[TC28] closed_loop_control.simulate_modal_control_loop 输出结构正确性 FAILED'
assert 0.0 <= fs28 <= 1.0, '[TC28] closed_loop_control.simulate_modal_control_loop 输出结构正确性 FAILED'

# ---- TC29: closed_loop_control.parameter_sweep_optimization 最优参数在网格内 ----
def sim29(Kp, Ki, bw, dt, n_steps):
    return Kp * 0.1 + Ki * 0.1 + bw * 0.001
kp29 = np.array([0.1, 0.5])
ki29 = np.array([0.05, 0.2])
bw29 = np.array([50.0])
bp29, bs29, perf29 = closed_loop_control.parameter_sweep_optimization(kp29, ki29, bw29, sim29, dt=1e-3, n_steps=10)
assert bp29[0] in kp29 and bp29[1] in ki29 and bp29[2] in bw29, '[TC29] closed_loop_control.parameter_sweep_optimization 最优参数在网格内 FAILED'

# ---- TC30: main() 零参数集成测试返回值验证 ----
assert ret == 0, '[TC30] main() 零参数集成测试返回值验证 FAILED'
'''

# Write test_cases_inline.py
with open(os.path.join(out_dir, "test_cases_inline.py"), "w", encoding="utf-8") as f:
    f.write(test_cases_inline)

# Build test_main.py
# Replace the last line to save ret, seed, and append tests + summary (all unindented)
modified_main = main_content.replace(
    "if __name__ == '__main__':\n    sys.exit(main())",
    "if __name__ == '__main__':\n    ret = main()\n    np.random.seed(42)\n"
    + test_cases_inline
    + 'print("\\n全部 30 个测试通过!\\n")\nsys.exit(ret)'
)

# Add sys.path insert after imports
modified_main = modified_main.replace(
    "# 导入各模块",
    "# 添加原项目目录到模块搜索路径\nsys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\n\n# 导入各模块"
)

with open(os.path.join(out_dir, "test_main.py"), "w", encoding="utf-8") as f:
    f.write(modified_main)

print("Files generated successfully.")
