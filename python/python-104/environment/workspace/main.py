"""
main.py — 自适应光学波前校正系统的高保真数值模拟

统一入口, 零参数可运行.

执行流程:
  1. 系统参数初始化
  2. 生成大气湍流相位屏 (Kolmogorov + PME修正 + 光化学-热扰动)
  3. Shack-Hartmann波前传感器采样与斜率提取
  4. Zernike模态分解与zonal波前重构
  5. 变形镜面形计算与快速倾斜镜动力学响应
  6. 闭环PI控制迭代校正
  7. 光学传递函数、Strehl比、焦散奇点分析
  8. 控制参数扫描优化
  9. 结果输出与日志记录
"""

import os
import sys
import numpy as np

# 导入各模块
import data_io
import iterative_utils
import zernike_modes
import atmosphere_turbulence
import wavefront_propagation
import shack_hartmann_sensor
import wavefront_reconstruction
import optical_transfer
import deformable_mirror
import adaptive_sampling
import closed_loop_control


def main():
    print("=" * 70)
    print("Adaptive Optics Wavefront Correction System — High-Fidelity Simulation")
    print("=" * 70)

    # ============================================================
    # 1. 系统参数初始化
    # ============================================================
    print("\n[1/9] Initializing system parameters...")

    params = {
        'telescope_diameter_m': 1.0,
        'wavelength_m': 500e-9,
        'seeing_arcsec': 0.8,
        'grid_size': 128,
        'n_subapertures': 8,
        'n_zernike_modes': 21,
        'n_dm_actuators': 64,
        'fsm_bandwidth_hz': 100.0,
        'control_dt_ms': 0.5,
        'n_closed_loop_steps': 200,
        'seed': 42,
    }

    D = params['telescope_diameter_m']
    wavelength = params['wavelength_m']
    grid_size = params['grid_size']
    pixel_scale = D / grid_size
    n_subap = params['n_subapertures']
    n_modes = params['n_zernike_modes']
    n_actuators = params['n_dm_actuators']
    seed = params['seed']

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)

    data_io.log_system_parameters(
        os.path.join(output_dir, 'system_parameters.log'), params
    )
    print("  Parameters logged.")

    # ============================================================
    # 2. 生成大气湍流相位屏
    # ============================================================
    print("\n[2/9] Generating atmospheric turbulent phase screen...")

    phase_turb, r0, mask = atmosphere_turbulence.generate_turbulent_phase_screen(
        grid_size=grid_size,
        D_aperture=D,
        wavelength=wavelength,
        seeing=params['seeing_arcsec'],
        apply_pme_correction=True,
        apply_thermal_perturbation=True,
        seed=seed
    )

    # 计算Fried参数理论值
    Cn2_int = 1e-14 * D  # 简化的C_n^2积分
    r0_theory = atmosphere_turbulence.fried_parameter(wavelength, Cn2_int)
    print(f"  Fried parameter r0 = {r0:.4f} m (theory: {r0_theory:.4f} m)")
    print(f"  Phase RMS = {np.std(phase_turb[mask]):.4f} rad")

    # ============================================================
    # 3. Shack-Hartmann传感器采样
    # ============================================================
    print("\n[3/9] Shack-Hartmann wavefront sensor sampling...")

    subaps = shack_hartmann_sensor.generate_subaperture_grid(
        grid_size, n_subap, geometry='square'
    )
    print(f"  Number of subapertures: {len(subaps)}")

    sx, sy = shack_hartmann_sensor.compute_subaperture_slopes(
        phase_turb, subaps, pixel_scale,
        focal_length=0.1, noise_photon=0.005, noise_read=0.2
    )
    slope_vec = shack_hartmann_sensor.slopes_to_vector(sx, sy)
    print(f"  Slope vector norm: {np.linalg.norm(slope_vec):.4f}")

    data_io.write_subaperture_slopes(
        os.path.join(output_dir, 'subaperture_slopes.txt'), sx, sy
    )

    # ============================================================
    # 4. Zernike模态分解与波前重构
    # ============================================================
    print("\n[4/9] Zernike modal decomposition and wavefront reconstruction...")

    # TODO: Hole 3 - 连接Zernike基计算、模态分解与波前重构
    # 需要:
    #   1. 调用 zernike_modes.compute_zernike_basis 生成Zernike基,
    #      正确解包其返回值 (basis_flat, mask_z, x_vec, y_vec)
    #   2. 使用 zernike_modes.zernike_decompose 分解湍流相位
    #   3. 使用 wavefront_reconstruction.reconstruct_wavefront_modal 从斜率重构模态系数
    #   4. 使用 zernike_modes.zernike_reconstruct 重构相位屏
    #   5. 使用 wavefront_reconstruction.reconstruct_wavefront_zonal 进行zonal重构
    #   6. 使用 zernike_modes.zernike_coefficient_simplex_search 进行单形搜索
    #   7. 输出Zernike系数到文件
    # 注意: x_vec 和 y_vec 后续会被FSM倾斜镜计算使用, 必须正确定义
    # 注意: basis_flat 的格式必须与 zernike_modes.compute_zernike_basis 和
    #       wavefront_reconstruction.reconstruct_wavefront_modal 的约定一致
    raise NotImplementedError("Hole 3: 请实现 main.py 中Zernike基与波前重构的连接代码.")

    # ============================================================
    # 5. 变形镜与快速倾斜镜响应
    # ============================================================
    print("\n[5/9] Deformable mirror and fast steering mirror response...")

    dm = deformable_mirror.DeformableMirror(
        n_actuators=n_actuators,
        grid_size=grid_size,
        aperture_radius=D / 2.0,
        influence_sigma=0.08,
        use_magic_square_layout=True
    )

    # 计算DM电压 (简单映射: Zernike系数 -> 电压)
    R = dm.voltage_to_zernike_response(basis_flat, mask)
    voltages, _, _, _ = np.linalg.lstsq(R, coeffs_turb, rcond=None)
    voltages = np.clip(voltages, -1.0, 1.0)
    dm_surface = dm.compute_surface(voltages)

    # FSM双轴动力学响应
    fsm = deformable_mirror.FastSteeringMirrorDynamics(
        g=9.81, m1=0.01, m2=0.01, l1=0.05, l2=0.05,
        damping1=0.8, damping2=0.8, coupling=0.05
    )
    control_x = np.ones(500) * 0.01
    control_y = np.ones(500) * 0.005
    theta1_fsm, theta2_fsm = fsm.simulate_response(control_x, control_y, dt=1e-4)
    fsm_tilt_surface = theta1_fsm[-1] * x_vec[None, :] + theta2_fsm[-1] * y_vec[:, None]
    fsm_tilt_surface = fsm_tilt_surface[:grid_size, :grid_size]

    total_correction = dm_surface + fsm_tilt_surface
    phase_corrected = phase_turb - total_correction
    phase_corrected[mask] -= np.mean(phase_corrected[mask])

    strehl_before = wavefront_propagation.compute_strehl_ratio(phase_turb, wavelength, mask)
    strehl_after = wavefront_propagation.compute_strehl_ratio(phase_corrected, wavelength, mask)
    print(f"  Strehl ratio (before correction): {strehl_before:.4f}")
    print(f"  Strehl ratio (after correction):  {strehl_after:.4f}")

    # ============================================================
    # 6. 闭环PI控制迭代校正
    # ============================================================
    print("\n[6/9] Closed-loop PI control iteration...")

    turb_cov = zernike_modes.kolmogorov_zernike_covariance(n_modes, D / r0)
    dt_control = params['control_dt_ms'] * 1e-3
    n_steps = params['n_closed_loop_steps']

    final_strehl, mean_strehl, res_hist, str_hist = closed_loop_control.simulate_modal_control_loop(
        n_modes=n_modes,
        turb_covariance=turb_cov,
        Kp=0.6,
        Ki=0.15,
        bandwidth_hz=params['fsm_bandwidth_hz'],
        dt=dt_control,
        n_steps=n_steps,
        noise_std=0.02
    )

    print(f"  Final Strehl (closed-loop): {final_strehl:.4f}")
    print(f"  Mean Strehl (closed-loop):  {mean_strehl:.4f}")
    print(f"  Final residual RMS: {res_hist[-1]:.4f}")

    # HH控制电路响应验证
    hh = closed_loop_control.HHControlCircuit(C=1.0, dt=1e-5)
    for _ in range(1000):
        hh.step(75.0 if hh.V < -50 else 0.0)
    print(f"  HH circuit final V: {hh.V:.2f} mV")

    # ============================================================
    # 7. 光学传递函数与焦散分析
    # ============================================================
    print("\n[7/9] Optical transfer function and caustic singularity analysis...")

    P = optical_transfer.compute_pupil_function(grid_size, mask, phase_corrected)
    otf = optical_transfer.compute_otf_from_pupil(P)
    mtf = optical_transfer.compute_mtf_from_otf(otf)

    psf, x_coords = optical_transfer.compute_psf_from_pupil(
        P, pixel_scale, wavelength, focal_length=10.0
    )
    ee50 = optical_transfer.compute_encircled_energy(psf, x_coords, radius=0.5e-6)
    print(f"  Encircled energy (0.5 um): {ee50:.4f}")

    # 焦散奇点检测
    singularity, det_H = wavefront_propagation.detect_caustic_singularities(
        phase_turb, pixel_scale, mask
    )
    n_singularities = np.sum(singularity)
    print(f"  Caustic singularities detected: {n_singularities}")

    # 焦散线密度 (几何模拟)
    caustic_lines = wavefront_propagation.caustic_line_density(n=50, m=7)
    print(f"  Caustic network lines: {len(caustic_lines)}")

    # 四面体相位矩
    moments = optical_transfer.tetrahedral_phase_moments(phase_turb, mask, max_order=2)
    print(f"  Phase moments: M_00={moments.get((0,0),0):.4f}, M_10={moments.get((1,0),0):.4f}, M_01={moments.get((0,1),0):.4f}")

    # ============================================================
    # 8. 控制参数扫描优化
    # ============================================================
    print("\n[8/9] Control parameter sweep optimization...")

    def simulation_wrapper(Kp, Ki, bw, dt, n_steps):
        fs, ms, _, _ = closed_loop_control.simulate_modal_control_loop(
            n_modes=n_modes, turb_covariance=turb_cov,
            Kp=Kp, Ki=Ki, bandwidth_hz=bw, dt=dt, n_steps=n_steps, noise_std=0.02
        )
        return ms

    Kp_grid = np.linspace(0.1, 1.0, 5)
    Ki_grid = np.linspace(0.05, 0.3, 5)
    bw_grid = np.array([50.0, 100.0, 200.0])

    best_params, best_strehl, perf = closed_loop_control.parameter_sweep_optimization(
        Kp_grid, Ki_grid, bw_grid, simulation_wrapper, dt=dt_control, n_steps=100
    )
    print(f"  Optimal parameters: Kp={best_params[0]:.2f}, Ki={best_params[1]:.2f}, BW={best_params[2]:.1f} Hz")
    print(f"  Optimal mean Strehl: {best_strehl:.4f}")

    # ============================================================
    # 9. 自适应采样与CVT
    # ============================================================
    print("\n[9/9] Adaptive sampling and CVT optimization...")

    # CVT圆盘采样
    cvt_points = adaptive_sampling.cvt_disk_uniform(
        n_generators=32, radius=D / 2.0, n_iterations=20, seed=seed
    )
    print(f"  CVT generators: {len(cvt_points)}")

    # 自适应相位采样
    adaptive_pts = adaptive_sampling.adaptive_phase_sampling(
        phase_turb, mask, n_target_points=40, n_iterations=15
    )
    print(f"  Adaptive sampling points: {len(adaptive_pts)}")

    adaptive_sampling.log_sampling_info(
        os.path.join(output_dir, 'adaptive_sampling_points.txt'),
        adaptive_pts, iteration=15, residual=res_hist[-1]
    )

    # Collatz迭代步长分析
    for res_val in [1.0, 0.1, 0.01, 0.001]:
        step = iterative_utils.collatz_step_size(res_val, base_step=0.5)
        print(f"  Collatz step (res={res_val}): {step:.6f}")

    # ============================================================
    # 结果汇总
    # ============================================================
    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"  - system_parameters.log")
    print(f"  - subaperture_slopes.txt")
    print(f"  - zernike_coefficients_turbulence.txt")
    print(f"  - adaptive_sampling_points.txt")
    print(f"\nKey metrics:")
    print(f"  Initial Strehl:     {strehl_before:.4f}")
    print(f"  Corrected Strehl:   {strehl_after:.4f}")
    print(f"  Closed-loop Strehl: {final_strehl:.4f}")
    print(f"  Optimal (Kp,Ki,BW): ({best_params[0]:.2f}, {best_params[1]:.2f}, {best_params[2]:.1f})")
    print(f"  Caustic singularities: {n_singularities}")
    print("=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
