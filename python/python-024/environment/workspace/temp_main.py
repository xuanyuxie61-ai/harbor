
import sys
import numpy as np


from harris_equilibrium import HarrisEquilibrium, demo_harris
from continuation_solver import ContinuationSolver, demo_mhd_continuation
from fem_assembler import FEM1DAssembler, STtoGEAssembler, demo_fem
from mhd_stability import (HankelSolver, IntegerRREF,
                            MHDStabilityAnalyzer, demo_stability)
from resistivity_evolution import (AnomalousResistivity,
                                    WaveDampingModel, demo_resistivity)
from particle_acceleration import (MonteCarloParticleAccelerator,
                                    NonlinearOrbitTracker, demo_particles)
from field_rotation import Quaternion, MagneticTopology, demo_field_rotation
from periodic_interpolation import (TrigonometricInterpolator,
                                     GridReshaper, demo_periodic)
from wedge_flux import WedgeIntegrals, demo_wedge


def run_full_simulation():
    print("=" * 70)
    print("  太阳耀斑磁重联数值模拟平台")
    print("  等离子体物理：磁重联与太阳耀斑")
    print("=" * 70)




    print("\n[模块 1] Harris 电流片平衡态构造")
    eq = HarrisEquilibrium(
        B0=1.0e-2,
        lambda_cs=5.0e4,
        B_guide=2.0e-3,
        p0=2.0e-3,
        T_plasma=2.0e6,
        rho_inf=1.0e-12,
        y_max=3.0e5
    )
    y = np.linspace(-eq.y_max, eq.y_max, 129)
    B = eq.B_field(y)
    p = eq.pressure(y)
    J = eq.current_density(y)
    rho = eq.mass_density(y)
    va = eq.alfven_speed(y)
    beta = eq.plasma_beta(y)
    shear = eq.magnetic_shear(y)

    center_idx = len(y) // 2
    print(f"  电流片中心物理量:")
    print(f"    B_x(0)     = {B[center_idx, 0]:.3e} T")
    print(f"    p(0)       = {p[center_idx]:.3e} Pa")
    print(f"    J_z(0)     = {J[center_idx, 2]:.3e} A/m^2")
    print(f"    rho(0)     = {rho[center_idx]:.3e} kg/m^3")
    print(f"    v_A(0)     = {va[center_idx]:.3e} m/s")
    print(f"    beta(0)    = {beta[center_idx]:.3f}")
    print(f"    shear(0)   = {shear[center_idx]:.3e} T/m")


    nodes, elements = eq.generate_quadrilateral_mesh(nx=16, ny=32)
    field_interp = eq.bilinear_interpolate_on_mesh(nodes, elements, p, y)
    print(f"  生成四边形网格: {len(nodes)} 节点, {len(elements)} 单元")
    print(f"  压强场插值范围: [{np.min(field_interp):.3e}, {np.max(field_interp):.3e}] Pa")




    print("\n[模块 2] 伪弧长延拓法追踪 MHD 平衡态分支")
    xs, params = demo_mhd_continuation()
    print(f"  追踪到 {len(xs)} 个平衡点")
    print(f"  参数范围: eta = [{min(params):.4f}, {max(params):.4f}]")




    print("\n[模块 3] 有限元稀疏矩阵组装")
    demo_fem()




    print("\n[模块 4] MHD 线性稳定性与撕裂模分析")
    demo_stability()




    print("\n[模块 5] 反常电阻率反应-扩散演化")
    demo_resistivity()




    print("\n[模块 6] 高能粒子加速与非线性轨道")
    demo_particles()




    print("\n[模块 7] 磁场拓扑四元数旋转与对称性")
    demo_field_rotation()




    print("\n[模块 8] 周期性边界三角插值与数据重排")
    demo_periodic()




    print("\n[模块 9] 楔形区域精确通量计算")
    demo_wedge()




    print("\n" + "=" * 70)
    print("[综合诊断] 磁重联关键物理量估算")
    print("=" * 70)


    eta_spitzer = 1.0e-6
    v_in = 1.0e3
    v_out = va[center_idx]

    if v_out > 0:
        sp_rate = np.sqrt(eta_spitzer / (eq.lambda_cs * v_out))
        print(f"  Sweet-Parker 重联率: {sp_rate:.3e}")
    else:
        print(f"  Sweet-Parker 重联率: N/A (v_A=0)")


    petschek_rate = np.pi / (8.0 * np.log(1.0e2))
    print(f"  Petschek 快速重联率（理论参考）: {petschek_rate:.3e}")


    magnetic_energy = (eq.B0 ** 2) / (2.0 * 4.0 * np.pi * 1e-7)
    volume_estimate = (2.0 * eq.y_max) ** 2 * eq.lambda_cs
    total_energy = magnetic_energy * volume_estimate
    print(f"  电流片磁能密度: {magnetic_energy:.3e} J/m^3")
    print(f"  估算总储能: {total_energy:.3e} J")
    print(f"  等效耀斑级别: {total_energy / 1.0e25:.2f} X-class")

    print("\n" + "=" * 70)
    print("  模拟完成，所有模块运行正常。")
    print("=" * 70)


def main():
    try:
        run_full_simulation()
        return 0
    except Exception as e:
        print(f"\n[错误] 模拟过程中发生异常: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    main()
