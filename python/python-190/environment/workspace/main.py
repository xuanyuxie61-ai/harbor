
import os
import sys
import time
import numpy as np


_project_dir = os.path.dirname(os.path.abspath(__file__))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)


def main():
    print("=" * 70)
    print("物理信息生成对抗网络（PI-GAN）三维流场合成系统")
    print("Physics-Informed GAN for 3D Incompressible Flow Generation")
    print("=" * 70)
    print()

    t_start = time.time()




    print("[1/6] 正在生成基于 Ethier 精确解的真实流场训练数据...")
    from navier_stokes_exact import generate_training_data, uvwp_ethier, ns_residual
    coords, states = generate_training_data(nx=6, ny=6, nz=6,
                                            a=np.pi / 4.0, d=np.pi / 2.0,
                                            t_val=0.05)
    N = coords.shape[0]
    print(f"      生成完成：空间网格 6×6×6，共 {N} 个时空采样点。")
    print(f"      坐标范围：x,y,z ∈ [-1,1]，t = 0.05")
    print()




    print("[2/6] 正在初始化纯 NumPy 坐标条件 GAN 网络...")
    from gan_numpy import Generator, Discriminator
    latent_dim = 8
    gen = Generator(latent_dim=latent_dim, coord_dim=4, hidden_dim=32,
                    output_dim=4, seed=42)
    disc = Discriminator(input_dim=8, hidden_dim=32, seed=43)
    print(f"      生成器：输入维度 {latent_dim+4} → 隐藏层 32 → 32 → 输出 4")
    print(f"      判别器：输入维度 8 → 隐藏层 32 → 16 → 输出 1 (sigmoid)")
    print()




    print("[3/6] 开始对抗训练（约 120 轮，每轮 2 个 batch）...")
    from training_engine import train_pigan
    results = train_pigan(epochs=120, batch_size=32,
                          lr_g=0.002, lr_d=0.002,
                          lambda_phys=0.5, lambda_equiv=0.1,
                          nx=6, ny=6, nz=6,
                          latent_dim=latent_dim, seed=42)
    print(f"      训练完成。")
    print(f"      最终判别器平均损失: {results['history']['loss_d'][-1]:.6f}")
    print(f"      最终生成器平均损失: {results['history']['loss_g'][-1]:.6f}")
    print(f"      最终物理代理损失:   {results['history']['phys_loss'][-1]:.6f}")
    print()




    print("[4/6] 正在进行四元数 SO(3) 旋转等变性验证...")
    from quaternion_equivariance import rotate_velocity_field, rotation_axis_to_quat, rotate_vector_by_quat

    test_coords = coords[:10, :3]
    test_z = np.random.randn(1, latent_dim)
    test_z_batch = np.tile(test_z, (10, 1))
    pred_vel = gen.forward(test_z_batch, coords[:10])[:, 0:3]
    axis = np.array([1.0, 0.0, 0.0])
    angle = np.pi / 6.0
    coords_rot, vel_rot = rotate_velocity_field(test_coords, pred_vel, axis, angle)

    rot_coords_4d = np.concatenate([coords_rot, coords[:10, 3:4]], axis=1)
    pred_vel_rot = gen.forward(test_z_batch, rot_coords_4d)[:, 0:3]

    equiv_error = float(np.mean((vel_rot - pred_vel_rot) ** 2))
    print(f"      旋转轴: {axis}, 旋转角: {angle:.4f} rad")
    print(f"      等变性误差 (MSE): {equiv_error:.6f}")
    print()




    print("[5/6] 正在进行高阶数值评估...")
    from training_engine import evaluate_with_geometry
    metrics = evaluate_with_geometry(results, nx=6, ny=6)
    print(f"      Wasserstein 近似距离: {metrics['wasserstein_approx']:.6f}")
    print(f"      球面速度模积分:       {metrics['sphere_speed_integral']:.6f}")
    print(f"      人体轮廓三角剖分:")
    print(f"        - 三角形数量: {metrics['mesh_num_triangles']}")
    print(f"        - 节点数量:   {metrics['mesh_num_nodes']}")
    print(f"        - 最小角:     {metrics['mesh_min_angle_deg']:.2f}°")
    print()


    from triangle_quadrature import integrate_over_triangle, triangle_area
    def f_test(pts):
        return pts[:, 0] ** 2 + pts[:, 1] ** 2
    v1 = np.array([0.0, 0.0])
    v2 = np.array([1.0, 0.0])
    v3 = np.array([0.0, 1.0])
    quad_val = integrate_over_triangle(f_test, v1, v2, v3, degree=5)
    exact_val = 1.0 / 6.0
    print(f"      三角形求积验证: 数值 = {quad_val:.8f}, 精确 = {exact_val:.8f}, 误差 = {abs(quad_val-exact_val):.2e}")


    from sphere_quad import integrate_on_sphere
    sphere_val = integrate_on_sphere(lambda x: 1.0, rule="icos1v")
    print(f"      球面求积验证: 数值 = {sphere_val:.8f}, 精确 = {4*np.pi:.8f}, 误差 = {abs(sphere_val-4*np.pi):.2e}")


    from special_functions import clausen, clausen_activation
    cl_val = clausen(np.pi / 2.0)
    print(f"      Clausen 函数验证: Cl_2(π/2) = {cl_val:.8f} (参考: 0.91596559...)")
    print()




    print("[6/6] 正在进行 Hooke-Jeeves 超参数微调演示...")
    from hooke_jeeves import optimize_gan_hyperparams

    def dummy_loss_evaluator(params):


        target = np.array([0.002, 0.002, 0.5, 0.1])
        diff = params - target
        return float(np.sum(diff ** 2) + 0.01 * np.random.rand())

    init_params = np.array([0.001, 0.003, 0.3, 0.2])
    best_params, hh_history = optimize_gan_hyperparams(
        init_params, dummy_loss_evaluator, rho=0.5, eps=1e-4, itermax=20)
    print(f"      初始超参数: {init_params}")
    print(f"      优化后超参数: {best_params}")
    print(f"      优化迭代次数: {len(hh_history)}")
    print()




    print("=" * 70)
    print("训练与评估总结")
    print("=" * 70)
    print(f"总运行时间: {time.time() - t_start:.2f} 秒")
    print(f"最终生成场 MSE (vs 真实 Ethier 解): {results['final_mse']:.6f}")
    print(f"等变性验证误差: {equiv_error:.6f}")
    print(f"球面速度模积分: {metrics['sphere_speed_integral']:.6f}")
    print(f"Wasserstein 近似距离: {metrics['wasserstein_approx']:.6f}")
    print()
    print("所有核心模块已成功运行，无报错。")
    print("=" * 70)


    report_path = os.path.join(_project_dir, "training_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("PI-GAN 训练报告\n")
        f.write("=" * 50 + "\n")
        f.write(f"最终 MSE: {results['final_mse']:.6f}\n")
        f.write(f"等变误差: {equiv_error:.6f}\n")
        f.write(f"Wasserstein 距离: {metrics['wasserstein_approx']:.6f}\n")
        f.write(f"球面积分: {metrics['sphere_speed_integral']:.6f}\n")
        f.write(f"三角形求积误差: {abs(quad_val-exact_val):.2e}\n")
        f.write(f"球面求积误差: {abs(sphere_val-4*np.pi):.2e}\n")
        f.write(f"Clausen(π/2): {cl_val:.8f}\n")
        f.write(f"运行时间: {time.time() - t_start:.2f} 秒\n")
    print(f"简要报告已保存至: {report_path}")


if __name__ == "__main__":
    main()
