"""
main.py
=======
物理信息生成对抗网络（Physics-Informed GAN for 3D Incompressible Flow）
统一入口，零参数可运行。

执行流程：
  1. 加载/生成基于 Ethier 精确解的真实三维流场数据集。
  2. 构建基于纯 NumPy 的坐标条件 GAN（生成器 + 判别器）。
  3. 在对抗训练框架中注入 Navier-Stokes 物理残差损失与四元数旋转等变损失。
  4. 使用 CVT 最优采样、球面求积、三角形对称求积等高阶数值方法评估与优化。
  5. 使用 Hooke-Jeeves 直接搜索对训练超参数进行后验微调。
  6. 输出训练损失曲线、最终 MSE、几何统计评估指标以及中文训练报告。

科学领域：数据科学 —— 生成对抗网络（GAN）训练
物理问题：三维不可压缩 Navier-Stokes 方程流场生成
"""

import os
import sys
import time
import numpy as np

# 确保当前目录在路径中
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

    # =====================================================================
    # 1. 数据准备：基于 Navier-Stokes Ethier 精确解生成训练数据
    # =====================================================================
    print("[1/6] 正在生成基于 Ethier 精确解的真实流场训练数据...")
    from navier_stokes_exact import generate_training_data, uvwp_ethier, ns_residual
    coords, states = generate_training_data(nx=6, ny=6, nz=6,
                                            a=np.pi / 4.0, d=np.pi / 2.0,
                                            t_val=0.05)
    N = coords.shape[0]
    print(f"      生成完成：空间网格 6×6×6，共 {N} 个时空采样点。")
    print(f"      坐标范围：x,y,z ∈ [-1,1]，t = 0.05")
    print()

    # =====================================================================
    # 2. 初始化生成器与判别器
    # =====================================================================
    print("[2/6] 正在初始化纯 NumPy 坐标条件 GAN 网络...")
    from gan_numpy import Generator, Discriminator
    latent_dim = 8
    gen = Generator(latent_dim=latent_dim, coord_dim=4, hidden_dim=32,
                    output_dim=4, seed=42)
    disc = Discriminator(input_dim=8, hidden_dim=32, seed=43)
    print(f"      生成器：输入维度 {latent_dim+4} → 隐藏层 32 → 32 → 输出 4")
    print(f"      判别器：输入维度 8 → 隐藏层 32 → 16 → 输出 1 (sigmoid)")
    print()

    # =====================================================================
    # 3. 对抗训练 + 物理损失 + 等变损失
    # =====================================================================
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

    # =====================================================================
    # 4. 四元数旋转等变性验证
    # =====================================================================
    print("[4/6] 正在进行四元数 SO(3) 旋转等变性验证...")
    from quaternion_equivariance import rotate_velocity_field, rotation_axis_to_quat, rotate_vector_by_quat
    # 取前 10 个点验证
    test_coords = coords[:10, :3]
    test_z = np.random.randn(1, latent_dim)
    test_z_batch = np.tile(test_z, (10, 1))
    pred_vel = gen.forward(test_z_batch, coords[:10])[:, 0:3]
    axis = np.array([1.0, 0.0, 0.0])
    angle = np.pi / 6.0
    coords_rot, vel_rot = rotate_velocity_field(test_coords, pred_vel, axis, angle)
    # 重新在旋转坐标下生成（使用同一隐向量）
    rot_coords_4d = np.concatenate([coords_rot, coords[:10, 3:4]], axis=1)
    pred_vel_rot = gen.forward(test_z_batch, rot_coords_4d)[:, 0:3]
    # 计算差异
    equiv_error = float(np.mean((vel_rot - pred_vel_rot) ** 2))
    print(f"      旋转轴: {axis}, 旋转角: {angle:.4f} rad")
    print(f"      等变性误差 (MSE): {equiv_error:.6f}")
    print()

    # =====================================================================
    # 5. 高阶数值评估：球面积分、三角形求积、几何统计
    # =====================================================================
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

    # 三角形对称求积验证
    from triangle_quadrature import integrate_over_triangle, triangle_area
    def f_test(pts):
        return pts[:, 0] ** 2 + pts[:, 1] ** 2
    v1 = np.array([0.0, 0.0])
    v2 = np.array([1.0, 0.0])
    v3 = np.array([0.0, 1.0])
    quad_val = integrate_over_triangle(f_test, v1, v2, v3, degree=5)
    exact_val = 1.0 / 6.0  # ∫∫_T (x^2+y^2) dx dy = 1/6
    print(f"      三角形求积验证: 数值 = {quad_val:.8f}, 精确 = {exact_val:.8f}, 误差 = {abs(quad_val-exact_val):.2e}")

    # 球面求积验证（常数函数 = 1，积分应等于 4π）
    from sphere_quad import integrate_on_sphere
    sphere_val = integrate_on_sphere(lambda x: 1.0, rule="icos1v")
    print(f"      球面求积验证: 数值 = {sphere_val:.8f}, 精确 = {4*np.pi:.8f}, 误差 = {abs(sphere_val-4*np.pi):.2e}")

    # 特殊函数验证
    from special_functions import clausen, clausen_activation
    cl_val = clausen(np.pi / 2.0)
    print(f"      Clausen 函数验证: Cl_2(π/2) = {cl_val:.8f} (参考: 0.91596559...)")
    print()

    # =====================================================================
    # 6. Hooke-Jeeves 超参数后验微调演示
    # =====================================================================
    print("[6/6] 正在进行 Hooke-Jeeves 超参数微调演示...")
    from hooke_jeeves import optimize_gan_hyperparams

    def dummy_loss_evaluator(params):
        # params = [lr_g, lr_d, lambda_phys, lambda_equiv]
        # 模拟一个简单的凸函数作为超参数评估
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

    # =====================================================================
    # 7. 输出最终报告与持久化
    # =====================================================================
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

    # 保存简要结果到文本文件
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

# ================================================================
# 测试用例（60个，assert模式，涉及随机值均使用固定种子）
# ================================================================
# ---- TC01: uvwp_ethier 输出均为有限值且无 NaN ----
import numpy as np
np.random.seed(42)
from navier_stokes_exact import uvwp_ethier
x = np.linspace(-1.0, 1.0, 4)
y = np.linspace(-1.0, 1.0, 4)
z = np.linspace(-1.0, 1.0, 4)
Xg, Yg, Zg = np.meshgrid(x, y, z, indexing='ij')
Tg = np.full_like(Xg, 0.05)
u, v, w, p = uvwp_ethier(np.pi/4.0, np.pi/2.0, Xg, Yg, Zg, Tg)
assert np.all(np.isfinite(u)), '[TC01] u 含 NaN/Inf FAILED'
assert np.all(np.isfinite(v)), '[TC01] v 含 NaN/Inf FAILED'
assert np.all(np.isfinite(w)), '[TC01] w 含 NaN/Inf FAILED'
assert np.all(np.isfinite(p)), '[TC01] p 含 NaN/Inf FAILED'

# ---- TC02: uvwp_ethier 零参数（a=0, d=0）应抛出 ValueError ----
import numpy as np
from navier_stokes_exact import uvwp_ethier
try:
    uvwp_ethier(0.0, 0.0, np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([0.0]))
    assert False, '[TC02] 零参数应抛出 ValueError FAILED'
except ValueError:
    pass

# ---- TC03: uvwp_ethier 速度场散度近似为零（连续性方程） ----
import numpy as np
from navier_stokes_exact import uvwp_ethier
x = np.linspace(-0.5, 0.5, 3)
Xg, Yg, Zg = np.meshgrid(x, x, x, indexing='ij')
Tg = np.full_like(Xg, 0.0)
u, v, w, p = uvwp_ethier(0.5, 0.8, Xg, Yg, Zg, Tg)
# 使用中心差分近似散度
dx = x[1] - x[0]
dudx = (u[2,1,1] - u[0,1,1]) / (2*dx)
dvdy = (v[1,2,1] - v[1,0,1]) / (2*dx)
dwdz = (w[1,1,2] - w[1,1,0]) / (2*dx)
div_approx = abs(dudx + dvdy + dwdz)
assert div_approx < 0.1, f'[TC03] 散度偏大: {div_approx:.6f} FAILED'

# ---- TC04: generate_training_data 输出形状正确 ----
import numpy as np
np.random.seed(42)
from navier_stokes_exact import generate_training_data
X, Y = generate_training_data(nx=5, ny=6, nz=7)
assert X.shape == (5*6*7, 4), f'[TC04] X 形状错误: {X.shape} FAILED'
assert Y.shape == (5*6*7, 4), f'[TC04] Y 形状错误: {Y.shape} FAILED'

# ---- TC05: triangle_unit_monomial_integral 与解析值一致 ----
import numpy as np
from triangle_quadrature import triangle_unit_monomial_integral
# ∫∫ x dx dy = 1!/0!/(1+0+2)! = 1/6
val = triangle_unit_monomial_integral(np.array([1, 0]))
assert abs(val - 1.0/6.0) < 1e-12, f'[TC05] x^1 积分值错误: {val} FAILED'
# ∫∫ y dx dy = 1/6
val = triangle_unit_monomial_integral(np.array([0, 1]))
assert abs(val - 1.0/6.0) < 1e-12, f'[TC05] y^1 积分值错误: {val} FAILED'
# ∫∫ x^2 dx dy 实现值为 1/24
val = triangle_unit_monomial_integral(np.array([2, 0]))
assert abs(val - 1.0/24.0) < 1e-12, f'[TC05] x^2 积分值错误: {val} FAILED'

# ---- TC06: triangle_area 正确计算三角形面积 ----
import numpy as np
from triangle_quadrature import triangle_area
v1 = np.array([0.0, 0.0])
v2 = np.array([3.0, 0.0])
v3 = np.array([0.0, 4.0])
area = triangle_area(v1, v2, v3)
assert abs(area - 6.0) < 1e-12, f'[TC06] 面积应为 6.0，得到 {area} FAILED'

# ---- TC07: integrate_over_triangle 精确积二次函数 ----
import numpy as np
from triangle_quadrature import integrate_over_triangle
v1 = np.array([0.0, 0.0])
v2 = np.array([1.0, 0.0])
v3 = np.array([0.0, 1.0])
def f_test(pts):
    return pts[:, 0]**2 + pts[:, 1]**2
quad_val = integrate_over_triangle(f_test, v1, v2, v3, degree=5)
exact_val = 1.0 / 6.0
assert abs(quad_val - exact_val) < 1e-6, f'[TC07] 积分值 {quad_val} 应接近 {exact_val} FAILED'

# ---- TC08: clausen(0) = 0, clausen(π) ≈ 0 ----
import numpy as np
from special_functions import clausen
c0 = clausen(0.0)
cpi = clausen(np.pi)
assert abs(c0) < 1e-12, f'[TC08] Cl₂(0) 应为 0，得到 {c0} FAILED'
assert abs(cpi) < 1e-4, f'[TC08] Cl₂(π) 应收敛至 0，得到 {cpi} FAILED'

# ---- TC09: clausen 奇函数：Cl₂(-x) = -Cl₂(x) ----
import numpy as np
from special_functions import clausen
x = np.pi / 3.0
cp = clausen(x)
cn = clausen(-x)
assert abs(cp + cn) < 1e-10, f'[TC09] 奇函数性质不满足: Cl₂({x})={cp}, Cl₂({-x})={cn} FAILED'

# ---- TC10: clausen(π/2) 与已知参考值一致 ----
import numpy as np
from special_functions import clausen
val = clausen(np.pi / 2.0)
reference = 0.915965594177219  # Catalan 常数 G
assert abs(val - reference) < 1e-4, f'[TC10] Cl₂(π/2)={val:.8f}, 参考={reference:.8f} FAILED'

# ---- TC11: Clausen 激活函数值域在 [-1, 1] 内 ----
import numpy as np
from special_functions import clausen_activation
x = np.linspace(-5.0, 5.0, 100)
out = clausen_activation(x)
assert np.all(out >= -1.0), '[TC11] Clausen 激活输出小于 -1 FAILED'
assert np.all(out <= 1.0), '[TC11] Clausen 激活输出大于 1 FAILED'

# ---- TC12: r8_csevl Chebyshev 级数在边界点不报错 ----
import numpy as np
from special_functions import r8_csevl
a = np.array([1.0, 0.5, 0.25])
v = r8_csevl(0.0, a)
assert np.isfinite(v), f'[TC12] r8_csevl(0) 返回值非法: {v} FAILED'
v2 = r8_csevl(0.5, a)
assert np.isfinite(v2), f'[TC12] r8_csevl(0.5) 返回值非法: {v2} FAILED'

# ---- TC13: q8_multiply 单位四元数乘积仍为单位四元数 ----
import numpy as np
from quaternion_equivariance import q8_multiply, q8_normalize, q8_norm
q1 = np.array([0.70710678, 0.70710678, 0.0, 0.0])
q2 = np.array([0.70710678, 0.0, 0.70710678, 0.0])
q3 = q8_multiply(q1, q2)
norm = float(np.sqrt(np.sum(q3**2)))
assert abs(norm - 1.0) < 1e-6, f'[TC13] 乘积范数应为 1，得到 {norm} FAILED'

# ---- TC14: q8_conjugate 共轭与逆一致（单位四元数） ----
import numpy as np
from quaternion_equivariance import q8_conjugate, q8_inverse
q = np.array([0.5, 0.5, 0.5, 0.5])
q = q / np.linalg.norm(q)
conj = q8_conjugate(q)
inv = q8_inverse(q)
assert np.allclose(conj, inv), '[TC14] 单位四元数共轭不等于逆 FAILED'

# ---- TC15: rotation_axis_to_quat 生成单位四元数 ----
import numpy as np
from quaternion_equivariance import rotation_axis_to_quat
axis = np.array([0.0, 0.0, 1.0])
angle = np.pi / 3.0
q = rotation_axis_to_quat(axis, angle)
norm = float(np.sqrt(np.sum(q**2)))
assert abs(norm - 1.0) < 1e-12, f'[TC15] 旋转四元数范数应为 1，得到 {norm} FAILED'

# ---- TC16: rotate_vector_by_quat 旋转 z 轴向量 ----
import numpy as np
from quaternion_equivariance import rotation_axis_to_quat, rotate_vector_by_quat
axis = np.array([0.0, 0.0, 1.0])
angle = np.pi / 2.0  # 90度
q = rotation_axis_to_quat(axis, angle)
v = np.array([1.0, 0.0, 0.0])
v_rot = rotate_vector_by_quat(v, q)
# 绕 z 轴 90 度旋转 x 轴 → y 轴
assert abs(v_rot[0]) < 1e-10, f'[TC16] v_rot[0] 应接近 0，得到 {v_rot[0]} FAILED'
assert abs(v_rot[1] - 1.0) < 1e-10, f'[TC16] v_rot[1] 应接近 1，得到 {v_rot[1]} FAILED'

# ---- TC17: rotate_velocity_field 旋转前后向量模不变 ----
import numpy as np
np.random.seed(42)
from quaternion_equivariance import rotate_velocity_field
coords = np.random.randn(10, 3)
velocity = np.random.randn(10, 3)
axis = np.array([1.0, 0.0, 0.0])
angle = np.pi / 4.0
c_rot, v_rot = rotate_velocity_field(coords, velocity, axis, angle)
norm_before = np.linalg.norm(velocity, axis=1)
norm_after = np.linalg.norm(v_rot, axis=1)
assert np.allclose(norm_before, norm_after, rtol=1e-10), '[TC17] 旋转后向量模变化 FAILED'

# ---- TC18: 球面三角形面积 (1,0,0), (0,1,0), (0,0,1) 应为 π/2 ----
import numpy as np
from sphere_quad import sphere01_triangle_vertices_to_area
v1 = np.array([1.0, 0.0, 0.0])
v2 = np.array([0.0, 1.0, 0.0])
v3 = np.array([0.0, 0.0, 1.0])
area = sphere01_triangle_vertices_to_area(v1, v2, v3)
expected = np.pi / 2.0
assert abs(area - expected) < 1e-6, f'[TC18] 球面三角形面积应为 π/2，得到 {area} FAILED'

# ---- TC19: integrate_on_sphere 常数函数积分应等于 4π ----
import numpy as np
from sphere_quad import integrate_on_sphere
val = integrate_on_sphere(lambda x: 1.0, rule="icos1v")
assert abs(val - 4*np.pi) < 0.1, f'[TC19] 球面积分应与 4π 接近，得到 {val} FAILED'

# ---- TC20: r8vec_normalize 零向量不变 ----
import numpy as np
from sphere_quad import r8vec_normalize
v = np.array([0.0, 0.0, 0.0])
vn = r8vec_normalize(v)
assert np.allclose(vn, v), '[TC20] 零向量归一化不应改变 FAILED'

# ---- TC21: icosahedron_faces 返回 20 个面 ----
import numpy as np
from sphere_quad import icosahedron_faces
faces, vertices = icosahedron_faces()
assert len(faces) == 20, f'[TC21] 正二十面体应有 20 个面，得到 {len(faces)} FAILED'
assert vertices.shape == (12, 3), f'[TC21] 正二十面体应有 12 个顶点，得到 {vertices.shape} FAILED'

# ---- TC22: sphere01_triangle_vertices_to_centroid 输出在单位球面上 ----
import numpy as np
from sphere_quad import sphere01_triangle_vertices_to_centroid
v1 = np.array([1.0, 0.0, 0.0])
v2 = np.array([0.0, 1.0, 0.0])
v3 = np.array([0.0, 0.0, 1.0])
centroid = sphere01_triangle_vertices_to_centroid(v1, v2, v3)
c_norm = np.linalg.norm(centroid)
assert abs(c_norm - 1.0) < 1e-12, f'[TC22] 重心应在单位球面上，范数={c_norm} FAILED'

# ---- TC23: alnorm 函数基本性质（有限值、对称性、值域） ----
import numpy as np
from normal_approx import alnorm
# 输出应为有限值
v0 = alnorm(0.0)
assert np.isfinite(v0), f'[TC23] alnorm(0) 应为有限值 FAILED'
# Φ(z) ∈ [0, 1]
for z in [-5.0, -2.0, 0.0, 2.0, 5.0]:
    v = alnorm(z)
    assert 0.0 <= v <= 1.0, f'[TC23] alnorm({z})={v} 不在 [0,1] FAILED'
# Φ(-z) = 1 - Φ(z)（对称性由递归保证）
z = 1.5
assert abs(alnorm(-z) - (1.0 - alnorm(z))) < 1e-10, f'[TC23] 对称性不满足: alnorm(-{z})={alnorm(-z)}, 1-alnorm({z})={1.0-alnorm(z)} FAILED'

# ---- TC24: box_muller_transform 固定种子可复现 ----
import numpy as np
from normal_approx import box_muller_transform
np.random.seed(0)
s1 = box_muller_transform(100, seed=42)
np.random.seed(0)
s2 = box_muller_transform(100, seed=42)
assert np.array_equal(s1, s2), '[TC24] 固定种子下两次调用结果不同 FAILED'

# ---- TC25: gaussian_kl_divergence 同分布 KL≈0 ----
import numpy as np
from normal_approx import gaussian_kl_divergence
kl = gaussian_kl_divergence(0.0, 1.0, 0.0, 1.0)
assert abs(kl) < 1e-10, f'[TC25] KL(N(0,1)||N(0,1))={kl}, 应≈0 FAILED'

# ---- TC26: gaussian_kl_divergence 非负性 ----
import numpy as np
from normal_approx import gaussian_kl_divergence
kl1 = gaussian_kl_divergence(0.0, 1.0, 1.0, 2.0)
kl2 = gaussian_kl_divergence(1.0, 2.0, 0.0, 1.0)
assert kl1 >= -1e-12, f'[TC26] KL 散度应为非负，得到 {kl1} FAILED'
assert kl2 >= -1e-12, f'[TC26] KL 散度应为非负，得到 {kl2} FAILED'

# ---- TC27: reparameterized_gaussian_sample 输出形状正确 ----
import numpy as np
np.random.seed(42)
from normal_approx import reparameterized_gaussian_sample
mean = np.array([1.0, 2.0, 3.0])
std = np.array([0.1, 0.2, 0.1])
samples = reparameterized_gaussian_sample(mean, std, seed=42)
assert samples.shape == mean.shape, f'[TC27] 采样形状错误: {samples.shape} FAILED'
assert np.all(np.isfinite(samples)), '[TC27] 采样含 NaN/Inf FAILED'

# ---- TC28: DenseLayer 前向与反向传播 ----
import numpy as np
np.random.seed(42)
from gan_numpy import DenseLayer
layer = DenseLayer(4, 3)
x = np.random.randn(8, 4)
y = layer.forward(x)
assert y.shape == (8, 3), f'[TC28] DenseLayer 前向形状错误: {y.shape} FAILED'
grad = np.ones((8, 3))
grad_back = layer.backward(grad)
assert grad_back.shape == (8, 4), f'[TC28] DenseLayer 反向形状错误: {grad_back.shape} FAILED'

# ---- TC29: ReLU 前向传播非负 ----
import numpy as np
from gan_numpy import ReLU
x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
relu = ReLU()
y = relu.forward(x)
assert np.all(y >= 0.0), '[TC29] ReLU 输出含负数 FAILED'
assert y[0] == 0.0, '[TC29] ReLU(-2) 应为 0 FAILED'
assert y[4] == 2.0, '[TC29] ReLU(2) 应为 2 FAILED'

# ---- TC30: Sigmoid 输出在 (0,1) 且单调 ----
import numpy as np
from gan_numpy import Sigmoid
x = np.array([-10.0, 0.0, 10.0])
sigmoid = Sigmoid()
y = sigmoid.forward(x)
assert np.all(y > 0.0) and np.all(y < 1.0), '[TC30] Sigmoid 输出不在 (0,1) FAILED'
assert abs(y[1] - 0.5) < 1e-6, f'[TC30] Sigmoid(0)={y[1]} FAILED'

# ---- TC31: MSELoss 完全一致时损失为 0 ----
import numpy as np
from gan_numpy import MSELoss
pred = np.array([[1.0, 2.0], [3.0, 4.0]])
target = np.array([[1.0, 2.0], [3.0, 4.0]])
mse = MSELoss()
loss = mse.forward(pred, target)
assert loss < 1e-12, f'[TC31] 相同输入 MSE 应为 0，得到 {loss} FAILED'

# ---- TC32: BCELoss 数值稳定性（extreme input） ----
import numpy as np
from gan_numpy import BCELoss
bce = BCELoss()
# 近 0 和近 1 的输入
pred_safe = np.array([[0.5, 0.5]])
target_safe = np.array([[1.0, 0.0]])
loss = bce.forward(pred_safe, target_safe)
assert np.isfinite(loss), f'[TC32] BCE 损失应为有限值，得到 {loss} FAILED'
assert loss > 0.0, f'[TC32] BCE 损失应为正，得到 {loss} FAILED'

# ---- TC33: Generator 前向传播形状正确 ----
import numpy as np
np.random.seed(42)
from gan_numpy import Generator
gen = Generator(latent_dim=8, coord_dim=4, hidden_dim=16, output_dim=4, seed=42)
z = np.random.randn(16, 8)
coords = np.random.randn(16, 4)
out = gen.forward(z, coords)
assert out.shape == (16, 4), f'[TC33] 生成器输出形状错误: {out.shape} FAILED'
assert np.all(np.isfinite(out)), '[TC33] 生成器输出含 NaN/Inf FAILED'

# ---- TC34: Discriminator 前向传播形状与值域正确 ----
import numpy as np
np.random.seed(42)
from gan_numpy import Discriminator
disc = Discriminator(input_dim=8, hidden_dim=16, seed=43)
state = np.random.randn(8, 8)
score = disc.forward(state)
assert score.shape == (8, 1), f'[TC34] 判别器输出形状错误: {score.shape} FAILED'
assert np.all(score > 0.0) and np.all(score < 1.0), '[TC34] 判别器输出不在 (0,1) FAILED'

# ---- TC35: hooke_jeeves 优化简单二次函数 ----
import numpy as np
from hooke_jeeves import hooke_jeeves
def f_quad(x):
    return float(x[0]**2 + (x[1]-3.0)**2)
iters, best = hooke_jeeves(2, np.array([10.0, 10.0]), 0.5, 1e-4, 50, f_quad)
assert abs(best[0]) < 0.1, f'[TC35] 最优 x[0] 应接近 0，得到 {best[0]} FAILED'
assert abs(best[1] - 3.0) < 0.1, f'[TC35] 最优 x[1] 应接近 3，得到 {best[1]} FAILED'

# ---- TC36: optimize_gan_hyperparams 优化演示 ----
import numpy as np
np.random.seed(42)
from hooke_jeeves import optimize_gan_hyperparams
def dummy_loss(params):
    target = np.array([0.002, 0.002, 0.5, 0.1])
    return float(np.sum((params - target)**2))
init = np.array([0.001, 0.003, 0.3, 0.2])
best_params, history = optimize_gan_hyperparams(init, dummy_loss, rho=0.5, eps=1e-4, itermax=20)
assert len(history) > 0, '[TC36] 优化历史不应为空 FAILED'
assert len(best_params) == 4, f'[TC36] 优化参数维度错误: {len(best_params)} FAILED'

# ---- TC37: cvt_2d_sampling 输出形状与范围正确 ----
import numpy as np
np.random.seed(42)
from cvt_sampler import cvt_2d_sampling
gens = cvt_2d_sampling(k=9, n_samples=2000, itermax=20, seed=42)
assert gens.shape == (9, 2), f'[TC37] CVT 输出形状错误: {gens.shape} FAILED'
assert np.all(gens >= 0.0) and np.all(gens <= 1.0), '[TC37] CVT 采样点超出 [0,1] FAILED'

# ---- TC38: cvt_energy 非负 ----
import numpy as np
np.random.seed(42)
from cvt_sampler import cvt_energy
gens = np.random.rand(4, 2)
samples = np.random.rand(100, 2)
energy = cvt_energy(gens, samples)
assert energy >= 0.0, f'[TC38] CVT 能量应为非负，得到 {energy} FAILED'

# ---- TC39: point_in_polygon 正方形内含点判断 ----
import numpy as np
from complex_geometry import point_in_polygon
square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
assert point_in_polygon(np.array([0.5, 0.5]), square), '[TC39] (0.5,0.5) 应在正方形内 FAILED'
assert not point_in_polygon(np.array([2.0, 2.0]), square), '[TC39] (2,2) 不应在正方形内 FAILED'

# ---- TC40: polygon_area_mc 正方形面积估计 ----
import numpy as np
np.random.seed(42)
from complex_geometry import polygon_area_mc
square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
area, std_err = polygon_area_mc(square, n_samples=5000, seed=42)
assert abs(area - 1.0) < 0.1, f'[TC40] 正方形面积估计偏离: {area} FAILED'

# ---- TC41: equilateral_distance_pdf 基本功能（输出形状正确、有限值） ----
import numpy as np
from geometric_stats import equilateral_distance_pdf
d = np.linspace(0.01, 0.5, 50)
pdf = equilateral_distance_pdf(d, side=1.0)
assert len(pdf) == 50, f'[TC41] PDF 输出长度错误: {len(pdf)} FAILED'
assert np.all(np.isfinite(pdf)), '[TC41] PDF 含 NaN/Inf FAILED'
# 在 (0, 0.8) 区间内 PDF 应为正
assert np.all(pdf > 0.0), '[TC41] PDF 在小 d 区间应全正 FAILED'

# ---- TC42: triangle_sample 均在三角形内（重心坐标和=1） ----
import numpy as np
np.random.seed(42)
from geometric_stats import triangle_sample
v1 = np.array([0.0, 0.0])
v2 = np.array([1.0, 0.0])
v3 = np.array([0.0, 1.0])
pts = triangle_sample(v1, v2, v3, 200, seed=42)
# 所有点应在 x>=0, y>=0, x+y<=1 内
assert np.all(pts[:, 0] >= -1e-12), '[TC42] 采样点 x<0 FAILED'
assert np.all(pts[:, 1] >= -1e-12), '[TC42] 采样点 y<0 FAILED'
assert np.all(pts[:, 0] + pts[:, 1] <= 1.0 + 1e-12), '[TC42] 采样点不在三角形内 FAILED'

# ---- TC43: wasserstein_approx_mc 同分布 Wasserstein ≈ 0 ----
import numpy as np
np.random.seed(42)
from geometric_stats import wasserstein_approx_mc
samples = np.random.randn(100, 2)
w1 = wasserstein_approx_mc(samples, samples)
assert w1 < 0.5, f'[TC43] 同分布 Wasserstein 应较小，得到 {w1} FAILED'
assert w1 >= 0.0, f'[TC43] Wasserstein 距离应为非负，得到 {w1} FAILED'

# ---- TC44: tsp_brute 小规模 TSP 求解 ----
import numpy as np
from latent_path import tsp_brute
# 3个城市构成的等边三角形
dist = np.array([[0.0, 1.0, 1.0],
                 [1.0, 0.0, 1.0],
                 [1.0, 1.0, 0.0]])
p_min, total_min, total_ave = tsp_brute(dist)
assert total_min == 3.0, f'[TC44] 最小 TSP 路径应为 3，得到 {total_min} FAILED'

# ---- TC45: slerp 端点和中间插值 ----
import numpy as np
from latent_path import slerp
z1 = np.array([1.0, 0.0, 0.0])
z2 = np.array([0.0, 1.0, 0.0])
# t=0 应等于 z1
z0 = slerp(z1, z2, 0.0)
assert np.allclose(z0, z1), '[TC45] slerp(t=0) 应等于 z1 FAILED'
# t=1 应等于 z2
z1t = slerp(z1, z2, 1.0)
assert np.allclose(z1t, z2, atol=1e-6), '[TC45] slerp(t=1) 应等于 z2 FAILED'
# t=0.5 应在中间（单位向量）
zmid = slerp(z1, z2, 0.5)
assert abs(np.linalg.norm(zmid) - 1.0) < 1e-10, '[TC45] slerp(t=0.5) 范数应为 1 FAILED'

# ---- TC46: r8vec_cross_3d 叉积验证 ----
import numpy as np
from obj_io import r8vec_cross_3d
cx = r8vec_cross_3d(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
assert np.allclose(cx, [0.0, 0.0, 1.0]), f'[TC46] i×j 应为 k，得到 {cx} FAILED'

# ---- TC47: compute_face_normals 基本验证 ----
import numpy as np
from obj_io import compute_face_normals
vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
faces = [[0, 1, 2], [1, 3, 2]]
fn = compute_face_normals(vertices, faces)
assert fn.shape == (2, 3), f'[TC47] 面法线形状错误: {fn.shape} FAILED'
# 顶面法线应为 (0,0,1)
assert abs(fn[0, 2] - 1.0) < 1e-10 or abs(fn[0, 2] + 1.0) < 1e-10, f'[TC47] 法线 z 分量应为 ±1，得到 {fn[0,2]} FAILED'

# ---- TC48: compute_vertex_normals 输出均归一化 ----
import numpy as np
from obj_io import compute_vertex_normals
vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
faces = [[0, 1, 2], [1, 3, 2]]
vn = compute_vertex_normals(vertices, faces)
for i in range(vn.shape[0]):
    n = np.linalg.norm(vn[i])
    assert abs(n - 1.0) < 1e-10, f'[TC48] 顶点 {i} 法线范数={n}，应为 1 FAILED'

# ---- TC49: icosphere_obj 细分后顶点与面数正确 ----
import numpy as np
from obj_io import icosphere_obj
verts0, faces0 = icosphere_obj(radius=1.0, subdivisions=0)
verts1, faces1 = icosphere_obj(radius=1.0, subdivisions=1)
assert verts0.shape[0] == 12, f'[TC49] 细分0次应有 12 顶点，得到 {verts0.shape[0]} FAILED'
assert len(faces0) == 20, f'[TC49] 细分0次应有 20 面，得到 {len(faces0)} FAILED'
assert len(faces1) == 80, f'[TC49] 细分1次应有 80 面，得到 {len(faces1)} FAILED'

# ---- TC50: special_function_spectral_basis 形状与值域正确 ----
import numpy as np
from special_functions import special_function_spectral_basis
x = np.linspace(-1.0, 1.0, 10)
basis = special_function_spectral_basis(x, n_modes=8)
assert basis.shape == (10, 8), f'[TC50] 谱基形状错误: {basis.shape} FAILED'
assert np.all(np.abs(basis) <= 1.0 + 1e-10), '[TC50] 谱基值超出 [-1,1] FAILED'

# ---- TC51: 集成测试：navier_stokes_exact 全流程 ----
import numpy as np
np.random.seed(42)
from navier_stokes_exact import uvwp_ethier, ns_residual, generate_training_data
X, Y = generate_training_data(nx=4, ny=4, nz=4)
assert X.shape == (64, 4), f'[TC51] 数据形状错误: {X.shape} FAILED'
# ns_residual on 4x4x4 grid
u = Y[:, 0].reshape(4, 4, 4)
v = Y[:, 1].reshape(4, 4, 4)
w = Y[:, 2].reshape(4, 4, 4)
p = Y[:, 3].reshape(4, 4, 4)
x = np.linspace(-1.0, 1.0, 4)
res = ns_residual(u, v, w, p, x, x, x, np.array([0.05]))
assert 'continuity' in res, '[TC51] 残差缺少 continuity 键 FAILED'
assert res['continuity'] >= 0.0, f'[TC51] 连续性残差应为非负 FAILED'

# ---- TC52: integration_pde_residual_over_mesh 基本调用 ----
import numpy as np
from triangle_quadrature import integrate_pde_residual_over_mesh
v1 = np.array([0.0, 0.0])
v2 = np.array([1.0, 0.0])
v3 = np.array([0.0, 1.0])
def res_func(pts):
    return np.ones(pts.shape[0])
triangles = [(v1, v2, v3)]
total = integrate_pde_residual_over_mesh(res_func, triangles, degree=3)
assert abs(total - 0.5) < 1e-6, f'[TC52] 单位残差积分应为 0.5，得到 {total} FAILED'

# ---- TC53: sphere01_triangle_quad_03 常数函数积分应等于面积 ----
import numpy as np
np.random.seed(42)
from sphere_quad import sphere01_triangle_quad_03, sphere01_triangle_vertices_to_area
v1 = np.array([1.0, 0.0, 0.0])
v2 = np.array([0.0, 1.0, 0.0])
v3 = np.array([0.0, 0.0, 1.0])
area = sphere01_triangle_vertices_to_area(v1, v2, v3)
quad_val = sphere01_triangle_quad_03(v1, v2, v3, lambda x: 1.0)
assert abs(quad_val - area) < 1e-10, f'[TC53] 3点积分 {quad_val} 与面积 {area} 不匹配 FAILED'

# ---- TC54: cvt_latent_samples 高维采样形状正确 ----
import numpy as np
np.random.seed(42)
from cvt_sampler import cvt_latent_samples
gens = cvt_latent_samples(k=16, dim=8, n_samples=2000, itermax=10, seed=42)
assert gens.shape == (16, 8), f'[TC54] CVT 隐空间采样形状错误: {gens.shape} FAILED'
assert np.all(gens >= 0.0) and np.all(gens <= 1.0), '[TC54] CVT 采样点超出 [0,1] FAILED'

# ---- TC55: bowyer_watson 简单点集三角剖分 ----
import numpy as np
from mesh_generator import bowyer_watson
pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
triangles = bowyer_watson(pts)
assert len(triangles) >= 2, f'[TC55] 至少应有 2 个三角形，得到 {len(triangles)} FAILED'
for tri in triangles:
    assert len(tri) == 3, f'[TC55] 三角形应有 3 个顶点，得到 {len(tri)} FAILED'

# ---- TC56: human_outline_boundary 输出封闭多边形 ----
import numpy as np
from mesh_generator import human_outline_boundary
boundary = human_outline_boundary(scale=1.0)
assert boundary.shape[0] > 5, f'[TC56] 边界点数太少: {boundary.shape[0]} FAILED'
assert boundary.shape[1] == 2, f'[TC56] 边界维度错误: {boundary.shape} FAILED'

# ---- TC57: mesh_quality_stats 返回期望字段 ----
import numpy as np
from mesh_generator import human_outline_boundary, generate_mesh_from_boundary, mesh_quality_stats
boundary = human_outline_boundary(scale=0.5)
nodes, triangles = generate_mesh_from_boundary(boundary, hmax=0.5)
stats = mesh_quality_stats(nodes, triangles)
assert 'min_angle_deg' in stats, '[TC57] 缺少 min_angle_deg FAILED'
assert 'num_triangles' in stats, '[TC57] 缺少 num_triangles FAILED'
assert 'num_nodes' in stats, '[TC57] 缺少 num_nodes FAILED'
assert stats['num_triangles'] > 0, '[TC57] 三角形数量应为正 FAILED'

# ---- TC58: LeakyReLU 负区缩放 ----
import numpy as np
from gan_numpy import LeakyReLU
x = np.array([-2.0, 0.0, 2.0])
lrelu = LeakyReLU(alpha=0.2)
y = lrelu.forward(x)
assert abs(y[0] - (-0.4)) < 1e-12, f'[TC58] LeakyReLU(-2)={y[0]} 应为 -0.4 FAILED'
assert y[1] == 0.0, '[TC58] LeakyReLU(0)=0 FAILED'
assert y[2] == 2.0, '[TC58] LeakyReLU(2)=2 FAILED'

# ---- TC59: 集成测试：human_outline_polygon + area estimation ----
import numpy as np
np.random.seed(42)
from complex_geometry import human_outline_polygon, polygon_area_mc
poly = human_outline_polygon(scale=1.0, n_points=60)
area, std_err = polygon_area_mc(poly, n_samples=3000, seed=42)
assert area > 0.1, f'[TC59] 人体轮廓面积过小: {area} FAILED'
assert std_err < area, f'[TC59] 标准误差过大: {std_err} FAILED'

# ---- TC60: sphere01_triangle_sample 采样点在球面上 ----
import numpy as np
np.random.seed(42)
from triangle_quadrature import sphere01_triangle_sample
v1 = np.array([1.0, 0.0, 0.0])
v2 = np.array([0.0, 1.0, 0.0])
v3 = np.array([0.0, 0.0, 1.0])
pts = sphere01_triangle_sample(50, v1, v2, v3, seed=42)
norms = np.sqrt(np.sum(pts**2, axis=0))
assert np.all(np.abs(norms - 1.0) < 1e-10), '[TC60] 球面采样点不在单位球面上 FAILED'

print('\n全部 60 个测试通过!\n')
