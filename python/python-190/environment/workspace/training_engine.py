"""
training_engine.py
==================
物理信息 GAN 的训练引擎。
整合对抗训练、物理损失、旋转等变损失、几何统计评估以及
Hooke-Jeeves 超参数优化，实现完整的博士级训练流程。

训练流程：
  1. 数据准备：
     · 使用 Ethier 精确解生成真实流场数据（结构化网格 + 复杂域采样）。
     · 使用 CVT 采样优化评估点分布。
     · 使用 Box-Muller 变换生成隐空间先验样本。

  2. 对抗训练（每轮迭代）：
     · 更新判别器 D：最大化 log D(real) + log(1 - D(fake))。
     · 更新生成器 G：最小化 -log D(fake) + λ_phys·L_phys + λ_equiv·L_equiv。

  3. 物理损失评估：
     · 在每 k 轮对完整网格计算 NS 残差。
     · 使用三角形对称求积在二维截面网格上积分残差。

  4. 等变性验证：
     · 随机采样旋转轴与角度，用四元数旋转场后比较一致性。

  5. 评估指标：
     · Wasserstein 近似距离（基于 mesh distance stats）。
     · 球面求积验证三维能谱守恒。
     · 三角形直方图均匀性测试评估生成覆盖度。

  6. 超参数自适应：
     · 每 50 轮使用 Hooke-Jeeves 微调 λ_phys 与 λ_equiv。
"""

import numpy as np
from gan_numpy import Generator, Discriminator, BCELoss, MSELoss
from navier_stokes_exact import generate_training_data, uvwp_ethier
from normal_approx import box_muller_transform
from cvt_sampler import cvt_2d_sampling


def prepare_training_data(nx: int = 6, ny: int = 6, nz: int = 6,
                          a: float = np.pi / 4.0, d: float = np.pi / 2.0,
                          t_val: float = 0.05) -> tuple:
    """
    生成真实训练数据。

    Returns
    -------
    coords : np.ndarray, shape (N, 4)
        [x, y, z, t]
    states : np.ndarray, shape (N, 8)
        [x, y, z, t, u, v, w, p]
    """
    X, Y = generate_training_data(nx, ny, nz, a, d, t_val)
    coords = X  # (N, 4)
    states = np.concatenate([X, Y], axis=1)  # (N, 8)
    return coords, states


def train_pigan(epochs: int = 120, batch_size: int = 32,
                lr_g: float = 0.002, lr_d: float = 0.002,
                lambda_phys: float = 0.5, lambda_equiv: float = 0.1,
                nx: int = 6, ny: int = 6, nz: int = 6,
                latent_dim: int = 8, seed: int = 42) -> dict:
    """
    训练物理信息 GAN。

    Parameters
    ----------
    epochs : int
        训练轮数。
    batch_size : int
        每批采样点数。
    lr_g, lr_d : float
        生成器与判别器学习率。
    lambda_phys, lambda_equiv : float
        物理损失与等变损失权重。
    nx, ny, nz : int
        空间网格维度。
    latent_dim : int
        隐空间维度。
    seed : int
        随机种子。

    Returns
    -------
    results : dict
        训练历史与评估指标。
    """
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    # 数据准备
    coords, states = prepare_training_data(nx, ny, nz)
    N = coords.shape[0]
    n_batches = max(1, N // batch_size)

    # 模型初始化
    gen = Generator(latent_dim=latent_dim, coord_dim=4, hidden_dim=32,
                    output_dim=4, seed=seed)
    disc = Discriminator(input_dim=8, hidden_dim=32, seed=seed + 1)

    bce = BCELoss()

    history = {
        "loss_d": [],
        "loss_g": [],
        "phys_loss": [],
        "equiv_loss": [],
    }

    for epoch in range(epochs):
        # 随机打乱数据
        perm = rng.permutation(N)
        coords_shuffled = coords[perm]
        states_shuffled = states[perm]

        epoch_loss_d = 0.0
        epoch_loss_g = 0.0
        epoch_phys = 0.0
        epoch_equiv = 0.0

        for b in range(n_batches):
            start = b * batch_size
            end = min(start + batch_size, N)
            real_coords = coords_shuffled[start:end]
            real_states = states_shuffled[start:end]
            current_bs = real_coords.shape[0]

            # 隐向量采样
            z = box_muller_transform(current_bs * latent_dim, rng.integers(0, 2**31))
            z = z.reshape((current_bs, latent_dim))

            # ---- 判别器更新 ----
            disc.zero_grad()
            # 真实样本
            real_score = disc.forward(real_states)
            loss_d_real = bce.forward(real_score, np.ones_like(real_score))
            grad_d_real = bce.backward()
            disc.backward(grad_d_real)

            # 生成样本
            fake_out = gen.forward(z, real_coords)
            fake_states = np.concatenate([real_coords, fake_out], axis=1)
            fake_score = disc.forward(fake_states)
            loss_d_fake = bce.forward(fake_score, np.zeros_like(fake_score))
            grad_d_fake = bce.backward()
            disc.backward(grad_d_fake)

            loss_d = loss_d_real + loss_d_fake
            disc.step(lr_d)

            # ---- 生成器更新 ----
            gen.zero_grad()
            disc.zero_grad()

            z_g = box_muller_transform(current_bs * latent_dim, rng.integers(0, 2**31))
            z_g = z_g.reshape((current_bs, latent_dim))
            fake_out_g = gen.forward(z_g, real_coords)
            fake_states_g = np.concatenate([real_coords, fake_out_g], axis=1)
            fake_score_g = disc.forward(fake_states_g)

            # 对抗损失（非饱和）
            loss_g_adv = bce.forward(fake_score_g, np.ones_like(fake_score_g))
            grad_g_adv = bce.backward()
            grad_from_disc = disc.backward(grad_g_adv)
            # 只保留后 4 维（u,v,w,p）的梯度传给生成器
            grad_g_out = grad_from_disc[:, 4:]
            gen.backward(grad_g_out)

            # 物理损失（每 5 轮对完整网格评估一次，降低计算开销）
            if epoch % 5 == 0 and b == 0 and N == nx * ny * nz:
                z_phys = box_muller_transform(latent_dim, rng.integers(0, 2**31))
                z_phys = z_phys.reshape((1, latent_dim))
                z_phys_batch = np.tile(z_phys, (N, 1))
                fake_phys = gen.forward(z_phys_batch, coords)
                # TODO_HOLE_2_START: 使用 ns_residual 计算生成场的物理残差损失
                # 提示：
                #   1. fake_phys 形状为 (N, 4)，列顺序为 [u, v, w, p]
                #   2. coords 形状为 (N, 4)，列顺序为 [x, y, z, t]
                #   3. N = nx * ny * nz，对应一个完整的三维结构化网格
                #   4. 需要从 navier_stokes_exact 导入 ns_residual
                #   5. ns_residual 接受展平后的 u, v, w, p, x, y, z, t 数组
                #      内部会自动 reshape 为 (nx, ny, nz) 并使用中心差分
                #   6. 返回字典中的 'total' 键即为总残差值
                #   7. 需要将残差转化为可反向传播的梯度形式回传给生成器
                #      （提示：可通过生成器对 coords 的数值导数近似，
                #       或使用 fake_phys 与真实场之间的差异作为代理梯度）
                from navier_stokes_exact import ns_residual
                loss_phys = 0.0  # placeholder
                grad_phys = np.zeros_like(fake_phys)
                gen.backward(lambda_phys * grad_phys)
                epoch_phys += loss_phys
                # TODO_HOLE_2_END
            else:
                loss_phys = 0.0

            # 等变损失（每 10 轮评估一次）
            loss_equiv = 0.0
            if epoch % 10 == 0 and b == 0:
                # 简化的等变损失：比较两次随机生成场的差异上界
                z1 = box_muller_transform(latent_dim, rng.integers(0, 2**31)).reshape((1, latent_dim))
                z2 = box_muller_transform(latent_dim, rng.integers(0, 2**31)).reshape((1, latent_dim))
                z1b = np.tile(z1, (N, 1))
                z2b = np.tile(z2, (N, 1))
                f1 = gen.forward(z1b, coords)
                f2 = gen.forward(z2b, coords)
                loss_equiv = float(np.mean((f1 - f2) ** 2))
                # 不对等变损失反向传播（仅作为监控指标），避免梯度爆炸

            gen.step(lr_g)

            epoch_loss_d += loss_d
            epoch_loss_g += loss_g_adv + lambda_phys * loss_phys
            epoch_equiv += loss_equiv

        history["loss_d"].append(epoch_loss_d / n_batches)
        history["loss_g"].append(epoch_loss_g / n_batches)
        history["phys_loss"].append(epoch_phys)
        history["equiv_loss"].append(epoch_equiv)

    # 最终评估
    final_z = box_muller_transform(latent_dim, seed)
    final_z = final_z.reshape((1, latent_dim))
    final_z_batch = np.tile(final_z, (N, 1))
    final_pred = gen.forward(final_z_batch, coords)
    final_mse = float(np.mean((final_pred - states[:, 4:]) ** 2))

    results = {
        "history": history,
        "final_mse": final_mse,
        "generator": gen,
        "discriminator": disc,
        "coords": coords,
        "states": states,
    }
    return results


def evaluate_with_geometry(results: dict, nx: int = 6, ny: int = 6) -> dict:
    """
    使用几何统计与球面积分对训练结果进行综合评估。

    Returns
    -------
    metrics : dict
        评估指标字典。
    """
    coords = results["coords"]
    states = results["states"]
    gen = results["generator"]
    N = coords.shape[0]

    # 1. Wasserstein 近似距离（基于生成场与真实场在二维投影上的距离统计）
    from geometric_stats import wasserstein_approx_mc
    z_test = box_muller_transform(8, 12345).reshape((1, 8))
    z_test_batch = np.tile(z_test, (N, 1))
    fake_full = gen.forward(z_test_batch, coords)
    real_proj = states[:, 4:6]  # u,v 投影
    fake_proj = fake_full[:, 0:2]
    w1 = wasserstein_approx_mc(real_proj, fake_proj)

    # 2. 球面积分验证（生成场的速度模在球面上的积分）
    from sphere_quad import integrate_on_sphere
    def speed_norm(omega):
        # 将单位向量映射到坐标，通过生成器计算速度模
        # 简化：假设球面方向映射到单位球坐标
        x, y, z = omega[0], omega[1], omega[2]
        coord = np.array([[x, y, z, 0.05]])
        z_vec = box_muller_transform(8, 42).reshape((1, 8))
        pred = gen.forward(z_vec, coord)
        vel = pred[0, 0:3]
        return float(np.linalg.norm(vel))
    sphere_integral = integrate_on_sphere(speed_norm, rule="icos1v")

    # 3. 三角形网格质量评估（使用训练域边界）
    from mesh_generator import human_outline_boundary, generate_mesh_from_boundary, mesh_quality_stats
    boundary = human_outline_boundary(scale=0.5)
    nodes, triangles = generate_mesh_from_boundary(boundary, hmax=0.3)
    mesh_stats = mesh_quality_stats(nodes, triangles)

    metrics = {
        "wasserstein_approx": w1,
        "sphere_speed_integral": sphere_integral,
        "mesh_min_angle_deg": mesh_stats["min_angle_deg"],
        "mesh_num_triangles": mesh_stats["num_triangles"],
        "mesh_num_nodes": mesh_stats["num_nodes"],
    }
    return metrics
