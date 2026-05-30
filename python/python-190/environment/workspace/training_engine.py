
import numpy as np
from gan_numpy import Generator, Discriminator, BCELoss, MSELoss
from navier_stokes_exact import generate_training_data, uvwp_ethier
from normal_approx import box_muller_transform
from cvt_sampler import cvt_2d_sampling


def prepare_training_data(nx: int = 6, ny: int = 6, nz: int = 6,
                          a: float = np.pi / 4.0, d: float = np.pi / 2.0,
                          t_val: float = 0.05) -> tuple:
    X, Y = generate_training_data(nx, ny, nz, a, d, t_val)
    coords = X
    states = np.concatenate([X, Y], axis=1)
    return coords, states


def train_pigan(epochs: int = 120, batch_size: int = 32,
                lr_g: float = 0.002, lr_d: float = 0.002,
                lambda_phys: float = 0.5, lambda_equiv: float = 0.1,
                nx: int = 6, ny: int = 6, nz: int = 6,
                latent_dim: int = 8, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    np.random.seed(seed)


    coords, states = prepare_training_data(nx, ny, nz)
    N = coords.shape[0]
    n_batches = max(1, N // batch_size)


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


            z = box_muller_transform(current_bs * latent_dim, rng.integers(0, 2**31))
            z = z.reshape((current_bs, latent_dim))


            disc.zero_grad()

            real_score = disc.forward(real_states)
            loss_d_real = bce.forward(real_score, np.ones_like(real_score))
            grad_d_real = bce.backward()
            disc.backward(grad_d_real)


            fake_out = gen.forward(z, real_coords)
            fake_states = np.concatenate([real_coords, fake_out], axis=1)
            fake_score = disc.forward(fake_states)
            loss_d_fake = bce.forward(fake_score, np.zeros_like(fake_score))
            grad_d_fake = bce.backward()
            disc.backward(grad_d_fake)

            loss_d = loss_d_real + loss_d_fake
            disc.step(lr_d)


            gen.zero_grad()
            disc.zero_grad()

            z_g = box_muller_transform(current_bs * latent_dim, rng.integers(0, 2**31))
            z_g = z_g.reshape((current_bs, latent_dim))
            fake_out_g = gen.forward(z_g, real_coords)
            fake_states_g = np.concatenate([real_coords, fake_out_g], axis=1)
            fake_score_g = disc.forward(fake_states_g)


            loss_g_adv = bce.forward(fake_score_g, np.ones_like(fake_score_g))
            grad_g_adv = bce.backward()
            grad_from_disc = disc.backward(grad_g_adv)

            grad_g_out = grad_from_disc[:, 4:]
            gen.backward(grad_g_out)


            if epoch % 5 == 0 and b == 0 and N == nx * ny * nz:
                z_phys = box_muller_transform(latent_dim, rng.integers(0, 2**31))
                z_phys = z_phys.reshape((1, latent_dim))
                z_phys_batch = np.tile(z_phys, (N, 1))
                fake_phys = gen.forward(z_phys_batch, coords)












                from navier_stokes_exact import ns_residual
                loss_phys = 0.0
                grad_phys = np.zeros_like(fake_phys)
                gen.backward(lambda_phys * grad_phys)
                epoch_phys += loss_phys

            else:
                loss_phys = 0.0


            loss_equiv = 0.0
            if epoch % 10 == 0 and b == 0:

                z1 = box_muller_transform(latent_dim, rng.integers(0, 2**31)).reshape((1, latent_dim))
                z2 = box_muller_transform(latent_dim, rng.integers(0, 2**31)).reshape((1, latent_dim))
                z1b = np.tile(z1, (N, 1))
                z2b = np.tile(z2, (N, 1))
                f1 = gen.forward(z1b, coords)
                f2 = gen.forward(z2b, coords)
                loss_equiv = float(np.mean((f1 - f2) ** 2))


            gen.step(lr_g)

            epoch_loss_d += loss_d
            epoch_loss_g += loss_g_adv + lambda_phys * loss_phys
            epoch_equiv += loss_equiv

        history["loss_d"].append(epoch_loss_d / n_batches)
        history["loss_g"].append(epoch_loss_g / n_batches)
        history["phys_loss"].append(epoch_phys)
        history["equiv_loss"].append(epoch_equiv)


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
    coords = results["coords"]
    states = results["states"]
    gen = results["generator"]
    N = coords.shape[0]


    from geometric_stats import wasserstein_approx_mc
    z_test = box_muller_transform(8, 12345).reshape((1, 8))
    z_test_batch = np.tile(z_test, (N, 1))
    fake_full = gen.forward(z_test_batch, coords)
    real_proj = states[:, 4:6]
    fake_proj = fake_full[:, 0:2]
    w1 = wasserstein_approx_mc(real_proj, fake_proj)


    from sphere_quad import integrate_on_sphere
    def speed_norm(omega):


        x, y, z = omega[0], omega[1], omega[2]
        coord = np.array([[x, y, z, 0.05]])
        z_vec = box_muller_transform(8, 42).reshape((1, 8))
        pred = gen.forward(z_vec, coord)
        vel = pred[0, 0:3]
        return float(np.linalg.norm(vel))
    sphere_integral = integrate_on_sphere(speed_norm, rule="icos1v")


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
