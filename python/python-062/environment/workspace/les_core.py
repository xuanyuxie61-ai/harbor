
import numpy as np


def divergence(u, v, w, dx, dy, dz):
    nx, ny, nz = u.shape
    div = np.zeros_like(u)


    div[1:-1, 1:-1, 1:-1] = (
        (u[2:, 1:-1, 1:-1] - u[:-2, 1:-1, 1:-1]) / (2 * dx) +
        (v[1:-1, 2:, 1:-1] - v[1:-1, :-2, 1:-1]) / (2 * dy) +
        (w[1:-1, 1:-1, 2:] - w[1:-1, 1:-1, :-2]) / (2 * dz)
    )

    return div


def laplacian_3d(phi, dx, dy, dz):
    lap = np.zeros_like(phi)

    lap[1:-1, 1:-1, 1:-1] = (
        (phi[2:, 1:-1, 1:-1] - 2 * phi[1:-1, 1:-1, 1:-1] + phi[:-2, 1:-1, 1:-1]) / dx**2 +
        (phi[1:-1, 2:, 1:-1] - 2 * phi[1:-1, 1:-1, 1:-1] + phi[1:-1, :-2, 1:-1]) / dy**2 +
        (phi[1:-1, 1:-1, 2:] - 2 * phi[1:-1, 1:-1, 1:-1] + phi[1:-1, 1:-1, :-2]) / dz**2
    )

    return lap


def convection_term(u, v, w, dx, dy, dz):
    def grad_central(f, axis, h):
        df = np.zeros_like(f)
        slc_p = [slice(None)] * 3
        slc_m = [slice(None)] * 3
        slc_p[axis] = slice(2, None)
        slc_m[axis] = slice(None, -2)
        df[tuple(slc_p)] = (f[tuple(slc_p)] - f[tuple(slc_m)]) / (2 * h)
        return df

    dudx = grad_central(u, 0, dx)
    dudy = grad_central(u, 1, dy)
    dudz = grad_central(u, 2, dz)

    dvdx = grad_central(v, 0, dx)
    dvdy = grad_central(v, 1, dy)
    dvdz = grad_central(v, 2, dz)

    dwdx = grad_central(w, 0, dx)
    dwdy = grad_central(w, 1, dy)
    dwdz = grad_central(w, 2, dz)


    conv_u = u * dudx + v * dudy + w * dudz
    conv_v = u * dvdx + v * dvdy + w * dvdz
    conv_w = u * dwdx + v * dwdy + w * dwdz

    return conv_u, conv_v, conv_w


def solve_poisson_fft(rhs, dx, dy, dz):
    nx, ny, nz = rhs.shape
    rhs_hat = np.fft.fftn(rhs)

    kx = 2.0 * np.pi * np.fft.fftfreq(nx, dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, dy)
    kz = 2.0 * np.pi * np.fft.fftfreq(nz, dz)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    k2 = KX**2 + KY**2 + KZ**2
    k2[0, 0, 0] = 1.0

    p_hat = rhs_hat / k2
    p = np.fft.ifftn(p_hat).real
    return p


def projection_step(u_star, v_star, w_star, dx, dy, dz, dt, rho=1.0,
                    max_iter=100, tol=1e-8):









    raise NotImplementedError("HOLE 3: 请实现投影法速度修正步")



def initialize_turbulent_field(nx, ny, nz, dx, dy, dz, u_mean=5.0, v_mean=0.0,
                                turbulence_intensity=0.1, theta_mean=300.0,
                                theta_gradient=0.003):
    np.random.seed(42)

    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    z = np.arange(nz) * dz

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')


    z0 = 0.1
    kappa = 0.4
    u_star = u_mean * kappa / np.log((nz * dz) / z0)


    z_safe = np.maximum(Z, z0 * 1.1)
    u_profile = (u_star / kappa) * np.log(z_safe / z0)


    u_profile = u_profile * (u_mean / np.mean(u_profile))


    u = u_profile + turbulence_intensity * u_mean * np.random.randn(nx, ny, nz)
    v = v_mean + turbulence_intensity * u_mean * 0.5 * np.random.randn(nx, ny, nz)
    w = turbulence_intensity * u_mean * 0.3 * np.random.randn(nx, ny, nz)


    theta = theta_mean + theta_gradient * Z

    return u, v, w, theta
