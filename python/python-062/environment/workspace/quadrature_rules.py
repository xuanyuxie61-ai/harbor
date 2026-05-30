
import numpy as np



_PATTERSON_TABLE = {
    1: {
        'x': np.array([0.0]),
        'w': np.array([2.0])
    },
    3: {
        'x': np.array([-0.7745966692414834, 0.0, 0.7745966692414834]),
        'w': np.array([0.5555555555555556, 0.8888888888888889, 0.5555555555555556])
    },
    7: {
        'x': np.array([-0.9604912687080203, -0.7745966692414834, -0.43424374934680256,
                        0.0, 0.43424374934680256, 0.7745966692414834, 0.9604912687080203]),
        'w': np.array([0.10465622602646727, 0.26848808986833344, 0.40139741477596224,
                       0.45091653865847414, 0.40139741477596224, 0.26848808986833344,
                       0.10465622602646727])
    },
    15: {
        'x': np.array([-0.993831963212755, -0.9604912687080203, -0.888459232872257,
                       -0.7745966692414834, -0.6211029467372264, -0.43424374934680256,
                       -0.22338668642896688, 0.0, 0.22338668642896688,
                       0.43424374934680256, 0.6211029467372264, 0.7745966692414834,
                       0.888459232872257, 0.9604912687080203, 0.993831963212755]),
        'w': np.array([0.01700171962994026, 0.05160328299707974, 0.09292719531512454,
                       0.13441525524378423, 0.17151190913639138, 0.20062852937698903,
                       0.2191568584015875, 0.2255104997982067, 0.2191568584015875,
                       0.20062852937698903, 0.17151190913639138, 0.13441525524378423,
                       0.09292719531512454, 0.05160328299707974, 0.01700171962994026])
    },
    31: {
        'x': np.array([
            -0.9990981249676676, -0.993831963212755, -0.9815311495537401,
            -0.9604912687080203, -0.9296548574297401, -0.888459232872257,
            -0.8367259381688687, -0.7745966692414834, -0.7024962064915271,
            -0.6211029467372264, -0.5313197436443756, -0.43424374934680256,
            -0.33113539325797684, -0.22338668642896688, -0.11248894313318663,
            0.0, 0.11248894313318663, 0.22338668642896688, 0.33113539325797684,
            0.43424374934680256, 0.5313197436443756, 0.6211029467372264,
            0.7024962064915271, 0.7745966692414834, 0.8367259381688687,
            0.888459232872257, 0.9296548574297401, 0.9604912687080203,
            0.9815311495537401, 0.993831963212755, 0.9990981249676676
        ]),
        'w': np.array([
            0.0025447807915618745, 0.008434565739321106, 0.01644604985438781,
            0.025807598096176654, 0.03595710330712932, 0.046462893261757986,
            0.05697950949412336, 0.0672077542959907, 0.07687962049900353,
            0.08575592004999035, 0.09362710998126447, 0.10031427861179558,
            0.10566989358023481, 0.10957842105592464, 0.11195687302095346,
            0.11275525672076869, 0.11195687302095346, 0.10957842105592464,
            0.10566989358023481, 0.10031427861179558, 0.09362710998126447,
            0.08575592004999035, 0.07687962049900353, 0.0672077542959907,
            0.05697950949412336, 0.046462893261757986, 0.03595710330712932,
            0.025807598096176654, 0.01644604985438781, 0.008434565739321106,
            0.0025447807915618745
        ])
    }
}


def get_patterson_rule(n):
    if n not in _PATTERSON_TABLE:
        raise ValueError(f"get_patterson_rule: 不支持的阶数 {n}，支持 {list(_PATTERSON_TABLE.keys())}")
    return _PATTERSON_TABLE[n]['x'].copy(), _PATTERSON_TABLE[n]['w'].copy()


def rescale_rule(x, w, a, b):
    scale = (b - a) / 2.0
    shift = (a + b) / 2.0
    return x * scale + shift, w * scale


def patterson_integrate_1d(f_func, a, b, order=15):
    x_std, w_std = get_patterson_rule(order)
    x, w = rescale_rule(x_std, w_std, a, b)
    return np.sum(w * f_func(x))


def patterson_integrate_3d(f_func, xlim, ylim, zlim, order=7):
    x_std, w_x = get_patterson_rule(order)
    y_std, w_y = get_patterson_rule(order)
    z_std, w_z = get_patterson_rule(order)

    x, wx = rescale_rule(x_std, w_x, xlim[0], xlim[1])
    y, wy = rescale_rule(y_std, w_y, ylim[0], ylim[1])
    z, wz = rescale_rule(z_std, w_z, zlim[0], zlim[1])

    result = 0.0
    for i in range(len(x)):
        for j in range(len(y)):
            for k in range(len(z)):
                result += wx[i] * wy[j] * wz[k] * f_func(x[i], y[j], z[k])

    return result


def compute_energy_dissipation_rate(u, v, w, dx, dy, dz, nu):

    def central_diff(f, axis, h):
        df = np.zeros_like(f)
        slc_p = [slice(None)] * 3
        slc_m = [slice(None)] * 3
        slc_c = [slice(None)] * 3
        slc_p[axis] = slice(2, None)
        slc_m[axis] = slice(None, -2)
        slc_c[axis] = slice(1, -1)
        df[tuple(slc_c)] = (f[tuple(slc_p)] - f[tuple(slc_m)]) / (2 * h)
        return df

    dudx = central_diff(u, 0, dx)
    dudy = central_diff(u, 1, dy)
    dudz = central_diff(u, 2, dz)

    dvdx = central_diff(v, 0, dx)
    dvdy = central_diff(v, 1, dy)
    dvdz = central_diff(v, 2, dz)

    dwdx = central_diff(w, 0, dx)
    dwdy = central_diff(w, 1, dy)
    dwdz = central_diff(w, 2, dz)


    s11 = dudx
    s22 = dvdy
    s33 = dwdz
    s12 = 0.5 * (dudy + dvdx)
    s13 = 0.5 * (dudz + dwdx)
    s23 = 0.5 * (dvdz + dwdy)


    s2 = 2.0 * (s11**2 + s22**2 + s33**2 + 2.0 * s12**2 + 2.0 * s13**2 + 2.0 * s23**2)


    nx, ny, nz = u.shape
    n_inner = (nx - 2) * (ny - 2) * (nz - 2)
    if n_inner <= 0:
        n_inner = nx * ny * nz
        epsilon = 2.0 * nu * np.mean(s2)
    else:
        epsilon = 2.0 * nu * np.sum(s2[1:-1, 1:-1, 1:-1]) / n_inner

    return epsilon
