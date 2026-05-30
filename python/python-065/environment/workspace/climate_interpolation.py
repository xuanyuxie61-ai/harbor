
import numpy as np


def nearest_interp_1d(xd, yd, xi):
    xd = np.asarray(xd).reshape(-1)
    yd = np.asarray(yd).reshape(-1)
    xi = np.asarray(xi).reshape(-1)

    if xd.shape[0] != yd.shape[0]:
        raise ValueError("xd 和 yd 长度必须相同")
    if xd.shape[0] < 1:
        raise ValueError("至少需要一个数据点")

    idx = np.searchsorted(xd, xi)
    idx = np.clip(idx, 1, len(xd) - 1)
    left = xd[idx - 1]
    right = xd[idx]

    mask = (xi - left) <= (right - xi)
    idx[mask] -= 1
    return yd[idx]


def poly_eval(c, x):
    c = np.asarray(c)
    x = np.asarray(x)
    d = len(c) - 1
    value = c[0] * np.ones_like(x)
    xi = np.ones_like(x)
    for i in range(1, d + 1):
        xi = xi * x
        value = value + c[i] * xi
    return value


def wdk_roots(c, tol=1e-12, max_iter=1000):
    c = np.asarray(c)
    d = len(c) - 1
    if d < 1:
        return np.array([], dtype=complex)


    R = 1.0 + np.max(np.abs(c[:-1] / c[-1]))


    theta = np.linspace(0.0, 2.0 * np.pi, d + 1)[:-1]
    roots = R * np.exp(1j * theta)

    for _ in range(max_iter):
        roots_old = roots.copy()
        for i in range(d):
            zi = roots_old[i]
            denom = np.prod(zi - np.delete(roots, i))
            if abs(denom) < 1e-14:
                continue
            roots[i] = zi - poly_eval(c, zi) / denom

        max_change = np.max(np.abs(roots - roots_old))
        if max_change < tol:
            break

    return roots


def characteristic_roots_from_covariance(cov_matrix):
    C = np.asarray(cov_matrix)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("输入必须是方阵")

    eigvals = np.linalg.eigvals(C)
    return eigvals


def regrid_field_nearest(lon_src, lat_src, field_src, lon_tgt, lat_tgt):
    nlat_tgt = len(lat_tgt)
    nlon_tgt = len(lon_tgt)
    field_tgt = np.zeros((nlat_tgt, nlon_tgt), dtype=field_src.dtype)

    for i in range(nlat_tgt):
        lat_i = lat_tgt[i]
        lat_idx = int(np.argmin(np.abs(lat_src - lat_i)))
        for j in range(nlon_tgt):
            lon_j = lon_tgt[j]
            lon_idx = int(np.argmin(np.abs(lon_src - lon_j)))
            field_tgt[i, j] = field_src[lat_idx, lon_idx]
    return field_tgt


def dominant_scale_analysis(field_1d, max_degree=8):
    n = len(field_1d)
    fft_vals = np.fft.rfft(field_1d)
    power = np.abs(fft_vals) ** 2
    freqs = np.fft.rfftfreq(n)


    x = freqs[1:max_degree + 2]
    y = np.log(power[1:max_degree + 2] + 1e-14)
    coeffs = np.polyfit(x, y, max_degree)

    deriv_coeffs = np.polyder(coeffs)
    roots = wdk_roots(deriv_coeffs)

    real_roots = np.real(roots[np.abs(np.imag(roots)) < 1e-6])
    real_roots = real_roots[real_roots > 0]
    return real_roots, coeffs, freqs, power


def test_interpolation():
    xd = np.array([0.0, 1.0, 2.0, 3.0])
    yd = np.array([10.0, 20.0, 30.0, 40.0])
    xi = np.array([0.1, 1.8, 2.5])
    yi = nearest_interp_1d(xd, yd, xi)
    assert yi[0] == 10.0
    assert yi[1] == 30.0


    roots = wdk_roots(np.array([-1.0, 0.0, 1.0]))
    assert len(roots) == 2
    print("climate_interpolation 自测试通过")


if __name__ == "__main__":
    test_interpolation()
