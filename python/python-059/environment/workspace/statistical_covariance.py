
import numpy as np
from math import sqrt, pi, exp, lgamma
from numerical_utils import rnorm, wilson_hilferty_chi_square


class StatisticalError(Exception):
    pass


def cholesky_factor(Sigma):
    Sigma = np.asarray(Sigma, dtype=np.float64)
    p = Sigma.shape[0]
    if Sigma.shape[0] != Sigma.shape[1]:
        raise StatisticalError("cholesky_factor: 输入必须为方阵")

    try:
        L = np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:

        Sigma = Sigma + np.eye(p) * 1e-8
        L = np.linalg.cholesky(Sigma)


    d = []
    for j in range(p):
        for i in range(j + 1):
            d.append(L[i, j])
    return np.array(d, dtype=np.float64)


def wishart_variate(Sigma, n, np_var=None):
    Sigma = np.asarray(Sigma, dtype=np.float64)
    if np_var is None:
        np_var = Sigma.shape[0]
    p = np_var

    if n < 1 or n > p:
        raise StatisticalError(f"wishart_variate: 自由度 n={n} 必须在 [1, p={p}]")

    d = cholesky_factor(Sigma)
    nnp = p * (p + 1) // 2


    sb = np.zeros(nnp, dtype=np.float64)
    k = 0
    while k < nnp:
        z1, z2 = rnorm()
        sb[k] = z1
        k += 1
        if k < nnp:
            sb[k] = z2
            k += 1


    sa = np.zeros(nnp, dtype=np.float64)
    ns = 0
    for i in range(1, p + 1):
        df = p - i + 1
        ns += i
        sb[ns - 1] = wilson_hilferty_chi_square(df, sb[ns - 1])



    D_mat = np.zeros((p, p), dtype=np.float64)
    idx = 0
    for j in range(p):
        for i in range(j + 1):
            D_mat[i, j] = d[idx]
            idx += 1

    SB_mat = np.zeros((p, p), dtype=np.float64)
    idx = 0
    for j in range(p):
        for i in range(j + 1):
            SB_mat[i, j] = sb[idx]
            idx += 1

    SA_mat = D_mat @ SB_mat

    S = (SA_mat @ SA_mat.T) / n
    return S


def sample_covariance_matrix(data):
    data = np.asarray(data, dtype=np.float64)
    n = data.shape[0]
    if n < 2:
        raise StatisticalError("sample_covariance_matrix: 样本数至少为 2")
    mean = np.mean(data, axis=0)
    centered = data - mean
    S = (centered.T @ centered) / (n - 1)
    return S


def eof_analysis(cov_matrix, num_modes=None):
    cov = np.asarray(cov_matrix, dtype=np.float64)
    p = cov.shape[0]
    eigvals, eigvecs = np.linalg.eigh(cov)


    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]


    eigvals = np.maximum(eigvals, 0.0)
    total = np.sum(eigvals)
    if total > 0:
        evr = eigvals / total
    else:
        evr = np.zeros_like(eigvals)

    if num_modes is not None:
        eigvals = eigvals[:num_modes]
        eigvecs = eigvecs[:, :num_modes]
        evr = evr[:num_modes]

    return eigvals, eigvecs, evr


def aod_covariance_model(stations_lat_lon, correlation_length=500.0, sigma_aod=0.15):
    from atmospheric_mesh import ll_degrees_to_distance_earth

    N = stations_lat_lon.shape[0]
    Sigma = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        Sigma[i, i] = sigma_aod ** 2
        for j in range(i + 1, N):
            d = ll_degrees_to_distance_earth(
                stations_lat_lon[i, 0], stations_lat_lon[i, 1],
                stations_lat_lon[j, 0], stations_lat_lon[j, 1]
            )
            val = (sigma_aod ** 2) * exp(-d / correlation_length)
            Sigma[i, j] = val
            Sigma[j, i] = val


    eigvals = np.linalg.eigvalsh(Sigma)
    min_eig = np.min(eigvals)
    if min_eig < 1e-10:
        Sigma = Sigma + np.eye(N) * (1e-10 - min_eig)
    return Sigma
