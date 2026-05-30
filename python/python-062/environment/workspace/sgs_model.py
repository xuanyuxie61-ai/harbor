
import numpy as np


def compute_strain_rate_tensor(u, v, w, dx, dy, dz):
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

    S11 = dudx
    S22 = dvdy
    S33 = dwdz
    S12 = 0.5 * (dudy + dvdx)
    S13 = 0.5 * (dudz + dwdx)
    S23 = 0.5 * (dvdz + dwdy)

    return S11, S22, S33, S12, S13, S23


def smagorinsky_model(u, v, w, dx, dy, dz, Cs=0.16):








    raise NotImplementedError("HOLE 1: 请实现 Smagorinsky SGS 模型核心公式")



def dynamic_smagorinsky_model(u, v, w, dx, dy, dz, test_filter_width=2):
    try:
        from scipy.ndimage import uniform_filter
    except ImportError:

        def uniform_filter(arr, size, mode='nearest'):
            from scipy.ndimage import uniform_filter as uf
            return uf(arr, size=size, mode=mode)

    S11, S22, S33, S12, S13, S23 = compute_strain_rate_tensor(u, v, w, dx, dy, dz)

    S2 = 2.0 * (S11**2 + S22**2 + S33**2 + 2.0 * S12**2 + 2.0 * S13**2 + 2.0 * S23**2)
    S_mag = np.sqrt(np.clip(S2, 0.0, 1e12))

    Delta = (dx * dy * dz) ** (1.0 / 3.0)


    w_test = max(test_filter_width, 2)

    def safe_filter(f):
        return uniform_filter(f, size=w_test, mode='nearest')


    u_hat = safe_filter(u)
    v_hat = safe_filter(v)
    w_hat = safe_filter(w)


    Sh11, Sh22, Sh33, Sh12, Sh13, Sh23 = compute_strain_rate_tensor(
        u_hat, v_hat, w_hat, dx, dy, dz)
    Sh2 = 2.0 * (Sh11**2 + Sh22**2 + Sh33**2 + 2.0 * Sh12**2 + 2.0 * Sh13**2 + 2.0 * Sh23**2)
    Sh_mag = np.sqrt(np.clip(Sh2, 0.0, 1e12))


    L11 = safe_filter(u * u) - u_hat * u_hat
    L22 = safe_filter(v * v) - v_hat * v_hat
    L33 = safe_filter(w * w) - w_hat * w_hat
    L12 = safe_filter(u * v) - u_hat * v_hat
    L13 = safe_filter(u * w) - u_hat * w_hat
    L23 = safe_filter(v * w) - v_hat * w_hat


    alpha = 2.0
    scale = 2.0 * Delta**2
    M11 = scale * (alpha**2 * Sh_mag * Sh11 - S_mag * S11)
    M22 = scale * (alpha**2 * Sh_mag * Sh22 - S_mag * S22)
    M33 = scale * (alpha**2 * Sh_mag * Sh33 - S_mag * S33)
    M12 = scale * (alpha**2 * Sh_mag * Sh12 - S_mag * S12)
    M13 = scale * (alpha**2 * Sh_mag * Sh13 - S_mag * S13)
    M23 = scale * (alpha**2 * Sh_mag * Sh23 - S_mag * S23)


    LM = L11 * M11 + L22 * M22 + L33 * M33 + 2.0 * (L12 * M12 + L13 * M13 + L23 * M23)
    MM = M11**2 + M22**2 + M33**2 + 2.0 * (M12**2 + M13**2 + M23**2)


    LM = np.clip(LM, -1e12, 1e12)
    MM = np.clip(MM, 1e-30, 1e12)


    nz = u.shape[2]
    C_dynamic = np.zeros_like(u)
    for k in range(nz):
        lm_sum = np.sum(LM[:, :, k])
        mm_sum = np.sum(MM[:, :, k])
        if abs(mm_sum) > 1e-15:
            c_k = lm_sum / mm_sum
        else:
            c_k = 0.0
        C_dynamic[:, :, k] = c_k


    C_dynamic = np.clip(C_dynamic, -0.5, 0.5)


    nu_sgs = C_dynamic * Delta**2 * S_mag
    nu_sgs = np.clip(nu_sgs, 0.0, 5.0)

    return nu_sgs, C_dynamic
