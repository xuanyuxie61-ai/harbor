
import numpy as np


def l2_inner_product(f, g, r, method='trapezoid'):
    f = np.asarray(f)
    g = np.asarray(g)
    r = np.asarray(r)

    if len(f) != len(g) or len(f) != len(r):
        raise ValueError("f, g, r 长度必须一致")
    if len(r) < 2:
        raise ValueError("至少需要 2 个网格点")

    integrand = np.conj(f) * g

    if method == 'trapezoid':
        return np.trapezoid(integrand, r)
    elif method == 'simpson':
        from scipy.integrate import simpson
        return simpson(integrand, x=r)
    else:
        raise ValueError(f"未知积分方法: {method}")


def l2_norm(f, r, method='trapezoid'):
    return np.sqrt(np.abs(l2_inner_product(f, f, r, method)))


def normalize_wavefunction(u, r, method='trapezoid'):
    norm = l2_norm(u, r, method)
    if norm < 1e-30:
        return u.copy()
    return u / norm


def check_orthogonality(wavefunctions, r, threshold=1e-10):
    n = len(wavefunctions)
    overlap = np.zeros((n, n), dtype=complex)
    for i in range(n):
        for j in range(n):
            overlap[i, j] = l2_inner_product(wavefunctions[i], wavefunctions[j], r)


    diag_deviation = np.max(np.abs(np.diag(overlap) - 1.0))
    offdiag_max = np.max(np.abs(overlap - np.eye(n)))
    is_orthogonal = (diag_deviation < threshold) and (offdiag_max < threshold)

    return overlap, is_orthogonal


def gram_schmidt_orthogonalization(wavefunctions, r, method='trapezoid'):
    ortho = []
    for u in wavefunctions:
        v = u.copy()
        for e in ortho:
            proj = l2_inner_product(e, v, r, method)
            v = v - proj * e
        norm = l2_norm(v, r, method)
        if norm > 1e-30:
            v = v / norm
        ortho.append(v)
    return ortho


def coupling_matrix_element(u_alpha, u_beta, V_coupl, r, method='trapezoid'):
    integrand = np.conj(u_alpha) * V_coupl * u_beta
    if method == 'trapezoid':
        return np.trapezoid(integrand, r)
    else:
        from scipy.integrate import simpson
        return simpson(integrand, x=r)


def deformation_coupling_potential(r, beta_l, R0, a, l_order):
    from optical_potential import woods_saxon_derivative

    dV_dr = woods_saxon_derivative(r, 1.0, R0, a)

    V_coupl = -beta_l * R0 * dV_dr
    return V_coupl


def coupled_channels_overlap(wavefunctions_dict, r, method='trapezoid'):
    labels = list(wavefunctions_dict.keys())
    n = len(labels)
    overlap = np.zeros((n, n), dtype=complex)

    for i in range(n):
        for j in range(n):
            overlap[i, j] = l2_inner_product(
                wavefunctions_dict[labels[i]],
                wavefunctions_dict[labels[j]],
                r, method
            )

    return overlap, labels


if __name__ == "__main__":

    r = np.linspace(0.01, 10.0, 200)
    f = np.sin(r) * np.exp(-r)
    g = np.cos(r) * np.exp(-r)

    inner = l2_inner_product(f, g, r)
    norm_f = l2_norm(f, r)
    print(f"⟨f|g⟩ = {inner:.6f}")
    print(f"||f|| = {norm_f:.6f}")


    u1 = np.sin(r) * np.exp(-r / 2)
    u2 = r * np.sin(r) * np.exp(-r / 2)
    ortho = gram_schmidt_orthogonalization([u1, u2], r)
    overlap, is_ortho = check_orthogonality(ortho, r)
    print(f"正交化后重叠矩阵:\n{np.round(overlap, 6)}")
    print(f"是否正交: {is_ortho}")


    V_c = deformation_coupling_potential(r, beta_l=0.2, R0=5.0, a=0.65, l_order=2)
    V_ab = coupling_matrix_element(ortho[0], ortho[1], V_c, r)
    print(f"耦合矩阵元 V_01 = {V_ab:.6f}")
