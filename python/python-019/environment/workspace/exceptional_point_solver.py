
import numpy as np
from hamiltonian_builder import (
    build_pt_symmetric_hamiltonian_1d,
    build_pt_symmetric_hamiltonian_2d,
    build_nonhermitian_ssh_hamiltonian,
    discriminant_2x2,
)


def laguerre_root_find(f, x0, degree, abserr=1e-12, kmax=100):
    if degree < 2:
        raise ValueError("degree must be at least 2.")
    x = x0
    ierror = 0
    k = 0
    beta = 1.0 / (degree - 1)

    while True:
        fx = f(x, 0)
        if abs(fx) <= abserr:
            break
        k += 1
        if k > kmax:
            ierror = 2
            return x, ierror, k

        dfx = f(x, 1)
        d2fx = f(x, 2)

        z = dfx ** 2 - (beta + 1.0) * fx * d2fx

        z = complex(z)
        if z.real < 0:

            pass
        bot = beta * dfx + np.sqrt(z)

        if abs(bot) < 1e-30:
            ierror = 3
            return x, ierror, k

        dx = - (beta + 1.0) * fx / bot
        x = x + dx

    return x, ierror, k


def find_exceptional_points_1d(t=1.0, m=0.5, gamma=0.3, k_guess_grid=64):

    def delta_func(k, ider=0):
        eps = 1e-8
        if ider == 0:
            H = build_pt_symmetric_hamiltonian_1d(k, t, m, gamma)
            return discriminant_2x2(H)
        elif ider == 1:
            Hp = build_pt_symmetric_hamiltonian_1d(k + eps, t, m, gamma)
            Hm = build_pt_symmetric_hamiltonian_1d(k - eps, t, m, gamma)
            return (discriminant_2x2(Hp) - discriminant_2x2(Hm)) / (2.0 * eps)
        elif ider == 2:
            Hp = build_pt_symmetric_hamiltonian_1d(k + eps, t, m, gamma)
            H0 = build_pt_symmetric_hamiltonian_1d(k, t, m, gamma)
            Hm = build_pt_symmetric_hamiltonian_1d(k - eps, t, m, gamma)
            return (discriminant_2x2(Hp) - 2.0 * discriminant_2x2(H0) + discriminant_2x2(Hm)) / (eps ** 2)
        else:
            raise ValueError("ider must be 0, 1, or 2.")


    re_vals = np.linspace(-np.pi, np.pi, k_guess_grid)
    im_vals = np.linspace(-2.0, 2.0, k_guess_grid // 2)

    roots_found = []
    tol_merge = 1e-6

    for re_k in re_vals:
        for im_k in im_vals:
            k0 = re_k + 1j * im_k
            root, ierr, _ = laguerre_root_find(delta_func, k0, degree=4, abserr=1e-14, kmax=80)
            if ierr == 0:

                is_new = True
                for existing in roots_found:
                    if abs(root - existing) < tol_merge:
                        is_new = False
                        break
                if is_new:
                    roots_found.append(root)

    return roots_found


def find_exceptional_points_ssh(t1=1.0, t2=0.5, gamma=0.2, k_guess_grid=48):
    def delta_func(k, ider=0):


        raise NotImplementedError("SSH discriminant function is missing.")

    re_vals = np.linspace(-np.pi, np.pi, k_guess_grid)
    im_vals = np.linspace(-1.5, 1.5, k_guess_grid // 2)
    roots_found = []
    tol_merge = 1e-6

    for re_k in re_vals:
        for im_k in im_vals:
            k0 = re_k + 1j * im_k
            root, ierr, _ = laguerre_root_find(delta_func, k0, degree=4, abserr=1e-14, kmax=80)
            if ierr == 0:
                is_new = True
                for existing in roots_found:
                    if abs(root - existing) < tol_merge:
                        is_new = False
                        break
                if is_new:
                    roots_found.append(root)

    return roots_found


def local_exceptional_point_order(H, param, dH_dparam, eps=1e-8):
    from hamiltonian_builder import discriminant_2x2

    dH = dH_dparam(param)

    delta0 = discriminant_2x2(H)
    Hp = H + eps * dH
    Hm = H - eps * dH
    delta1 = (discriminant_2x2(Hp) - discriminant_2x2(Hm)) / (2.0 * eps)

    if abs(delta0) > 1e-8:
        return 1
    if abs(delta1) > 1e-6:
        return 2

    return 3
