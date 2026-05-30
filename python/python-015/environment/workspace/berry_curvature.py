
import numpy as np
from typing import Tuple
from weyl_hamiltonian import WeylHamiltonian, velocity_operator


def berry_connection_numeric(ham: WeylHamiltonian, k: np.ndarray,
                              band_index: int = 0,
                              delta: float = 1e-6) -> np.ndarray:
    k = np.asarray(k, dtype=float)
    if k.shape != (3,):
        raise ValueError("k必须是三维矢量")
    
    A = np.zeros(3, dtype=complex)
    

    _, u0 = ham.eigenproblem(k)
    u_ref = u0[:, band_index].copy()
    

    phase = np.exp(-1.0j * np.angle(u_ref[0])) if abs(u_ref[0]) > 1e-14 else 1.0
    u_ref = u_ref * phase
    
    for a in range(3):
        kp = k.copy()
        km = k.copy()
        kp[a] += delta
        km[a] -= delta
        
        _, up = ham.eigenproblem(kp)
        _, um = ham.eigenproblem(km)
        
        up_vec = up[:, band_index]
        um_vec = um[:, band_index]
        

        if abs(up_vec[0]) > 1e-14:
            up_vec *= np.exp(-1.0j * np.angle(up_vec[0]))
        if abs(um_vec[0]) > 1e-14:
            um_vec *= np.exp(-1.0j * np.angle(um_vec[0]))
        

        du = (up_vec - um_vec) / (2.0 * delta)
        A[a] = 1.0j * np.vdot(u_ref, du)
    

    if np.max(np.abs(A.imag)) > 1e-8:
        raise RuntimeWarning(f"Berry联络虚部过大: {np.max(np.abs(A.imag))}")
    
    return A.real


def berry_curvature_numeric(ham: WeylHamiltonian, k: np.ndarray,
                             band_index: int = 0,
                             delta: float = 1e-5) -> np.ndarray:
    k = np.asarray(k, dtype=float)
    energies, eigenvectors = ham.eigenproblem(k)
    
    if band_index not in (0, 1):
        raise ValueError("band_index必须是0或1")
    
    n_bands = 2
    Omega = np.zeros((3, 3))
    

    v_ops = velocity_operator(ham, k, delta)
    








    raise NotImplementedError("Hole_2: 数值Berry曲率张量计算待实现")


def berry_curvature_analytic_linear(k: np.ndarray, chirality: int = 1) -> np.ndarray:
    k = np.asarray(k, dtype=float)
    k_norm = np.linalg.norm(k)
    
    if k_norm < 1e-14:

        return np.zeros((3, 3))
    
    Omega = np.zeros((3, 3))
    sign = chirality
    
    for i in range(3):
        for j in range(3):

            k_idx = 3 - i - j
            if i == j or k_idx < 0 or k_idx > 2:
                continue

            eps = 1 if ((i, j, k_idx) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]) else -1
            Omega[i, j] = sign * 0.5 * eps * k[k_idx] / (k_norm ** 3)
    
    return Omega


def berry_phase_1d(ham: WeylHamiltonian, path: np.ndarray,
                    band_index: int = 0) -> float:
    if path.ndim != 2 or path.shape[1] != 3:
        raise ValueError("path必须是(N,3)数组")
    
    n_points = path.shape[0]
    if n_points < 2:
        return 0.0
    

    eigenvectors = []
    for i in range(n_points):
        _, v = ham.eigenproblem(path[i])
        vec = v[:, band_index].copy()

        if abs(vec[0]) > 1e-14:
            vec *= np.exp(-1.0j * np.angle(vec[0]))
        eigenvectors.append(vec)
    

    prod = 1.0 + 0.0j
    for i in range(n_points - 1):
        overlap = np.vdot(eigenvectors[i], eigenvectors[i + 1])
        prod *= overlap
    

    if np.linalg.norm(path[0] - path[-1]) < 1e-10:
        overlap = np.vdot(eigenvectors[-1], eigenvectors[0])
        prod *= overlap
    
    gamma = -np.angle(prod)
    return gamma


def chern_number_2d_slice(ham: WeylHamiltonian,
                           kx_range: Tuple[float, float],
                           ky_range: Tuple[float, float],
                           kz_fixed: float,
                           grid_size: int = 40,
                           band_index: int = 0) -> float:
    kx = np.linspace(kx_range[0], kx_range[1], grid_size)
    ky = np.linspace(ky_range[0], ky_range[1], grid_size)
    dkx = (kx_range[1] - kx_range[0]) / (grid_size - 1) if grid_size > 1 else 0.0
    dky = (ky_range[1] - ky_range[0]) / (grid_size - 1) if grid_size > 1 else 0.0
    
    total = 0.0
    for i in range(grid_size):
        for j in range(grid_size):
            k = np.array([kx[i], ky[j], kz_fixed])
            Omega = berry_curvature_numeric(ham, k, band_index)
            total += Omega[0, 1]
    
    chern = total * dkx * dky / (2.0 * np.pi)
    return chern


def weyl_charge_surface_integral(ham: WeylHamiltonian,
                                  center: np.ndarray,
                                  radius: float,
                                  n_theta: int = 20,
                                  n_phi: int = 20,
                                  band_index: int = 0) -> float:
    center = np.asarray(center, dtype=float)
    
    theta = np.linspace(0.0, np.pi, n_theta)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi)
    d_theta = np.pi / (n_theta - 1) if n_theta > 1 else 0.0
    d_phi = 2.0 * np.pi / (n_phi - 1) if n_phi > 1 else 0.0
    
    total = 0.0
    for i in range(n_theta):
        st = np.sin(theta[i])
        ct = np.cos(theta[i])
        for j in range(n_phi):
            sp = np.sin(phi[j])
            cp = np.cos(phi[j])
            

            r_vec = radius * np.array([st * cp, st * sp, ct])
            k = center + r_vec
            

            Omega = berry_curvature_numeric(ham, k, band_index)
            


            dS = r_vec * radius * st * d_theta * d_phi
            


            omega_vec = np.array([Omega[1, 2], Omega[2, 0], Omega[0, 1]])
            total += np.dot(omega_vec, dS)
    
    charge = total / (2.0 * np.pi)
    return charge
