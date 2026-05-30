
import numpy as np
from typing import Callable, Tuple


def legendre_com(n: int) -> Tuple[np.ndarray, np.ndarray]:
    if n < 1:
        raise ValueError("n must be at least 1")
    



    x = np.zeros(n)
    w = np.zeros(n)
    
    m = (n + 1) // 2
    eps = 1e-14
    
    for i in range(1, m + 1):

        z = np.cos(np.pi * (i - 0.25) / (n + 0.5))
        z1 = 0.0
        while abs(z - z1) > eps:
            p1 = 1.0
            p2 = 0.0
            for j in range(1, n + 1):
                p3 = p2
                p2 = p1
                p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j
            pp = n * (z * p1 - p2) / (z * z - 1.0)
            z1 = z
            z = z1 - p1 / pp
        x[i - 1] = -z
        x[n - i] = z
        w[i - 1] = 2.0 / ((1.0 - z * z) * pp * pp)
        w[n - i] = w[i - 1]
    return x, w


def local_basis_1d(order: int, node_x: np.ndarray, x: float) -> np.ndarray:
    if len(node_x) != order:
        raise ValueError("node_x length must equal order")
    phi = np.ones(order)
    for i in range(order):
        for j in range(order):
            if i != j:
                denom = node_x[i] - node_x[j]
                if abs(denom) < 1e-14:
                    raise ValueError("Nodes too close")
                phi[i] *= (x - node_x[j]) / denom
    return phi


def local_basis_prime_1d(order: int, node_x: np.ndarray, x: float) -> np.ndarray:
    dphi = np.zeros(order)
    for i in range(order):
        for j in range(order):
            if i != j:
                denom_ij = node_x[i] - node_x[j]
                if abs(denom_ij) < 1e-14:
                    raise ValueError("Nodes too close")
                term = 1.0 / denom_ij
                for k in range(order):
                    if k != i and k != j:
                        denom_jk = node_x[j] - node_x[k]
                        if abs(denom_jk) < 1e-14:
                            raise ValueError("Nodes too close")
                        term *= (x - node_x[k]) / denom_jk
                dphi[i] += term
    return dphi


def solve_steady_smoluchowski_1d(x_nodes: np.ndarray, free_energy: np.ndarray,
                                 D: float = 1.0, kT: float = 1.0,
                                 p_left: float = 1.0, p_right: float = 1.0) -> np.ndarray:
    N = len(x_nodes)
    if len(free_energy) != N:
        raise ValueError("x_nodes and free_energy must have the same length")
    if N < 3:
        raise ValueError("Need at least 3 nodes")
    

    A = np.zeros((N, N))
    b = np.zeros(N)
    

    gauss_xi, gauss_w = legendre_com(2)
    
    for e in range(N - 1):
        xL, xR = x_nodes[e], x_nodes[e + 1]
        h = xR - xL
        if h <= 0:
            raise ValueError("x_nodes must be strictly increasing")
        

        dF_dx = (free_energy[e + 1] - free_energy[e]) / h
        

        Ke = np.zeros((2, 2))
        for q in range(len(gauss_xi)):
            xi = gauss_xi[q]
            w = gauss_w[q]


            N1 = 0.5 * (1.0 - xi)
            N2 = 0.5 * (1.0 + xi)
            dN1 = -0.5
            dN2 = 0.5
            






            raise NotImplementedError("Hole 2: 请补全单元刚度矩阵漂移-扩散算子组装")
        

        A[e, e] += Ke[0, 0]
        A[e, e + 1] += Ke[0, 1]
        A[e + 1, e] += Ke[1, 0]
        A[e + 1, e + 1] += Ke[1, 1]
    

    A[0, :] = 0.0
    A[0, 0] = 1.0
    b[0] = p_left
    A[-1, :] = 0.0
    A[-1, -1] = 1.0
    b[-1] = p_right
    
    p = np.linalg.solve(A, b)

    p = np.maximum(p, 1e-12)
    return p


def solve_fokker_planck_eigenvalue_1d(x_nodes: np.ndarray, potential: np.ndarray,
                                      D: float = 1.0, kT: float = 1.0,
                                      n_modes: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    N = len(x_nodes)
    K = np.zeros((N, N))
    M = np.zeros((N, N))
    
    gauss_xi, gauss_w = legendre_com(3)
    
    for e in range(N - 1):
        xL, xR = x_nodes[e], x_nodes[e + 1]
        h = xR - xL
        jac = 0.5 * h
        
        dV_dx = (potential[e + 1] - potential[e]) / h
        
        Ke = np.zeros((2, 2))
        Me = np.zeros((2, 2))
        for q in range(len(gauss_xi)):
            xi = gauss_xi[q]
            w = gauss_w[q]
            N1 = 0.5 * (1.0 - xi)
            N2 = 0.5 * (1.0 + xi)
            dN1 = -0.5
            dN2 = 0.5
            

            factor = D * w / jac
            Ke[0, 0] += factor * dN1 * dN1
            Ke[0, 1] += factor * dN1 * dN2
            Ke[1, 0] += factor * dN2 * dN1
            Ke[1, 1] += factor * dN2 * dN2
            

            factor_mass = jac * w
            Me[0, 0] += factor_mass * N1 * N1
            Me[0, 1] += factor_mass * N1 * N2
            Me[1, 0] += factor_mass * N2 * N1
            Me[1, 1] += factor_mass * N2 * N2
            

            drift = (D / kT) * dV_dx * w
            Ke[0, 0] += drift * N1 * dN1
            Ke[0, 1] += drift * N1 * dN2
            Ke[1, 0] += drift * N2 * dN1
            Ke[1, 1] += drift * N2 * dN2
        
        idx = [e, e + 1]
        for i in range(2):
            for j in range(2):
                K[idx[i], idx[j]] += Ke[i, j]
                M[idx[i], idx[j]] += Me[i, j]
    

    K[0, :] = 0.0
    K[0, 0] = 1.0
    M[0, :] = 0.0
    M[0, 0] = 1e-10
    K[-1, :] = 0.0
    K[-1, -1] = 1.0
    M[-1, :] = 0.0
    M[-1, -1] = 1e-10
    

    eigvals, eigvecs = np.linalg.eig(np.linalg.solve(M + 1e-12 * np.eye(N), K))

    idx_sorted = np.argsort(np.real(eigvals))
    eigvals = np.real(eigvals[idx_sorted])
    eigvecs = np.real(eigvecs[:, idx_sorted])
    return eigvals[:n_modes], eigvecs[:, :n_modes]
