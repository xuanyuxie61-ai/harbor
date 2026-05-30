
import numpy as np
from numpy.polynomial.legendre import leggauss


def jacobi_gauss_lobatto(N):
    if N == 0:
        return np.array([-1.0, 1.0]), np.array([1.0, 1.0])
    



    

    n = N
    J = np.zeros((n + 1, n + 1))
    
    for i in range(1, n + 1):
        beta = i / np.sqrt(4.0 * i * i - 1.0)
        J[i, i-1] = beta
        J[i-1, i] = beta
    


    

    gauss_r, gauss_w = leggauss(n)
    
    r = np.concatenate([[-1.0], gauss_r, [1.0]])
    w = np.concatenate([[2.0 / (n * (n + 1))], gauss_w, [2.0 / (n * (n + 1))]])
    

    r = np.sort(r)
    
    return r, w


def vandermonde_1d(N, r):
    V = np.zeros((len(r), N + 1))
    

    V[:, 0] = 1.0
    if N >= 1:
        V[:, 1] = r
    
    for j in range(1, N):
        V[:, j+1] = ((2.0 * j + 1.0) * r * V[:, j] - j * V[:, j-1]) / (j + 1.0)
    
    return V


def grad_vandermonde_1d(N, r):
    Vr = np.zeros((len(r), N + 1))
    
    Vr[:, 0] = 0.0
    if N >= 1:
        Vr[:, 1] = 1.0
    

    V = vandermonde_1d(N, r)
    
    for j in range(1, N):
        Vr[:, j+1] = Vr[:, j-1] + (2.0 * j + 1.0) * V[:, j]
    
    return Vr


def d_matrix_1d(N, r):
    V = vandermonde_1d(N, r)
    Vr = grad_vandermonde_1d(N, r)
    
    try:
        V_inv = np.linalg.inv(V)
    except np.linalg.LinAlgError:
        V_inv = np.linalg.pinv(V)
    
    D = Vr @ V_inv
    return D


def lift_matrix_1d(N, r):
    V = vandermonde_1d(N, r)
    n_nodes = len(r)
    

    Emat = np.zeros((n_nodes, 2))
    Emat[0, 0] = 1.0
    Emat[-1, 1] = 1.0
    
    lift = V @ V.T @ Emat
    return lift


def mesh_gen_1d(xmin, xmax, K):
    VX = np.linspace(xmin, xmax, K + 1)
    EToV = np.zeros((K, 2), dtype=int)
    
    for k in range(K):
        EToV[k, 0] = k
        EToV[k, 1] = k + 1
    
    return VX, EToV


class DGInternalWaveSolver:
    
    def __init__(self, N=4, K=20, xmin=0.0, xmax=2000.0,
                 wave_speed=1.0, N_buoyancy=0.01):
        self.N = N
        self.K = K
        self.wave_speed = wave_speed
        self.N_buoyancy = N_buoyancy
        

        self.r, _ = jacobi_gauss_lobatto(N)
        self.Np = len(self.r)
        

        self.Dr = d_matrix_1d(N, self.r)
        

        self.LIFT = lift_matrix_1d(N, self.r)
        

        self.VX, self.EToV = mesh_gen_1d(xmin, xmax, K)
        

        self.x = np.zeros((K, self.Np))
        self.dx = (xmax - xmin) / K
        
        for k in range(K):
            x_l = self.VX[self.EToV[k, 0]]
            x_r = self.VX[self.EToV[k, 1]]
            self.x[k, :] = 0.5 * (x_r + x_l) + 0.5 * (x_r - x_l) * self.r
        

        self.rx = 2.0 / self.dx
        self.J = self.dx / 2.0
        

        self.Fscale = 1.0 / self.J
        

        self.vmapM = np.zeros((K, 2), dtype=int)
        self.vmapP = np.zeros((K, 2), dtype=int)
        
        for k in range(K):
            self.vmapM[k, 0] = k * self.Np
            self.vmapM[k, 1] = k * self.Np + N
        

        self.vmapP[0, 0] = (K - 1) * self.Np + N
        self.vmapP[0, 1] = 1 * self.Np
        
        for k in range(1, K - 1):
            self.vmapP[k, 0] = (k - 1) * self.Np + N
            self.vmapP[k, 1] = (k + 1) * self.Np
        
        self.vmapP[K-1, 0] = (K - 2) * self.Np + N
        self.vmapP[K-1, 1] = 0
        

        self.rk4a = np.array([0.0, -567301805773.0/1357537059087.0,
                              -2404267990393.0/2016746695238.0,
                              -3550918686646.0/2091501179385.0,
                              -1275806237668.0/842570457699.0])
        self.rk4b = np.array([1432997174477.0/9575080441755.0,
                              5161836677717.0/13612068292357.0,
                              1720146321549.0/2090206949498.0,
                              3134564353537.0/4481467310338.0,
                              2277821191437.0/14882151754819.0])
        self.rk4c = np.array([0.0, 1432997174477.0/9575080441755.0,
                              2526269341429.0/6820363962896.0,
                              2006345519317.0/3224310063776.0,
                              2802321613138.0/2924317926251.0])
    
    def initial_condition(self, mode='sech2'):
        if mode == 'sech2':
            u = np.zeros((self.K, self.Np))
            for k in range(self.K):
                u[k, :] = 2.0 / np.cosh((self.x[k, :] - 1000.0) / 100.0)**2
        elif mode == 'sin':
            u = np.zeros((self.K, self.Np))
            for k in range(self.K):
                u[k, :] = np.sin(2.0 * np.pi * self.x[k, :] / 2000.0)
        else:
            u = np.zeros((self.K, self.Np))
        
        return u
    
    def source_term(self, u, x, t):

        S_buoyancy = -self.N_buoyancy**2 * np.sin(2.0 * np.pi * x / 2000.0 - 0.1 * t)
        

        S_diss = -1.0e-6 * u
        
        S = S_buoyancy + S_diss
        return S
    
    def rhs_dg(self, u, t):

        dudx = np.zeros_like(u)
        for k in range(self.K):
            dudx[k, :] = self.rx * (self.Dr @ u[k, :])
        

        du = np.zeros((self.K, 2))
        for k in range(K := self.K):

            idxM = self.vmapM[k, 0]
            idxP = self.vmapP[k, 0]
            kM = idxM // self.Np
            iM = idxM % self.Np
            kP = idxP // self.Np
            iP = idxP % self.Np
            
            if kM < self.K and kP < self.K:
                uM = u[kM, iM]
                uP = u[kP, iP]
            else:
                uM = u[k, 0]
                uP = u[k, 0]
            

            alpha = 1.0
            du[k, 0] = self.wave_speed * (uM - uP) * 0.5 * (1.0 - alpha * np.sign(self.wave_speed))
        

            idxM = self.vmapM[k, 1]
            idxP = self.vmapP[k, 1]
            kM = idxM // self.Np
            iM = idxM % self.Np
            kP = idxP // self.Np
            iP = idxP % self.Np
            
            if kM < self.K and kP < self.K:
                uM = u[kM, iM]
                uP = u[kP, iP]
            else:
                uM = u[k, -1]
                uP = u[k, -1]
            
            du[k, 1] = self.wave_speed * (uM - uP) * 0.5 * (1.0 - alpha * np.sign(self.wave_speed))
        

        S = np.zeros_like(u)
        for k in range(self.K):
            S[k, :] = self.source_term(u[k, :], self.x[k, :], t)
        

        rhs = np.zeros_like(u)
        for k in range(self.K):
            flux_term = self.LIFT @ (self.Fscale * du[k, :])
            rhs[k, :] = -self.wave_speed * dudx[k, :] + flux_term + S[k, :]
        
        return rhs
    
    def solve(self, t_final=100.0, dt=0.5):
        u = self.initial_condition()
        
        nsteps = int(t_final / dt)
        t_history = np.zeros(nsteps + 1)
        u_history = np.zeros((nsteps + 1, self.K, self.Np))
        u_history[0, :, :] = u
        
        time = 0.0
        
        for n in range(nsteps):
            resu = np.zeros_like(u)
            
            for INTRK in range(5):
                rhsu = self.rhs_dg(u, time)
                resu = self.rk4a[INTRK] * resu + dt * rhsu
                u = u + self.rk4b[INTRK] * resu
            
            time += dt
            t_history[n+1] = time
            u_history[n+1, :, :] = u
            

            u = np.clip(u, -10.0, 10.0)
        
        return t_history, u_history
