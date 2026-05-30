
import numpy as np
from typing import Tuple






def laplacian9_torus(field: np.ndarray, dx: float, dy: float) -> np.ndarray:
    nx, ny = field.shape
    L = np.zeros_like(field)
    


    for i in range(nx):
        for j in range(ny):

            im1 = (i - 1) % nx
            ip1 = (i + 1) % nx
            jm1 = (j - 1) % ny
            jp1 = (j + 1) % ny
            
            L[i, j] = (
                4.0 * field[im1, j] + 4.0 * field[ip1, j] +
                4.0 * field[i, jm1] + 4.0 * field[i, jp1] +
                field[im1, jm1] + field[im1, jp1] +
                field[ip1, jm1] + field[ip1, jp1] -
                20.0 * field[i, j]
            ) / (6.0 * dx * dy)
    
    return L


def laplacian5_zero_boundary(field: np.ndarray, dx: float, dy: float) -> np.ndarray:
    nx, ny = field.shape
    L = np.zeros_like(field)
    h2 = dx * dy
    
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            L[i, j] = (
                field[i - 1, j] + field[i + 1, j] +
                field[i, j - 1] + field[i, j + 1] -
                4.0 * field[i, j]
            ) / h2
    

    return L






def gray_scott_synaptic_step(
    U: np.ndarray,
    V: np.ndarray,
    Du: float,
    Dv: float,
    F: float,
    K: float,
    dt: float,
    dx: float,
    dy: float,
    boundary: str = 'periodic'
) -> Tuple[np.ndarray, np.ndarray]:
    if boundary == 'periodic':
        LU = laplacian9_torus(U, dx, dy)
        LV = laplacian9_torus(V, dx, dy)
    else:
        LU = laplacian5_zero_boundary(U, dx, dy)
        LV = laplacian5_zero_boundary(V, dx, dy)
    

    UV2 = U * V ** 2
    

    U_new = U + dt * (Du * LU - UV2 + F * (1.0 - U))
    V_new = V + dt * (Dv * LV + UV2 - (F + K) * V)
    

    U_new = np.clip(U_new, 0.0, 1.0)
    V_new = np.clip(V_new, 0.0, 1.0)
    
    return U_new, V_new


def simulate_synaptic_transmission(
    nx: int,
    ny: int,
    n_steps: int,
    Du: float = 0.16,
    Dv: float = 0.08,
    F: float = 0.035,
    K: float = 0.060,
    dt: float = 1.0,
    dx: float = 0.5,
    dy: float = 0.5,
    initial_condition: str = 'localized',
    boundary: str = 'periodic'
) -> dict:

    stable_dt = (dx * dy) / (4.0 * max(Du, Dv) + 1e-14)
    if dt > stable_dt:
        print(f"Warning: dt={dt} exceeds stability limit {stable_dt:.4f}. Adjusting...")
        dt = 0.5 * stable_dt
    

    U = np.ones((nx, ny), dtype=np.float64)
    V = np.zeros((nx, ny), dtype=np.float64)
    
    if initial_condition == 'localized':

        cx, cy = nx // 2, ny // 2
        radius = min(nx, ny) // 10
        for i in range(nx):
            for j in range(ny):
                dist2 = (i - cx) ** 2 + (j - cy) ** 2
                if dist2 < radius ** 2:
                    U[i, j] = 0.5
                    V[i, j] = 0.25
    elif initial_condition == 'wavefront':

        for i in range(nx):
            for j in range(ny):
                if i < nx // 4:
                    U[i, j] = 0.5
                    V[i, j] = 0.25
    elif initial_condition == 'random':
        np.random.seed(42)
        noise = np.random.random((nx, ny))
        U = U - 0.1 * noise
        V = 0.1 * noise
    

    save_interval = max(1, n_steps // 100)
    n_saved = n_steps // save_interval + 1
    U_history = np.zeros((n_saved, nx, ny), dtype=np.float64)
    V_history = np.zeros((n_saved, nx, ny), dtype=np.float64)
    
    U_history[0] = U.copy()
    V_history[0] = V.copy()
    save_idx = 1
    

    for step in range(n_steps):
        U, V = gray_scott_synaptic_step(U, V, Du, Dv, F, K, dt, dx, dy, boundary)
        
        if (step + 1) % save_interval == 0 and save_idx < n_saved:
            U_history[save_idx] = U.copy()
            V_history[save_idx] = V.copy()
            save_idx += 1
    
    return {
        'U_history': U_history[:save_idx],
        'V_history': V_history[:save_idx],
        'final_U': U,
        'final_V': V,
        'n_steps': n_steps,
        'dt': dt,
        'save_interval': save_interval,
    }






def compute_synaptic_efficacy(
    V_field: np.ndarray,
    threshold: float = 0.1,
    receptor_density: np.ndarray = None
) -> dict:
    if receptor_density is None:
        receptor_density = np.ones_like(V_field)
    
    peak_conc = float(np.max(V_field))
    mean_conc = float(np.mean(V_field))
    active_mask = V_field > threshold
    active_area = int(np.sum(active_mask))
    

    efficacy_field = V_field * receptor_density
    total_efficacy = float(np.sum(efficacy_field))
    
    return {
        'peak_concentration': peak_conc,
        'mean_concentration': mean_conc,
        'active_area_pixels': active_area,
        'total_efficacy': total_efficacy,
        'active_fraction': active_area / V_field.size,
    }
