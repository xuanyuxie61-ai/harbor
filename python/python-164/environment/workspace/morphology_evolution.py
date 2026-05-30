
import numpy as np


def box_counting_dimension(x_coords, y_coords, n_scales=12):
    if len(x_coords) < 2 or len(y_coords) < 2:
        return 1.0
    
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    
    L = max(x_max - x_min, y_max - y_min)
    if L < 1e-15:
        return 0.0
    
    points = np.column_stack((x_coords, y_coords))
    n_pts = len(points)
    
    N_list = []
    epsilon_list = []
    

    for scale in range(1, n_scales + 1):
        epsilon = L / scale
        if epsilon < 1e-15:
            break
        
        nx = max(1, int(np.ceil((x_max - x_min) / epsilon)))
        ny = max(1, int(np.ceil((y_max - y_min) / epsilon)))
        
        occupied = set()
        for pt in points:
            ix = min(int((pt[0] - x_min) / epsilon), nx - 1)
            iy = min(int((pt[1] - y_min) / epsilon), ny - 1)
            occupied.add((ix, iy))
        
        N = len(occupied)
        if N <= 1 or N >= n_pts:
            continue
        
        N_list.append(float(N))
        epsilon_list.append(float(epsilon))
    
    if len(N_list) < 3:
        return 1.0
    

    log_eps = np.log(epsilon_list)
    log_N = np.log(N_list)
    

    A = np.vstack([log_eps, np.ones(len(log_eps))]).T
    coeff, residuals, _, _ = np.linalg.lstsq(A, log_N, rcond=None)
    D_f = -coeff[0]
    


    D_f = float(np.clip(D_f, 0.0, 2.0))
    
    return D_f


def effective_surface_area_fractal(A0, L_scale, l0, D_f):
    if l0 <= 0 or L_scale <= 0 or A0 <= 0:
        return A0
    
    ratio = L_scale / l0
    if ratio <= 0:
        return A0
    
    A_eff = A0 * (ratio ** (D_f - 2.0))
    

    A_eff = np.clip(A_eff, A0 * 0.01, A0 * 100.0)
    
    return float(A_eff)


def ubvec_next_gray(t):
    t = np.array(t, dtype=int)
    n = len(t)
    
    if n <= 0:
        return t
    
    weight = np.sum(t)
    
    t_next = t.copy()
    
    if weight % 2 == 0:

        t_next[n - 1] = 1 - t_next[n - 1]
    else:

        flipped = False
        for i in range(n - 1, 0, -1):
            if t[i] == 1:
                t_next[i - 1] = 1 - t_next[i - 1]
                flipped = True
                break
        
        if not flipped:

            t_next[:] = 0
    
    return t_next


def enumerate_catalyst_surface_states(n_sites, max_states=1024):
    if n_sites <= 0:
        return np.zeros((1, 0))
    
    total = 2 ** n_sites
    num_states = min(total, max_states)
    
    states = np.zeros((num_states, n_sites), dtype=int)
    
    t = np.zeros(n_sites, dtype=int)
    states[0, :] = t
    
    for i in range(1, num_states):
        t = ubvec_next_gray(t)
        states[i, :] = t
    
    return states


def mandelbrot_like_escape_time(c_real, c_imag, max_iter=50, escape_radius=2.0):
    z_real = 0.0
    z_imag = 0.0
    
    for i in range(max_iter):
        zr2 = z_real * z_real
        zi2 = z_imag * z_imag
        
        if zr2 + zi2 > escape_radius * escape_radius:
            return i
        
        z_imag = 2.0 * z_real * z_imag + c_imag
        z_real = zr2 - zi2 + c_real
    
    return max_iter


def pore_network_connectivity_map(n_grid=64, max_iter=50):
    x_min, x_max = -2.0, 2.0
    y_min, y_max = -2.0, 2.0
    
    x = np.linspace(x_min, x_max, n_grid)
    y = np.linspace(y_min, y_max, n_grid)
    
    connectivity = np.zeros((n_grid, n_grid))
    
    for i in range(n_grid):
        for j in range(n_grid):
            c_real = x[i]
            c_imag = y[j]
            escape = mandelbrot_like_escape_time(c_real, c_imag, max_iter)


            connectivity[j, i] = 1.0 - float(escape) / max_iter
    
    return connectivity, (x_min, x_max), (y_min, y_max)


def morphology_degradation_index(D_f_initial, D_f_current, connectivity_drop):
    if D_f_initial <= 0:
        return 0.0
    
    w1, w2 = 0.6, 0.4
    
    frac_loss = max(0.0, D_f_initial - D_f_current) / D_f_initial
    conn_loss = np.clip(connectivity_drop, 0.0, 1.0)
    
    mdi = w1 * frac_loss + w2 * conn_loss
    return float(np.clip(mdi, 0.0, 1.0))


if __name__ == "__main__":

    theta = np.linspace(0, 2*np.pi, 100)
    x = np.cos(theta)
    y = np.sin(theta)
    D_f = box_counting_dimension(x, y)
    print(f"圆的分形维数: {D_f:.4f} (理论值: 1.0)")
    
    states = enumerate_catalyst_surface_states(4, max_states=16)
    print(f"枚举 {len(states)} 个表面状态 (4位点)")
    
    conn, xr, yr = pore_network_connectivity_map(n_grid=32, max_iter=30)
    print(f"孔隙连通性均值: {np.mean(conn):.4f}")
