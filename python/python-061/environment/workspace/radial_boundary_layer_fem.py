
import numpy as np


def radial_boundary_layer_fem(n_nodes, r_inner, r_outer, p_gradient_func,
                               epsilon=50.0, f=5e-5, rho=1.225):
    if n_nodes < 3:
        raise ValueError("节点数 n_nodes 必须至少为 3")
    if n_nodes % 2 == 0:
        raise ValueError("节点数 n_nodes 必须为奇数")
    

    r = np.linspace(r_inner, r_outer, n_nodes) * 1000.0
    n_elements = (n_nodes - 1) // 2
    

    abscissa = np.array([-0.7745966692414834, 0.0, 0.7745966692414834])
    weight = np.array([0.5555555555555556, 0.8888888888888889, 0.5555555555555556])
    quad_num = 3
    


    A = np.zeros((2 * n_nodes, 2 * n_nodes))
    b = np.zeros(2 * n_nodes)
    
    def shape_functions(xi):
        phi_l = 0.5 * xi * (xi - 1.0)
        phi_m = 1.0 - xi**2
        phi_r = 0.5 * xi * (xi + 1.0)
        
        dphi_l = xi - 0.5
        dphi_m = -2.0 * xi
        dphi_r = xi + 0.5
        
        return (phi_l, phi_m, phi_r), (dphi_l, dphi_m, dphi_r)
    

    for e in range(n_elements):
        l = 2 * e
        m = 2 * e + 1
        r_r = 2 * e + 2
        
        xl = r[l]
        xm = r[m]
        xr = r[r_r]
        h_elem = xr - xl
        
        for q in range(quad_num):

            xi = abscissa[q]
            rq = 0.5 * ((1.0 - xi) * xl + (1.0 + xi) * xr)
            wq = weight[q] * h_elem / 2.0
            
            (phi_l, phi_m, phi_r), (dphi_l, dphi_m, dphi_r) = shape_functions(xi)
            

            dr_dxi = h_elem / 2.0
            dphi_dx_l = dphi_l / dr_dxi
            dphi_dx_m = dphi_m / dr_dxi
            dphi_dx_r = dphi_r / dr_dxi
            

            pgf = p_gradient_func(rq) / rho
            




            


            v_guess = rankine_vortex_v(rq, r_outer * 1000.0)
            dv_guess = 0.0
            

            for i_idx, (i, phi_i, dphi_i) in enumerate([(l, phi_l, dphi_dx_l),
                                                         (m, phi_m, dphi_dx_m),
                                                         (r_r, phi_r, dphi_dx_r)]):
                row_u = i
                
                for j_idx, (j, phi_j, dphi_j) in enumerate([(l, phi_l, dphi_dx_l),
                                                              (m, phi_m, dphi_dx_m),
                                                              (r_r, phi_r, dphi_dx_r)]):
                    col_u = j
                    

                    diff = epsilon * (dphi_i * dphi_j + phi_i * phi_j / (rq**2))
                    A[row_u, col_u] += wq * diff
                    

                    col_v = j + n_nodes
                    A[row_u, col_v] += wq * (-f * phi_i * phi_j)
                

                b[row_u] += wq * phi_i * pgf
    



    

    A[0, :] = 0.0
    A[0, 0] = 1.0
    b[0] = 0.0
    

    A[n_nodes - 1, :] = 0.0
    A[n_nodes - 1, n_nodes - 1] = 1.0
    b[n_nodes - 1] = 0.0
    


    row_v_outer = 2 * n_nodes - 1
    A[row_v_outer, :] = 0.0
    A[row_v_outer, row_v_outer] = 1.0
    b[row_v_outer] = f * r_outer * 1000.0 / 2.0
    

    try:
        sol = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:

        sol = np.linalg.lstsq(A, b, rcond=None)[0]
    
    u = sol[:n_nodes]
    v = sol[n_nodes:]
    

    r_km = r / 1000.0
    
    return r_km, u, v


def rankine_vortex_v(r, r_max, v_max=50.0):
    r = np.atleast_1d(r)
    v = np.zeros_like(r, dtype=float)
    
    mask_inner = r <= r_max
    mask_outer = r > r_max
    
    v[mask_inner] = v_max * r[mask_inner] / r_max
    v[mask_outer] = v_max * r_max / r[mask_outer]
    
    return v


def compute_boundary_layer_inflow_profile(r_min=10.0, r_max=300.0,
                                           p_drop=50.0, n_nodes=101):

    r_max_m = r_max * 1000.0
    B = 1.8
    p_env = 101000.0
    p_c = p_env - p_drop * 100.0
    
    def p_gradient(r):
        r_km = r / 1000.0
        if r_km < 1.0:
            r_km = 1.0

        term = (r_max / r_km)**B
        dpdr = B * (p_env - p_c) * term * np.exp(-term) / (r_km * 1000.0)
        return dpdr
    
    r, u, v = radial_boundary_layer_fem(n_nodes, r_min, r_max, p_gradient)
    



    H_bl = 1000.0
    w = np.zeros_like(u)
    dr = np.diff(r * 1000.0)
    
    for i in range(len(r) - 1):
        r_mid = 0.5 * (r[i] + r[i + 1]) * 1000.0
        du = (r[i + 1] * 1000.0 * u[i + 1] - r[i] * 1000.0 * u[i])
        divergence = du / (r_mid * dr[i])
        w[i] = -H_bl * divergence
    
    w[-1] = w[-2]
    
    return r, u, v, w
