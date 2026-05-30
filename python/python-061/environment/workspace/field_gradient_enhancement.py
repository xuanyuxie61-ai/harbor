
import numpy as np


def local_average_1d(field, boundary='symmetric'):
    n = len(field)
    avg = np.zeros(n)
    

    avg[1:-1] = (field[:-2] + field[1:-1] + field[2:]) / 3.0
    

    if boundary == 'symmetric':
        avg[0] = (2.0 * field[0] + field[1]) / 3.0
        avg[-1] = (2.0 * field[-1] + field[-2]) / 3.0
    elif boundary == 'periodic':
        avg[0] = (field[-1] + field[0] + field[1]) / 3.0
        avg[-1] = (field[-2] + field[-1] + field[0]) / 3.0
    else:
        avg[0] = field[0]
        avg[-1] = field[-1]
    
    return avg


def gradient_enhancement_1d(field, sharpness=1.5, boundary='symmetric'):
    avg = local_average_1d(field, boundary)
    enhanced = sharpness * field + (1.0 - sharpness) * avg
    return enhanced


def compute_gradient_1d(x, field):
    n = len(field)
    grad = np.zeros(n)
    
    dx_forward = np.diff(x)
    

    grad[1:-1] = (field[2:] - field[:-2]) / (x[2:] - x[:-2])
    

    if n > 1:
        grad[0] = (field[1] - field[0]) / dx_forward[0]
        grad[-1] = (field[-1] - field[-2]) / dx_forward[-1]
    
    return grad


def compute_laplacian_1d(x, field):
    n = len(field)
    lap = np.zeros(n)
    
    dx = np.diff(x)
    
    for i in range(1, n - 1):
        dx_forward = x[i + 1] - x[i]
        dx_backward = x[i] - x[i - 1]
        dx_total = x[i + 1] - x[i - 1]
        
        if dx_forward > 0 and dx_backward > 0 and dx_total > 0:
            lap[i] = 2.0 * ((field[i + 1] - field[i]) / dx_forward
                            - (field[i] - field[i - 1]) / dx_backward) / dx_total
    
    return lap


def anisotropic_diffusion_1d(field, x, n_iter=10, dt=0.1, lambda_param=1.0,
                              diffusion_type='exponential'):
    phi = field.copy()
    n = len(phi)
    
    for _ in range(n_iter):
        phi_new = phi.copy()
        
        for i in range(1, n - 1):
            dx_forward = x[i + 1] - x[i]
            dx_backward = x[i] - x[i - 1]
            
            if dx_forward <= 0 or dx_backward <= 0:
                continue
            

            grad_forward = (phi[i + 1] - phi[i]) / dx_forward

            grad_backward = (phi[i] - phi[i - 1]) / dx_backward
            

            if diffusion_type == 'exponential':
                c_forward = np.exp(-(abs(grad_forward) / lambda_param)**2)
                c_backward = np.exp(-(abs(grad_backward) / lambda_param)**2)
            else:
                c_forward = 1.0 / (1.0 + (grad_forward / lambda_param)**2)
                c_backward = 1.0 / (1.0 + (grad_backward / lambda_param)**2)
            

            flux = c_forward * grad_forward - c_backward * grad_backward

            dx_avg = 0.5 * (dx_forward + dx_backward)
            phi_new[i] += dt * flux / dx_avg
        
        phi = phi_new
    
    return phi


def detect_fronts_1d(x, field, gradient_threshold=0.5, laplacian_threshold=0.1):
    grad = compute_gradient_1d(x, field)
    lap = compute_laplacian_1d(x, field)
    
    front_indices = []
    front_strength = []
    
    n = len(field)
    for i in range(1, n - 1):

        if abs(grad[i]) < gradient_threshold:
            continue
        

        if lap[i - 1] * lap[i + 1] < 0 or abs(lap[i]) < laplacian_threshold:
            front_indices.append(i)
            front_strength.append(abs(grad[i]))
    
    return front_indices, front_strength


def apply_spatial_filter_pipeline(field, x, enhance=True, smooth=True, detect=True):
    result = field.copy()
    info = {}
    
    if enhance:
        result = gradient_enhancement_1d(result, sharpness=1.3)
        info['enhanced'] = True
    
    if smooth:
        result = anisotropic_diffusion_1d(result, x, n_iter=5, lambda_param=2.0)
        info['smoothed'] = True
    
    if detect:
        fronts, strength = detect_fronts_1d(x, result)
        info['fronts'] = fronts
        info['front_strength'] = strength
        info['n_fronts'] = len(fronts)
    
    return result, info
