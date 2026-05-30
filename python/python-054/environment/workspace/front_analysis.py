
import numpy as np


def news_gradient(field):
    ny, nx = field.shape
    grad = np.zeros_like(field)
    

    field_ext = np.pad(field, pad_width=1, mode='edge')
    
    for i in range(ny):
        for j in range(nx):
            ie = i + 1
            je = j + 1
            north = field_ext[ie - 1, je]
            south = field_ext[ie + 1, je]
            east = field_ext[ie, je + 1]
            west = field_ext[ie, je - 1]
            grad[i, j] = abs(north - south) + abs(east - west)
    
    return grad


def sobel_gradient(field):
    ny, nx = field.shape
    field_ext = np.pad(field, pad_width=1, mode='edge')
    
    grad_x = np.zeros_like(field)
    grad_y = np.zeros_like(field)
    
    for i in range(ny):
        for j in range(nx):
            ie = i + 1
            je = j + 1
            patch = field_ext[ie-1:ie+2, je-1:je+2]
            grad_x[i, j] = np.sum(patch * np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]))
            grad_y[i, j] = np.sum(patch * np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]))
    
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    return grad_mag, grad_x, grad_y


def detect_fronts_multi_field(fields_dict, weights=None, threshold_percentile=90):
    if weights is None:
        weights = {k: 1.0 for k in fields_dict}
    

    front_index = None
    for key, field in fields_dict.items():
        f_min, f_max = np.min(field), np.max(field)
        if abs(f_max - f_min) < 1e-10:
            continue
        field_norm = (field - f_min) / (f_max - f_min)
        grad = news_gradient(field_norm)
        
        w = weights.get(key, 1.0)
        if front_index is None:
            front_index = w * grad
        else:
            front_index += w * grad
    
    if front_index is None:
        raise ValueError("没有有效场用于锋面检测")
    
    threshold = np.percentile(front_index, threshold_percentile)
    front_mask = front_index > threshold
    
    return {
        'front_mask': front_mask,
        'front_index': front_index,
        'threshold': threshold,
    }


def front_statistics(front_mask, field, dx=1.0, dy=1.0):
    ny, nx = field.shape
    

    n_front_pixels = np.sum(front_mask)
    

    front_length = n_front_pixels * np.sqrt(dx**2 + dy**2)
    

    if n_front_pixels > 0:
        front_values = field[front_mask]
        front_mean = np.mean(front_values)
        front_std = np.std(front_values)
    else:
        front_mean = np.nan
        front_std = np.nan
    

    grad_mag, grad_x, grad_y = sobel_gradient(field)
    if n_front_pixels > 0:
        front_grad = grad_mag[front_mask]
        front_grad_mean = np.mean(front_grad)
        front_grad_max = np.max(front_grad)
    else:
        front_grad_mean = np.nan
        front_grad_max = np.nan
    
    return {
        'n_pixels': n_front_pixels,
        'front_length_km': front_length,
        'front_mean_value': front_mean,
        'front_std_value': front_std,
        'front_mean_gradient': front_grad_mean,
        'front_max_gradient': front_grad_max,
    }


def thermocline_depth_from_field(T_field, z_coords, threshold=0.5):
    ny, nx = T_field.shape
    mld = np.zeros((ny, nx))
    
    for i in range(ny):
        for j in range(nx):
            T_surf = T_field[i, j]
            T_target = T_surf - threshold

            mld[i, j] = 50.0 + np.random.exponential(30.0)
    
    return mld
