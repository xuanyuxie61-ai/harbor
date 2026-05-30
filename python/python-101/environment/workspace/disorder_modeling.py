
import numpy as np






def chebyshev2_rejection_sample(n):
    if n < 1:
        raise ValueError("采样数必须 >= 1")
    
    pdfmax = 2.0 / np.pi
    samples = np.zeros(n)
    trials = 0
    i = 0
    
    while i < n:
        trials += 1
        x = -1.0 + 2.0 * np.random.rand()
        y = pdfmax * np.random.rand()
        z = (2.0 / np.pi) * np.sqrt(max(1.0 - x ** 2, 0.0))
        
        if y <= z:
            samples[i] = x
            i += 1
    
    return samples, trials


def cvt_1d_rejection_sample(n):
    if n < 1:
        raise ValueError("采样数必须 >= 1")
    
    pdfmax = 1.0 / np.sqrt(np.pi)
    samples = np.zeros(n)
    i = 0
    
    while i < n:
        x = np.random.rand()
        y = pdfmax * np.random.rand()
        z = 1.0 / np.sqrt(np.pi) / (np.sqrt(max(1.0 - x ** 2, 1e-12)) ** (1.0 / 6.0))
        
        if y <= z:
            samples[i] = x
            i += 1
    
    return samples


def gaussian_rejection_sample(n, mu, sigma):
    if sigma <= 0:
        raise ValueError("标准差必须为正")
    

    samples = np.zeros(n)
    for i in range(0, n, 2):
        u1 = np.random.rand()
        u2 = np.random.rand()
        r = sigma * np.sqrt(-2.0 * np.log(max(u1, 1e-18)))
        theta = 2.0 * np.pi * u2
        samples[i] = mu + r * np.cos(theta)
        if i + 1 < n:
            samples[i + 1] = mu + r * np.sin(theta)
    
    return samples






def positional_disorder(eps_r, x, y, a, disorder_strength, n_disorder_samples=1):
    if disorder_strength < 0:
        raise ValueError("无序强度必须非负")
    if disorder_strength < 1e-12:
        return [eps_r.copy()]
    
    nx, ny = eps_r.shape
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    eps_disordered = []
    
    for _ in range(n_disorder_samples):

        eps_threshold = (np.max(eps_r) + np.min(eps_r)) / 2.0
        

        dx_shift = gaussian_rejection_sample(nx * ny, 0.0, disorder_strength * a)
        dy_shift = gaussian_rejection_sample(nx * ny, 0.0, disorder_strength * a)
        
        dx_field = dx_shift[:nx * ny].reshape(nx, ny)
        dy_field = dy_shift[:nx * ny].reshape(nx, ny)
        

        X_shifted = X + dx_field
        Y_shifted = Y + dy_field
        

        eps_new = np.full_like(eps_r, np.max(eps_r))
        

        hole_centers = []
        for i in range(nx):
            for j in range(ny):
                if eps_r[i, j] < eps_threshold:
                    hole_centers.append([X[i, j], Y[i, j]])
        
        if len(hole_centers) == 0:
            eps_disordered.append(eps_r.copy())
            continue
        

        r_hole_est = a * 0.3
        

        for hc in hole_centers:
            dx = np.random.normal(0, disorder_strength * a)
            dy = np.random.normal(0, disorder_strength * a)
            hc_new = [hc[0] + dx, hc[1] + dy]
            dist = np.sqrt((X - hc_new[0]) ** 2 + (Y - hc_new[1]) ** 2)
            eps_new[dist < r_hole_est] = np.min(eps_r)
        
        eps_disordered.append(eps_new)
    
    return eps_disordered


def size_disorder(eps_r, x, y, a, r_hole, relative_variation, n_samples=1):
    if relative_variation < 0 or r_hole <= 0:
        raise ValueError("参数超出允许范围")
    
    nx, ny = eps_r.shape
    X, Y = np.meshgrid(x, y, indexing='ij')
    eps_threshold = (np.max(eps_r) + np.min(eps_r)) / 2.0
    

    hole_centers = []
    for i in range(nx):
        for j in range(ny):
            if eps_r[i, j] < eps_threshold:
                hole_centers.append([X[i, j], Y[i, j]])
    
    eps_samples = []
    for _ in range(n_samples):
        eps_new = np.full_like(eps_r, np.max(eps_r))
        
        for hc in hole_centers:

            delta = np.random.normal(0, relative_variation)
            delta = max(delta, -0.9)
            r_new = r_hole * (1.0 + delta)
            r_new = max(r_new, 1e-9)
            
            dist = np.sqrt((X - hc[0]) ** 2 + (Y - hc[1]) ** 2)
            eps_new[dist < r_new] = np.min(eps_r)
        
        eps_samples.append(eps_new)
    
    return eps_samples


def dielectric_disorder(eps_r, correlation_length, sigma_eps, n_samples=1):
    if sigma_eps < 0 or correlation_length < 0:
        raise ValueError("参数必须非负")
    
    nx, ny = eps_r.shape
    eps_samples = []
    
    for _ in range(n_samples):

        white_noise = np.random.randn(nx, ny)
        
        if correlation_length > 0.5:

            from scipy.ndimage import gaussian_filter
            kernel_sigma = correlation_length
            correlated_noise = gaussian_filter(white_noise, sigma=kernel_sigma)

            current_std = np.std(correlated_noise)
            if current_std > 1e-12:
                correlated_noise *= sigma_eps / current_std
        else:
            correlated_noise = white_noise * sigma_eps
        
        eps_new = eps_r + correlated_noise

        eps_new = np.maximum(eps_new, 1.0)
        
        eps_samples.append(eps_new)
    
    return eps_samples






def defect_histogram_sampling(nx, ny, defect_density, defect_size_pdf, n_defects):
    if np.any(defect_density < 0):
        raise ValueError("缺陷密度必须非负")
    

    joint_pdf = defect_density * defect_size_pdf
    joint_pdf = np.maximum(joint_pdf, 0.0)
    

    pdf_flat = joint_pdf.flatten()
    if np.sum(pdf_flat) < 1e-18:
        return []
    
    pdf_flat /= np.sum(pdf_flat)
    cdf = np.cumsum(pdf_flat)
    
    defects = []
    for _ in range(n_defects):
        u = np.random.rand()
        idx = np.searchsorted(cdf, u)
        idx = min(idx, nx * ny - 1)
        
        i = idx % nx
        j = idx // nx
        

        local_size = np.random.exponential(scale=defect_size_pdf[i, j])
        
        defects.append({
            'x_index': i,
            'y_index': j,
            'size': max(local_size, 1e-9)
        })
    
    return defects
