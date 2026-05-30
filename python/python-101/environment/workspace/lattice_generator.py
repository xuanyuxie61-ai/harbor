
import numpy as np


def ellipse_grid(n, rx, ry, cx, cy):
    if n < 1:
        raise ValueError("n 必须 >= 1")
    if rx <= 0 or ry <= 0:
        raise ValueError("半轴长度必须为正")
    
    if rx < ry:
        h = 2.0 * rx / (2.0 * n + 1.0)
        ni = n
        nj = int(np.ceil(ry / rx) * n)
    else:
        h = 2.0 * ry / (2.0 * n + 1.0)
        nj = n
        ni = int(np.ceil(rx / ry) * n)
    
    points = []
    for j in range(nj + 1):
        i = 0
        x = cx
        y = cy + j * h
        points.append([x, y])
        if j > 0:
            points.append([x, 2 * cy - y])
        while True:
            i += 1
            x = cx + i * h
            if ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 > 1.0:
                break
            points.append([x, y])
            points.append([2 * cx - x, y])
            if j > 0:
                points.append([x, 2 * cy - y])
                points.append([2 * cx - x, 2 * cy - y])
    
    xy = np.array(points)

    xy = np.unique(np.round(xy, 12), axis=0)
    return xy


def ellipsoid_grid(n, rx, ry, rz, cx, cy, cz):
    if n < 1:
        raise ValueError("n 必须 >= 1")
    if rx <= 0 or ry <= 0 or rz <= 0:
        raise ValueError("半轴长度必须为正")
    
    r = np.array([rx, ry, rz])
    c = np.array([cx, cy, cz])
    rmin = np.min(r)
    h = 2.0 * rmin / (2.0 * n + 1.0)
    ni = int(np.ceil(rx / rmin) * n)
    nj = int(np.ceil(ry / rmin) * n)
    nk = int(np.ceil(rz / rmin) * n)
    
    points = []
    for k in range(nk + 1):
        z = cz + k * h
        for j in range(nj + 1):
            y = cy + j * h
            for i in range(ni + 1):
                x = cx + i * h
                if ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 + ((z - cz) / rz) ** 2 > 1.0:
                    break
                p = np.array([[x, y, z]])
                np_ = 1
                if i > 0:
                    q = p.copy()
                    q[:, 0] = 2 * cx - q[:, 0]
                    p = np.vstack([p, q])
                    np_ *= 2
                if j > 0:
                    q = p.copy()
                    q[:, 1] = 2 * cy - q[:, 1]
                    p = np.vstack([p, q])
                    np_ *= 2
                if k > 0:
                    q = p.copy()
                    q[:, 2] = 2 * cz - q[:, 2]
                    p = np.vstack([p, q])
                    np_ *= 2
                points.extend(p.tolist())
    
    xyz = np.array(points)
    xyz = np.unique(np.round(xyz, 12), axis=0)
    return xyz


def magic_matrix(n):
    if n % 2 != 1 or n < 1:
        raise ValueError("n 必须为正奇数")
    
    A = np.zeros((n, n), dtype=int)
    k = 1
    i = 0
    j = n // 2
    A[i, j] = k
    
    while k < n * n:
        k += 1
        im1 = (i - 1) % n
        jp1 = (j + 1) % n
        if A[im1, jp1] != 0:
            im1 = (i + 1) % n
            jp1 = j
        A[im1, jp1] = k
        i, j = im1, jp1
    
    return A


def square_photonic_crystal(nx, ny, a, r_hole, eps_bg, eps_hole):
    if nx < 3 or ny < 3:
        raise ValueError("网格分辨率至少为 3×3")
    if a <= 0 or r_hole < 0:
        raise ValueError("晶格常数必须为正，孔半径必须非负")
    if r_hole > a / 2.0:
        raise ValueError("孔半径不能超过 a/2（避免孔重叠）")
    
    dx = a / nx
    dy = a / ny
    x = np.linspace(0, a - dx, nx)
    y = np.linspace(0, a - dy, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    

    dx_p = np.minimum(np.mod(X, a), a - np.mod(X, a))
    dy_p = np.minimum(np.mod(Y, a), a - np.mod(Y, a))
    dist = np.sqrt(dx_p ** 2 + dy_p ** 2)
    
    eps_r = np.where(dist < r_hole, eps_hole, eps_bg)
    return eps_r, x, y


def triangular_photonic_crystal(nx, ny, a, r_hole, eps_bg, eps_hole):
    if nx < 3 or ny < 3:
        raise ValueError("网格分辨率至少为 3×3")
    if a <= 0 or r_hole < 0 or r_hole > a / np.sqrt(3):
        raise ValueError("参数超出物理允许范围")
    
    a1 = np.array([a, 0.0])
    a2 = np.array([a * 0.5, a * np.sqrt(3.0) / 2.0])
    

    i_idx = np.arange(nx)
    j_idx = np.arange(ny)
    I, J = np.meshgrid(i_idx, j_idx, indexing='ij')
    
    X = (I / nx) * a1[0] + (J / ny) * a2[0]
    Y = (I / nx) * a1[1] + (J / ny) * a2[1]
    


    eps_r = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):

            min_dist = float('inf')
            for m in range(-1, 2):
                for n in range(-1, 2):
                    px = m * a1[0] + n * a2[0]
                    py = m * a1[1] + n * a2[1]
                    dx_ = X[i, j] - px
                    dy_ = Y[i, j] - py
                    d = np.sqrt(dx_ ** 2 + dy_ ** 2)
                    if d < min_dist:
                        min_dist = d
            eps_r[i, j] = eps_hole if min_dist < r_hole else eps_bg
    
    return eps_r, X, Y


def quasiperiodic_photonic_crystal(n, a_avg, r_hole, eps_bg, eps_hole, magic_order=5):
    if n < 3:
        raise ValueError("网格点数至少为 3")
    if magic_order % 2 != 1:
        raise ValueError("幻方阶数必须为奇数")
    
    M = magic_matrix(magic_order)
    M_norm = (M - np.min(M)) / (np.max(M) - np.min(M) + 1e-12)
    
    L = n * a_avg
    x = np.linspace(0, L, n)
    y = np.linspace(0, L, n)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    eps_r = np.full((n, n), eps_bg)
    

    n_holes = magic_order * magic_order
    for idx in range(1, n_holes + 1):
        pos = np.argwhere(M == idx)[0]
        i, j = pos
        cx = (i + 0.5 * M_norm[i, j]) * (L / magic_order)
        cy = (j + 0.5 * M_norm[j, i]) * (L / magic_order)
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        mask = dist < r_hole
        eps_r[mask] = eps_hole
    
    return eps_r, x, y


def woodpile_photonic_crystal(nx, ny, nz, a, r_rod, eps_bg, eps_rod):
    if nx < 3 or ny < 3 or nz < 3:
        raise ValueError("网格分辨率至少为 3×3×3")
    if a <= 0 or r_rod < 0 or r_rod > a / 2:
        raise ValueError("参数超出物理允许范围")
    
    dx = a / nx
    dy = a / ny
    dz = a / nz
    x = np.linspace(0, a - dx, nx)
    y = np.linspace(0, a - dy, ny)
    z = np.linspace(0, a - dz, nz)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    eps_r = np.full((nx, ny, nz), eps_bg)
    
    for layer in range(nz):
        z_layer = z[layer]
        layer_idx = int(np.round(z_layer / (a / 4))) % 4
        
        if layer_idx in [0, 2]:

            y_center = a / 4 if layer_idx == 0 else 3 * a / 4
            for rod in range(2):
                yc = y_center + rod * a / 2
                dist = np.sqrt((Y[:, :, layer] - yc) ** 2)
                mask = dist < r_rod
                eps_r[:, :, layer][mask] = eps_rod
        else:

            x_center = a / 4 if layer_idx == 1 else 3 * a / 4
            for rod in range(2):
                xc = x_center + rod * a / 2
                dist = np.sqrt((X[:, :, layer] - xc) ** 2)
                mask = dist < r_rod
                eps_r[:, :, layer][mask] = eps_rod
    
    return eps_r, x, y, z


def inverse_opal_structure(n, a, r_sphere, eps_bg, eps_sphere):
    if n < 3:
        raise ValueError("网格点数至少为 3")
    if a <= 0 or r_sphere < 0:
        raise ValueError("参数必须为正")
    
    dx = a / n
    x = np.linspace(0, a - dx, n)
    y = np.linspace(0, a - dx, n)
    z = np.linspace(0, a - dx, n)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    eps_r = np.full((n, n, n), eps_bg)
    

    a1 = np.array([a, 0, 0])
    a2 = np.array([a / 2, a * np.sqrt(3) / 2, 0])
    a3 = np.array([a / 2, a * np.sqrt(3) / 6, a * np.sqrt(6) / 3])
    

    for i in range(n):
        for j in range(n):
            for k in range(n):
                min_dist = float('inf')
                for ii in range(-1, 2):
                    for jj in range(-1, 2):
                        for kk in range(-1, 2):
                            px = ii * a1[0] + jj * a2[0] + kk * a3[0]
                            py = ii * a1[1] + jj * a2[1] + kk * a3[1]
                            pz = ii * a1[2] + jj * a2[2] + kk * a3[2]
                            d = np.sqrt((X[i, j, k] - px) ** 2 +
                                        (Y[i, j, k] - py) ** 2 +
                                        (Z[i, j, k] - pz) ** 2)
                            if d < min_dist:
                                min_dist = d
                if min_dist < r_sphere:
                    eps_r[i, j, k] = eps_sphere
    
    return eps_r, x, y, z
