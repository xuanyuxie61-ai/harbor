
import numpy as np






def circle_segment_area_from_height(R, h):
    if R <= 0:
        raise ValueError("R > 0")
    if h <= 0.0:
        return 0.0
    if h >= 2.0 * R:
        return np.pi * R * R


    tmp = np.sqrt(max(0.0, R * R - (R - h) * (R - h))) / R
    tmp = min(1.0, max(-1.0, tmp))
    theta = 2.0 * np.arcsin(tmp)

    if h <= R:
        area = R * R * (theta - np.sin(theta)) / 2.0
    else:
        theta = 2.0 * np.pi - theta
        area = R * R * (theta - np.sin(theta)) / 2.0

    return area


def circle_segment_centroid_from_height(R, h):
    if R <= 0:
        raise ValueError("R > 0")
    if h <= 0.0:
        return 0.0
    if h >= 2.0 * R:
        return 0.0

    tmp = np.sqrt(max(0.0, R * R - (R - h) * (R - h))) / R
    tmp = min(1.0, max(-1.0, tmp))
    theta = 2.0 * np.arcsin(tmp)
    if h > R:
        theta = 2.0 * np.pi - theta

    area = R * R * (theta - np.sin(theta)) / 2.0
    if abs(area) < 1e-30:
        return 0.0
    d = 4.0 * R * (np.sin(theta / 2.0) ** 3) / (3.0 * (theta - np.sin(theta)))
    return d


def circle_segment_height_from_area(R, area):
    if area <= 0.0:
        return 0.0
    if area >= np.pi * R * R:
        return 2.0 * R

    h = R
    for _ in range(50):
        f = circle_segment_area_from_height(R, h) - area
        if abs(f) < 1e-12:
            break

        dh = max(1e-8, h * 1e-6)
        fp = (circle_segment_area_from_height(R, h + dh) - circle_segment_area_from_height(R, h - dh)) / (2 * dh)
        if abs(fp) < 1e-30:
            break
        h = h - f / fp
        h = max(0.0, min(2.0 * R, h))
    return h






def quadrature_on_curved_domain(func, x_range, z_range, arc_centers, arc_radii,
                                n_x=20, n_z=20):
    xmin, xmax = x_range
    zmin, zmax = z_range
    dx = (xmax - xmin) / n_x
    dz = (zmax - zmin) / n_z

    total = 0.0
    for i in range(n_x):
        x0 = xmin + (i + 0.5) * dx
        for j in range(n_z):
            z0 = zmin + (j + 0.5) * dz

            inside = True
            for (cx, cz), r in zip(arc_centers, arc_radii):
                dist = np.sqrt((x0 - cx) ** 2 + (z0 - cz) ** 2)
                if r < 0 and dist < abs(r):
                    inside = False
                    break
                if r > 0 and dist > r:
                    inside = False
                    break
            if inside:
                total += func(x0, z0) * dx * dz

    return total


def coastal_boundary_length(arc_centers, arc_radii, arc_angles):
    length = 0.0
    for r, ang in zip(arc_radii, arc_angles):
        length += abs(r) * ang
    return length






def local_coordinates_circle_segment(x, z, cx, cz, R, theta_start, theta_end):
    dx = x - cx
    dz = z - cz
    r_local = np.sqrt(dx ** 2 + dz ** 2)
    theta = np.arctan2(dz, dx)


    theta_norm = (theta - theta_start) % (2.0 * np.pi)
    arc_len = abs(R) * theta_norm
    normal_dist = r_local - abs(R)

    return arc_len, normal_dist


def project_velocity_to_boundary(u, w, boundary_normal):
    nx, nz = boundary_normal
    un = u * nx + w * nz
    ut = -u * nz + w * nx
    return un, ut
