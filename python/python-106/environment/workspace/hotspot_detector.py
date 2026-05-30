
import numpy as np


def find_connected_components_2d(field, threshold):
    m, n = field.shape
    labels = np.zeros((m, n), dtype=int)
    component_index = 0

    for i2 in range(m):
        for j2 in range(n):
            if abs(field[i2, j2]) > threshold and labels[i2, j2] == 0:
                component_index += 1
                plist = [(i2, j2)]
                while plist:
                    i, j = plist.pop()
                    if labels[i, j] != 0:
                        continue
                    labels[i, j] = component_index

                    if i > 0 and abs(field[i - 1, j]) > threshold and labels[i - 1, j] == 0:
                        plist.append((i - 1, j))
                    if i + 1 < m and abs(field[i + 1, j]) > threshold and labels[i + 1, j] == 0:
                        plist.append((i + 1, j))
                    if j > 0 and abs(field[i, j - 1]) > threshold and labels[i, j - 1] == 0:
                        plist.append((i, j - 1))
                    if j + 1 < n and abs(field[i, j + 1]) > threshold and labels[i, j + 1] == 0:
                        plist.append((i, j + 1))

    return labels, component_index


def find_connected_components_3d(field, threshold):
    m, n, p = field.shape
    labels = np.zeros((m, n, p), dtype=int)
    component_index = 0

    for i2 in range(m):
        for j2 in range(n):
            for k2 in range(p):
                if abs(field[i2, j2, k2]) > threshold and labels[i2, j2, k2] == 0:
                    component_index += 1
                    plist = [(i2, j2, k2)]
                    while plist:
                        i, j, k = plist.pop()
                        if labels[i, j, k] != 0:
                            continue
                        labels[i, j, k] = component_index
                        if i > 0 and abs(field[i - 1, j, k]) > threshold and labels[i - 1, j, k] == 0:
                            plist.append((i - 1, j, k))
                        if i + 1 < m and abs(field[i + 1, j, k]) > threshold and labels[i + 1, j, k] == 0:
                            plist.append((i + 1, j, k))
                        if j > 0 and abs(field[i, j - 1, k]) > threshold and labels[i, j - 1, k] == 0:
                            plist.append((i, j - 1, k))
                        if j + 1 < n and abs(field[i, j + 1, k]) > threshold and labels[i, j + 1, k] == 0:
                            plist.append((i, j + 1, k))
                        if k > 0 and abs(field[i, j, k - 1]) > threshold and labels[i, j, k - 1] == 0:
                            plist.append((i, j, k - 1))
                        if k + 1 < p and abs(field[i, j, k + 1]) > threshold and labels[i, j, k + 1] == 0:
                            plist.append((i, j, k + 1))

    return labels, component_index


def polygon_contains_point(polygon, q):
    n = polygon.shape[0]
    inside = False
    x1, y1 = polygon[n - 1]
    for i in range(n):
        x2, y2 = polygon[i]
        if ((y1 < q[1] <= y2) or (q[1] <= y1 < y2)):
            if (q[0] - x1) - (q[1] - y1) * (x2 - x1) / (y2 - y1) <= 0.0:
                inside = not inside
        x1, y1 = x2, y2


    x1, y1 = polygon[n - 1]
    for i in range(n):
        x2, y2 = polygon[i]
        cross = (q[1] - y1) * (x2 - x1) - (y2 - y1) * (q[0] - x1)
        if abs(cross) < 1e-12:
            dot = (q[0] - x1) * (q[0] - x2) + (q[1] - y1) * (q[1] - y2)
            if dot <= 1e-12:
                return True
        x1, y1 = x2, y2

    return inside


def extract_hotspot_polygons_2d(field, dx, dy, threshold_factor=5.0):
    intensity = np.abs(field) ** 2
    threshold = threshold_factor * np.mean(intensity)
    labels, num_comp = find_connected_components_2d(intensity, threshold)

    polygons = []
    intensities = []
    m, n = field.shape

    for cid in range(1, num_comp + 1):
        mask = (labels == cid)
        indices = np.argwhere(mask)
        if len(indices) < 3:
            continue



        centroid = np.mean(indices, axis=0)
        angles = np.arctan2(indices[:, 1] - centroid[1], indices[:, 0] - centroid[0])
        order = np.argsort(angles)
        boundary = indices[order]


        phys = boundary * np.array([dx, dy])
        polygons.append(phys)
        intensities.append(float(np.mean(intensity[mask])))

    return polygons, intensities


def hot_carrier_generation_rate(field, omega, eps_metal, dx, dy, dz=1.0):
    hbar = 1.054571817e-34
    e_charge = 1.602176634e-19
    if omega <= 0:
        raise ValueError("omega must be positive.")
    prefactor = np.pi * (e_charge ** 2) / hbar
    G = prefactor * (np.abs(field) ** 2) * np.imag(eps_metal) / omega

    G = np.maximum(G, 0.0)
    return G
