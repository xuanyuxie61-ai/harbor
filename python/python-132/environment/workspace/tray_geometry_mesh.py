
import numpy as np
from utils import clip_with_warning






def drectangle(p, x1, x2, y1, y2):
    p = np.asarray(p, dtype=float)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    dx = np.minimum(-x1 + p[:, 0], x2 - p[:, 0])
    dy = np.minimum(-y1 + p[:, 1], y2 - p[:, 1])
    d = -np.minimum(np.minimum(dx, dy), 0.0)

    inside = (p[:, 0] >= x1) & (p[:, 0] <= x2) & (p[:, 1] >= y1) & (p[:, 1] <= y2)
    d = np.where(inside, -np.minimum(dx, dy), np.maximum(-np.minimum(dx, dy), 0.0))
    return d






def reference_to_physical_q4(q4, n, rs):
    rs = np.asarray(rs, dtype=float)
    if rs.shape[1] != n:
        n = rs.shape[1]

    psi = np.zeros((4, n), dtype=float)
    psi[0, :] = (1.0 - rs[0, :]) * (1.0 - rs[1, :])
    psi[1, :] = rs[0, :] * (1.0 - rs[1, :])
    psi[2, :] = rs[0, :] * rs[1, :]
    psi[3, :] = (1.0 - rs[0, :]) * rs[1, :]

    xy = np.dot(q4, psi)
    return xy


def q4_jacobian(q4, rs):
    R, S = float(rs[0]), float(rs[1])
    dpsi_dR = np.array([-(1 - S), (1 - S), S, -S], dtype=float)
    dpsi_dS = np.array([-(1 - R), -R, R, (1 - R)], dtype=float)

    J = np.zeros((2, 2), dtype=float)
    J[0, 0] = np.dot(q4[0, :], dpsi_dR)
    J[0, 1] = np.dot(q4[0, :], dpsi_dS)
    J[1, 0] = np.dot(q4[1, :], dpsi_dR)
    J[1, 1] = np.dot(q4[1, :], dpsi_dS)

    detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
    return detJ, J






def generate_tray_mesh(tray_width, tray_height, nx, ny):
    if nx < 1:
        nx = 1
    if ny < 1:
        ny = 1

    x = np.linspace(0.0, tray_width, nx + 1)
    y = np.linspace(0.0, tray_height, ny + 1)
    xv, yv = np.meshgrid(x, y)
    nodes = np.column_stack((xv.ravel(), yv.ravel()))

    n_nodes_x = nx + 1
    n_nodes_y = ny + 1
    elements = []
    for j in range(ny):
        for i in range(nx):
            n1 = j * n_nodes_x + i
            n2 = j * n_nodes_x + i + 1
            n3 = (j + 1) * n_nodes_x + i + 1
            n4 = (j + 1) * n_nodes_x + i
            elements.append([n1, n2, n3, n4])
    elements = np.array(elements, dtype=int)


    dx = tray_width / nx
    dy = tray_height / ny
    areas = np.full(len(elements), dx * dy, dtype=float)

    return nodes, elements, areas


def compute_local_efficiency_on_mesh(nodes, elements, x_liq, y_vap, K_eq):
    nc = len(x_liq)
    n_elem = len(elements)
    E_local = np.zeros(n_elem, dtype=float)

    y_star = K_eq * x_liq
    y_star = np.clip(y_star, 0.0, 1.0)

    denom = y_star - y_vap

    for e in range(n_elem):
        effs = []
        for c in range(nc):
            d = denom[c]
            num = y_star[c] - y_vap[c]
            if abs(d) > 1e-12:
                effs.append(num / d)
            else:
                effs.append(0.0)
        E_local[e] = np.mean(effs)

    E_local = np.clip(E_local, 0.0, 1.0)
    return E_local


def mesh_average_efficiency(nodes, elements, areas, E_local):
    total_area = np.sum(areas)
    if total_area < 1e-15:
        return 0.0
    return float(np.sum(areas * E_local) / total_area)
