
import numpy as np


def tetrahedron_signed_volume(nodes, element):
    x1, x2, x3, x4 = nodes[element]
    J = np.column_stack([x2 - x1, x3 - x1, x4 - x1])
    return np.linalg.det(J) / 6.0


def orient_elements(mesh, tol=1e-12):
    neg_count = 0
    zero_count = 0

    for e in range(mesh.n_elements):
        vol = tetrahedron_signed_volume(mesh.nodes, mesh.elements[e])

        if vol < -tol:

            mesh.elements[e][1], mesh.elements[e][2] = mesh.elements[e][2], mesh.elements[e][1]
            neg_count += 1


            vol_new = tetrahedron_signed_volume(mesh.nodes, mesh.elements[e])
            if vol_new < -tol:
                raise RuntimeError(f"Orientation fix failed for element {e}")

        elif abs(vol) <= tol:
            zero_count += 1

    return neg_count, zero_count


def compute_mesh_quality_metrics(mesh):
    volumes = np.zeros(mesh.n_elements)
    qualities = np.zeros(mesh.n_elements)

    for e in range(mesh.n_elements):
        idx = mesh.elements[e]
        x = mesh.nodes[idx]
        vol = tetrahedron_signed_volume(mesh.nodes, idx)
        volumes[e] = abs(vol)


        edge_sq_sum = 0.0
        pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        for i, j in pairs:
            edge_sq_sum += np.sum((x[i] - x[j]) ** 2)

        if edge_sq_sum > 1e-15:
            qualities[e] = 216.0 * (abs(vol) ** 2) / (edge_sq_sum ** 3)
        else:
            qualities[e] = 0.0

    return {
        'min_volume': float(np.min(volumes)),
        'max_volume': float(np.max(volumes)),
        'mean_volume': float(np.mean(volumes)),
        'min_quality': float(np.min(qualities)),
        'mean_quality': float(np.mean(qualities)),
        'volumes': volumes,
        'qualities': qualities,
    }


def check_mesh_validity(mesh, min_quality_tol=1e-4):
    neg, zero = orient_elements(mesh)
    metrics = compute_mesh_quality_metrics(mesh)

    valid = True
    issues = []

    if zero > 0:
        valid = False
        issues.append(f"Zero-volume elements: {zero}")

    if metrics['min_quality'] < min_quality_tol:
        valid = False
        issues.append(f"Poor element quality: min={metrics['min_quality']:.2e}")

    if not np.all(np.isfinite(mesh.nodes)):
        valid = False
        issues.append("Non-finite node coordinates detected")

    return valid, issues, metrics
