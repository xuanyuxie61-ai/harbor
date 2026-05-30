
import numpy as np


def triangulation_boundary_edges(triangles):
    triangles = np.asarray(triangles)
    if triangles.shape[1] != 3:
        raise ValueError("triangulation_boundary_edges: 三角形必须为 3 节点")


    edge_dict = {}
    for tri in triangles:
        edges = [
            tuple(sorted((tri[0], tri[1]))),
            tuple(sorted((tri[1], tri[2]))),
            tuple(sorted((tri[2], tri[0]))),
        ]
        for e in edges:
            edge_dict[e] = edge_dict.get(e, 0) + 1


    boundary_edges = [list(e) for e, count in edge_dict.items() if count == 1]
    return np.array(boundary_edges, dtype=int)


def extract_boundary_nodes_3d(element_nodes, all_nodes, lower_z=0.0, upper_z=1.0, tol=1e-6):
    n_node = all_nodes.shape[0]


    lower_nodes = np.where(np.abs(all_nodes[:, 2] - lower_z) < tol)[0]
    upper_nodes = np.where(np.abs(all_nodes[:, 2] - upper_z) < tol)[0]



    side_nodes = set()


    face_dict = {}
    for elem in element_nodes:
        faces = [
            tuple(sorted((elem[0], elem[1], elem[2]))),
            tuple(sorted((elem[0], elem[1], elem[3]))),
            tuple(sorted((elem[0], elem[2], elem[3]))),
            tuple(sorted((elem[1], elem[2], elem[3]))),
        ]
        for f in faces:
            face_dict[f] = face_dict.get(f, 0) + 1


    boundary_faces = [f for f, count in face_dict.items() if count == 1]

    for f in boundary_faces:
        p1, p2, p3 = all_nodes[f[0]], all_nodes[f[1]], all_nodes[f[2]]
        normal = np.cross(p2 - p1, p3 - p1)
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-15:
            continue
        normal = normal / normal_norm


        if abs(normal[2]) < 0.5:
            side_nodes.update(f)

    side_nodes = np.array(sorted(side_nodes), dtype=int)

    return {
        'lower': lower_nodes,
        'upper': upper_nodes,
        'side': side_nodes,
        'all': np.unique(np.concatenate([lower_nodes, upper_nodes, side_nodes]))
    }


def apply_surface_layer_bc(u, v, w, theta, nodes, lower_nodes, u_star=0.3, z0=0.1, kappa=0.4):
    u_new = np.copy(u)
    v_new = np.copy(v)
    w_new = np.copy(w)

    for idx in lower_nodes:
        z = nodes[idx, 2]
        if z < z0:
            z = z0


        u_mag = (u_star / kappa) * np.log(z / z0)


        u_old = u[idx]
        v_old = v[idx]
        uv_mag = np.sqrt(u_old**2 + v_old**2)

        if uv_mag > 1e-12:
            scale = u_mag / uv_mag
            u_new[idx] = u_old * scale
            v_new[idx] = v_old * scale
        else:

            u_new[idx] = u_mag
            v_new[idx] = 0.0

        w_new[idx] = 0.0

    return u_new, v_new, w_new
