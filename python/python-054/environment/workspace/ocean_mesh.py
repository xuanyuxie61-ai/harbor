
import numpy as np






def triangle_area(x1, y1, x2, y2, x3, y3):
    return 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))


def quadrilateral_area(xy_quad):
    a1 = triangle_area(*xy_quad[0], *xy_quad[1], *xy_quad[2])
    a2 = triangle_area(*xy_quad[0], *xy_quad[2], *xy_quad[3])
    return a1 + a2


def q4_shape_functions(r, s):
    psi = np.array([
        0.25 * (1.0 - r) * (1.0 - s),
        0.25 * (1.0 + r) * (1.0 - s),
        0.25 * (1.0 + r) * (1.0 + s),
        0.25 * (1.0 - r) * (1.0 + s),
    ])
    dpsi_dr = np.array([
        -0.25 * (1.0 - s),  0.25 * (1.0 - s),
         0.25 * (1.0 + s), -0.25 * (1.0 + s),
    ])
    dpsi_ds = np.array([
        -0.25 * (1.0 - r), -0.25 * (1.0 + r),
         0.25 * (1.0 + r),  0.25 * (1.0 - r),
    ])
    return psi, dpsi_dr, dpsi_ds


def reference_to_physical_q4(xy_nodes, r, s):
    psi, _, _ = q4_shape_functions(r, s)
    x = np.dot(psi, xy_nodes[:, 0])
    y = np.dot(psi, xy_nodes[:, 1])
    return x, y






def generate_ocean_rectangle_mesh(Lx, Ly, nx, ny, x0=0.0, y0=0.0):
    node_num_x = nx + 1
    node_num_y = ny + 1
    node_num = node_num_x * node_num_y
    element_num = nx * ny
    
    dx = Lx / nx
    dy = Ly / ny
    
    node_xy = np.zeros((node_num, 2), dtype=float)
    for j in range(node_num_y):
        for i in range(node_num_x):
            idx = j * node_num_x + i
            node_xy[idx, 0] = x0 + i * dx
            node_xy[idx, 1] = y0 + j * dy
    
    element_node = np.zeros((element_num, 4), dtype=int)
    for j in range(ny):
        for i in range(nx):
            e_idx = j * nx + i
            n1 = j * node_num_x + i
            n2 = n1 + 1
            n3 = n2 + node_num_x
            n4 = n1 + node_num_x
            element_node[e_idx, :] = [n1, n2, n3, n4]
    
    return node_xy, element_node, nx, ny


def generate_ocean_semicircle_mesh(R, nx, ny):
    node_num = (nx + 1) * (ny + 1)
    element_num = nx * ny
    
    node_xy = np.zeros((node_num, 2), dtype=float)
    element_node = np.zeros((element_num, 4), dtype=int)
    

    for j in range(ny + 1):
        theta = np.pi * j / ny
        for i in range(nx + 1):

            t = i / nx
            r = R * (t**1.5)
            idx = j * (nx + 1) + i
            node_xy[idx, 0] = r * np.cos(theta)
            node_xy[idx, 1] = r * np.sin(theta)
    
    for j in range(ny):
        for i in range(nx):
            e_idx = j * nx + i
            n1 = j * (nx + 1) + i
            n2 = n1 + 1
            n3 = n2 + (nx + 1)
            n4 = n1 + (nx + 1)
            element_node[e_idx, :] = [n1, n2, n3, n4]
    
    return node_xy, element_node






def compute_adjacency(element_node, node_num):
    adjacency = [set() for _ in range(node_num)]
    element_num = element_node.shape[0]
    
    for e in range(element_num):
        nodes = element_node[e, :]
        for i in range(4):
            for j in range(4):
                if i != j:
                    adjacency[nodes[i]].add(nodes[j])
    

    for i in range(node_num):
        adjacency[i].add(i)
    
    return adjacency


def mesh_bandwidth(element_node, node_num):
    ml = 0
    mu = 0
    element_num = element_node.shape[0]
    
    for e in range(element_num):
        nodes = element_node[e, :]
        for i in range(4):
            for j in range(4):
                gi = nodes[i]
                gj = nodes[j]
                if gi < gj:
                    mu = max(mu, gj - gi)
                elif gi > gj:
                    ml = max(ml, gi - gj)
    
    m = ml + 1 + mu
    return {'ml': ml, 'mu': mu, 'm': m}


def compute_element_areas(node_xy, element_node):
    element_num = element_node.shape[0]
    areas = np.zeros(element_num)
    for e in range(element_num):
        nodes = element_node[e, :]
        xy_quad = node_xy[nodes, :]
        areas[e] = quadrilateral_area(xy_quad)
    return areas, np.sum(areas)


def compute_boundary_edges(element_node):
    edge_count = {}
    element_num = element_node.shape[0]
    local_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    
    for e in range(element_num):
        nodes = element_node[e, :]
        for le in local_edges:
            n1, n2 = nodes[le[0]], nodes[le[1]]
            key = (min(n1, n2), max(n1, n2))
            edge_count[key] = edge_count.get(key, 0) + 1
    
    boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
    return boundary_edges, len(boundary_edges)


def sample_q4_mesh(node_xy, element_node, n_samples):
    areas, _ = compute_element_areas(node_xy, element_node)
    element_num = element_node.shape[0]
    

    probs = areas / np.sum(areas)
    chosen_elements = np.random.choice(element_num, size=n_samples, p=probs)
    
    samples = np.zeros((n_samples, 2))
    for k in range(n_samples):
        e = chosen_elements[k]
        nodes = element_node[e, :]
        xy_quad = node_xy[nodes, :]
        


        a1 = triangle_area(*xy_quad[0], *xy_quad[1], *xy_quad[2])
        a2 = triangle_area(*xy_quad[0], *xy_quad[2], *xy_quad[3])
        if np.random.rand() < a1 / (a1 + a2):

            p = xy_quad[0]
            q = xy_quad[1]
            r = xy_quad[2]
        else:

            p = xy_quad[0]
            q = xy_quad[2]
            r = xy_quad[3]
        

        u = np.random.rand()
        v = np.random.rand()
        if u + v > 1.0:
            u = 1.0 - u
            v = 1.0 - v
        samples[k, :] = p + u * (q - p) + v * (r - p)
    
    return samples
