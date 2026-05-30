
import numpy as np






def polyiamond_hexagon_mesh(order, scale=1.0):
    if order < 1:
        order = 1
    

    nodes_list = []
    node_index_map = {}
    


    idx = 0
    for q in range(-order, order + 1):
        for r in range(-order, order + 1):
            s = -q - r
            if -order <= s <= order:

                x = scale * (q + r / 2.0)
                y = scale * (r * np.sqrt(3.0) / 2.0)
                nodes_list.append([x, y])
                node_index_map[(q, r)] = idx
                idx += 1
    
    nodes = np.array(nodes_list)
    n_nodes = len(nodes)
    

    elements_list = []
    boundary_nodes = set()
    
    for q in range(-order, order + 1):
        for r in range(-order, order + 1):
            s = -q - r
            if not (-order <= s <= order):
                continue
            
            i = node_index_map.get((q, r))
            if i is None:
                continue
            

            if q == -order or q == order or r == -order or r == order or s == -order or s == order:
                boundary_nodes.add(i)
            


            i1 = node_index_map.get((q, r))
            i2 = node_index_map.get((q + 1, r))
            i3 = node_index_map.get((q, r + 1))
            if i1 is not None and i2 is not None and i3 is not None:
                elements_list.append([i1, i2, i3])
            

            j1 = node_index_map.get((q + 1, r + 1))
            j2 = node_index_map.get((q, r + 1))
            j3 = node_index_map.get((q + 1, r))
            if j1 is not None and j2 is not None and j3 is not None:
                elements_list.append([j1, j2, j3])
    
    elements = np.array(elements_list)
    boundary_nodes = np.array(sorted(boundary_nodes), dtype=int)
    
    return nodes, elements, boundary_nodes


def generate_perforated_plate_mesh(nx, ny, hole_positions=None, hole_radius=0.15,
                                    Lx=1.0, Ly=1.0):
    if hole_positions is None:
        hole_positions = [(0.3, 0.3), (0.7, 0.7), (0.5, 0.5)]
    

    x = np.linspace(0, Lx, nx + 1)
    y = np.linspace(0, Ly, ny + 1)
    xx, yy = np.meshgrid(x, y)
    
    nodes_list = []
    node_map = {}
    idx = 0
    for j in range(ny + 1):
        for i in range(nx + 1):
            xi, yi = xx[j, i], yy[j, i]

            inside_hole = False
            for hx, hy in hole_positions:
                if (xi - hx) ** 2 + (yi - hy) ** 2 < hole_radius ** 2:
                    inside_hole = True
                    break
            if not inside_hole:
                nodes_list.append([xi, yi])
                node_map[(i, j)] = idx
                idx += 1
    
    nodes = np.array(nodes_list)
    

    elements_list = []
    boundary_nodes = set()
    
    for j in range(ny):
        for i in range(nx):

            n1 = node_map.get((i, j))
            n2 = node_map.get((i + 1, j))
            n3 = node_map.get((i, j + 1))
            n4 = node_map.get((i + 1, j + 1))
            
            if n1 is not None and n2 is not None and n3 is not None:
                elements_list.append([n1, n2, n3])

                if i == 0 or j == 0:
                    boundary_nodes.update([n1, n2, n3])
            if n2 is not None and n4 is not None and n3 is not None:
                elements_list.append([n2, n4, n3])
                if i == nx - 1 or j == ny - 1:
                    boundary_nodes.update([n2, n4, n3])
    
    elements = np.array(elements_list)
    boundary_nodes = np.array(sorted(boundary_nodes), dtype=int)
    
    return nodes, elements, boundary_nodes






def magic_square(n):
    if n % 2 == 1:

        M = np.zeros((n, n), dtype=int)
        i, j = 0, n // 2
        for num in range(1, n * n + 1):
            M[i, j] = num
            new_i, new_j = (i - 1) % n, (j + 1) % n
            if M[new_i, new_j] != 0:
                new_i, new_j = (i + 1) % n, j
            i, j = new_i, new_j
    elif n % 4 == 0:

        M = np.arange(1, n * n + 1).reshape(n, n)
        mask = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(n):
                if (i % 4 == j % 4) or ((i % 4 + j % 4) == 3):
                    mask[i, j] = True
        M[mask] = n * n + 1 - M[mask]
    else:

        M = np.zeros((n, n), dtype=int)
        half = n // 2
        M[:half, :half] = magic_square(half)
        M[:half, half:] = magic_square(half) + 2 * half * half
        M[half:, :half] = magic_square(half) + 3 * half * half
        M[half:, half:] = magic_square(half) + half * half


    return M


def construct_test_stiffness_matrix(n, magic_based=False):
    if magic_based and n >= 3:
        M_magic = magic_square(n).astype(float)

        K = M_magic.T @ M_magic
        K = K + np.diag(np.sum(np.abs(K), axis=1) + 1.0)

        K = (K + K.T) / 2.0
    else:

        K = np.zeros((n, n))
        np.fill_diagonal(K, 2.0)
        np.fill_diagonal(K[1:], -1.0)
        np.fill_diagonal(K[:, 1:], -1.0)
        K[0, 0] = 1.0
        K[-1, -1] = 1.0
    
    M = np.eye(n) * 0.1
    return K, M






def nodal_stress_recovery(element_stresses, elements, n_nodes):
    element_stresses = np.asarray(element_stresses)
    is_vector = element_stresses.ndim == 2
    
    if is_vector:
        n_comp = element_stresses.shape[1]
        nodal_sum = np.zeros((n_nodes, n_comp))
    else:
        nodal_sum = np.zeros(n_nodes)
    
    nodal_count = np.zeros(n_nodes)
    
    for e in range(elements.shape[0]):
        for n in elements[e]:
            nodal_sum[n] += element_stresses[e]
            nodal_count[n] += 1.0
    
    nodal_count = np.maximum(nodal_count, 1.0)
    
    if is_vector:
        return nodal_sum / nodal_count[:, np.newaxis]
    return nodal_sum / nodal_count


def compute_element_stresses(displacement, node_coords, elements, E, nu):
    n_elements = elements.shape[0]
    element_stresses = np.zeros((n_elements, 3))
    
    D_mat = E / (1.0 - nu ** 2) * np.array([
        [1.0, nu, 0.0],
        [nu, 1.0, 0.0],
        [0.0, 0.0, (1.0 - nu) / 2.0]
    ])
    
    for e in range(n_elements):
        n1, n2, n3 = elements[e]
        x = node_coords[[n1, n2, n3], 0]
        y = node_coords[[n1, n2, n3], 1]
        
        area = 0.5 * abs((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]))
        if area < 1e-14:
            continue
        


        dN_dx = np.zeros(3)
        dN_dy = np.zeros(3)
        B = np.zeros((3, 6))


        
        u_e = np.array([
            displacement[2 * n1], displacement[2 * n1 + 1],
            displacement[2 * n2], displacement[2 * n2 + 1],
            displacement[2 * n3], displacement[2 * n3 + 1]
        ])
        
        strain = B @ u_e
        stress = D_mat @ strain
        element_stresses[e] = stress
    

    sxx = element_stresses[:, 0]
    syy = element_stresses[:, 1]
    txy = element_stresses[:, 2]
    element_von_mises = np.sqrt(sxx ** 2 + syy ** 2 - sxx * syy + 3.0 * txy ** 2)
    
    return element_stresses, element_von_mises
