"""
Finite Element Quadrature and Integration Module
=================================================
Based on projects:
  - 1324_triangle_wandzura_rule
  - 532_hexahedron_witherden_rule  
  - 1347_triangulation_quad
  - 347_faces_average

Provides high-precision quadrature rules for 2D triangular and 3D hexahedral
finite elements, triangulation-level integration, and nodal stress averaging
for structural mechanics post-processing.

Key formulas:
- Reference-to-physical map for T3: x_phys = N1*x1 + N2*x2 + N3*x3
- Jacobian determinant: |J| = 2 * Area
- Numerical integration: integral f(x) dx ≈ sum_i w_i * f(x_i) * |J|
- Nodal stress recovery: sigma_node = (sum_{elements containing node} sigma_elem) / count
"""

import numpy as np


# ============================================================================
# Wandzura Triangle Quadrature Rules (Project 1324)
# ============================================================================

# Wandzura rule data: degree, order, (r,s,w) triples in barycentric-like coordinates
# These are symmetric rules on the reference triangle with vertices (0,0), (1,0), (0,1)
WANDZURA_RULES = {
    1: {
        'order': 1,
        'points': np.array([[1.0/3.0, 1.0/3.0]]),
        'weights': np.array([1.0])
    },
    2: {
        'order': 3,
        'points': np.array([
            [2.0/3.0, 1.0/6.0],
            [1.0/6.0, 2.0/3.0],
            [1.0/6.0, 1.0/6.0]
        ]),
        'weights': np.array([1.0/3.0, 1.0/3.0, 1.0/3.0])
    },
    5: {
        'order': 7,
        'points': np.array([
            [0.33333333333333, 0.33333333333333],
            [0.05971587178977, 0.47014206410511],
            [0.47014206410511, 0.05971587178977],
            [0.47014206410511, 0.47014206410511],
            [0.79742698535309, 0.10128650732345],
            [0.10128650732345, 0.79742698535309],
            [0.10128650732345, 0.10128650732345]
        ]),
        'weights': np.array([
            0.2250000000000000,
            0.1323941527885062,
            0.1323941527885062,
            0.1323941527885062,
            0.1259391805448271,
            0.1259391805448271,
            0.1259391805448271
        ])
    }
}


def triangle_area(nodes):
    """
    Compute area of triangle given three vertices.
    
    Area = 0.5 * | (x2-x1)(y3-y1) - (x3-x1)(y2-y1) |
    
    Parameters
    ----------
    nodes : ndarray, shape (3, 2) or (2, 3)
        Vertex coordinates.
    
    Returns
    -------
    area : float
    """
    nodes = np.asarray(nodes)
    if nodes.shape[0] == 2 and nodes.shape[1] == 3:
        nodes = nodes.T
    x1, y1 = nodes[0]
    x2, y2 = nodes[1]
    x3, y3 = nodes[2]
    return 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))


def reference_to_physical_t3(nodes, ref_points):
    """
    Map reference triangle points to physical triangle.
    
    Reference triangle: (0,0), (1,0), (0,1)
    Physical triangle: nodes[0], nodes[1], nodes[2]
    
    x_phys = x1 + (x2-x1)*r + (x3-x1)*s
    y_phys = y1 + (y2-y1)*r + (y3-y1)*s
    
    Parameters
    ----------
    nodes : ndarray, shape (3, 2)
    ref_points : ndarray, shape (n, 2)
        Points in reference coordinates (r, s).
    
    Returns
    -------
    phys_points : ndarray, shape (n, 2)
    """
    nodes = np.asarray(nodes)
    ref_points = np.asarray(ref_points)
    r = ref_points[:, 0:1]
    s = ref_points[:, 1:2]
    x = nodes[0, 0] + (nodes[1, 0] - nodes[0, 0]) * r + (nodes[2, 0] - nodes[0, 0]) * s
    y = nodes[0, 1] + (nodes[1, 1] - nodes[0, 1]) * r + (nodes[2, 1] - nodes[0, 1]) * s
    return np.hstack([x, y])


def integrate_triangle_wandzura(f, nodes, degree=5):
    """
    Integrate function f over triangle using Wandzura rule.
    
    Parameters
    ----------
    f : callable
        Function f(x, y) -> scalar or array.
    nodes : ndarray, shape (3, 2)
        Triangle vertices.
    degree : int
        Polynomial degree of rule (1, 2, or 5).
    
    Returns
    -------
    integral : float or ndarray
    """
    if degree not in WANDZURA_RULES:
        degree = 5
    rule = WANDZURA_RULES[degree]
    ref_pts = rule['points']
    weights = rule['weights']
    
    phys_pts = reference_to_physical_t3(nodes, ref_pts)
    area = triangle_area(nodes)
    
    values = np.array([f(p[0], p[1]) for p in phys_pts])
    # Wandzura weights are normalized to sum to 1 on the reference triangle;
    # multiply by physical triangle area.
    integral = area * np.sum(weights * values)
    return integral


# ============================================================================
# Hexahedron Witherden Rules (Project 532)
# ============================================================================

# Witherden rules on unit hexahedron [0,1]^3
WITHERDEN_RULES = {
    1: {
        'points': np.array([[0.5, 0.5, 0.5]]),
        'weights': np.array([1.0])
    },
    3: {
        'points': np.array([
            [0.0, 0.5, 0.5],
            [0.5, 0.5, 1.0],
            [0.5, 1.0, 0.5],
            [0.5, 0.0, 0.5],
            [1.0, 0.5, 0.5],
            [0.5, 0.5, 0.0]
        ]),
        'weights': np.array([
            0.1666666666666667,
            0.1666666666666667,
            0.1666666666666667,
            0.1666666666666667,
            0.1666666666666667,
            0.1666666666666667
        ])
    }
}


def integrate_hexahedron_witherden(f, bounds, degree=3):
    """
    Integrate function f over hexahedral region using Witherden rule.
    
    Parameters
    ----------
    f : callable
        f(x, y, z) -> value
    bounds : tuple
        ((xmin, xmax), (ymin, ymax), (zmin, zmax))
    degree : int
    
    Returns
    -------
    integral : float
    """
    if degree not in WITHERDEN_RULES:
        degree = 3
    rule = WITHERDEN_RULES[degree]
    ref_pts = rule['points']
    weights = rule['weights']
    
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
    vol = (xmax - xmin) * (ymax - ymin) * (zmax - zmin)
    
    phys_pts = np.zeros_like(ref_pts)
    phys_pts[:, 0] = xmin + ref_pts[:, 0] * (xmax - xmin)
    phys_pts[:, 1] = ymin + ref_pts[:, 1] * (ymax - ymin)
    phys_pts[:, 2] = zmin + ref_pts[:, 2] * (zmax - zmin)
    
    values = np.array([f(p[0], p[1], p[2]) for p in phys_pts])
    integral = vol * np.sum(weights * values)
    return integral


# ============================================================================
# Triangulation-Level Integration (Project 1347)
# ============================================================================

def triangulation_quad(node_coords, elements, node_values):
    """
    Compute integral of nodal data over triangulation.
    
    Uses linear interpolation within each triangle:
    integral_over_T f(x) dx ≈ Area(T) * (f1 + f2 + f3) / 3
    
    Parameters
    ----------
    node_coords : ndarray, shape (n_nodes, 2)
    elements : ndarray, shape (n_elements, 3)
        Triangle connectivity (0-based indices).
    node_values : ndarray, shape (n_nodes,) or (n_nodes, n_components)
    
    Returns
    -------
    integral : float or ndarray
    total_area : float
    """
    node_coords = np.asarray(node_coords)
    elements = np.asarray(elements)
    node_values = np.asarray(node_values)
    
    n_elements = elements.shape[0]
    integral = 0.0
    total_area = 0.0
    
    is_vector = (node_values.ndim == 2)
    if is_vector:
        n_comp = node_values.shape[1]
        integral = np.zeros(n_comp)
    
    for e in range(n_elements):
        n1, n2, n3 = elements[e]
        pts = node_coords[[n1, n2, n3]]
        area = triangle_area(pts)
        total_area += area
        
        if is_vector:
            avg_val = (node_values[n1] + node_values[n2] + node_values[n3]) / 3.0
            integral += area * avg_val
        else:
            avg_val = (node_values[n1] + node_values[n2] + node_values[n3]) / 3.0
            integral += area * avg_val
    
    return integral, total_area


# ============================================================================
# Face/Node Stress Averaging (Project 347)
# ============================================================================

def faces_average(element_stresses, elements, n_nodes):
    """
    Average element stresses to nodal stresses.
    
    In finite element post-processing, stresses are typically computed
    at Gauss points within elements. For nodal stress visualization and
    failure criteria, we average surrounding element stresses to nodes.
    
    Parameters
    ----------
    element_stresses : ndarray, shape (n_elements,) or (n_elements, n_comp)
        Stress value per element (typically average or centroid value).
    elements : ndarray, shape (n_elements, n_nodes_per_element)
        Element connectivity.
    n_nodes : int
        Total number of nodes.
    
    Returns
    -------
    nodal_stresses : ndarray
        Averaged stress at each node.
    """
    element_stresses = np.asarray(element_stresses)
    elements = np.asarray(elements)
    
    is_vector = (element_stresses.ndim == 2)
    if is_vector:
        n_comp = element_stresses.shape[1]
        nodal_sum = np.zeros((n_nodes, n_comp))
    else:
        nodal_sum = np.zeros(n_nodes)
    
    nodal_count = np.zeros(n_nodes)
    
    for e in range(elements.shape[0]):
        nodes_in_elem = elements[e]
        for n in nodes_in_elem:
            nodal_sum[n] += element_stresses[e]
            nodal_count[n] += 1.0
    
    # Avoid division by zero
    nodal_count = np.maximum(nodal_count, 1.0)
    
    if is_vector:
        nodal_stresses = nodal_sum / nodal_count[:, np.newaxis]
    else:
        nodal_stresses = nodal_sum / nodal_count
    
    return nodal_stresses


# ============================================================================
# FEM Matrix Assembly Helpers
# ============================================================================

def t3_shape_functions(xi, eta):
    """
    Linear shape functions and derivatives on reference triangle.
    
    N1 = 1 - xi - eta
    N2 = xi
    N3 = eta
    
    Parameters
    ----------
    xi, eta : float
        Reference coordinates.
    
    Returns
    -------
    N : ndarray, shape (3,)
    dN_dxi : ndarray, shape (3,)
    dN_deta : ndarray, shape (3,)
    """
    N = np.array([1.0 - xi - eta, xi, eta])
    dN_dxi = np.array([-1.0, 1.0, 0.0])
    dN_deta = np.array([-1.0, 0.0, 1.0])
    return N, dN_dxi, dN_deta


def assemble_triangular_fem_matrices(node_coords, elements, 
                                      material_thickness=1.0,
                                      young_modulus=1.0,
                                      poisson_ratio=0.3,
                                      density=1.0):
    """
    Assemble stiffness and mass matrices for 2D plane-stress triangular elements.
    
    Uses one-point centroidal integration for simplicity (exact for constant strain).
    
    Parameters
    ----------
    node_coords : ndarray, shape (n_nodes, 2)
    elements : ndarray, shape (n_elements, 3)
    material_thickness : float
    young_modulus : float or ndarray
    poisson_ratio : float
    density : float
    
    Returns
    -------
    K : ndarray, shape (2*n_nodes, 2*n_nodes)
        Stiffness matrix.
    M : ndarray, shape (2*n_nodes, 2*n_nodes)
        Consistent mass matrix.
    """
    node_coords = np.asarray(node_coords)
    elements = np.asarray(elements)
    n_nodes = node_coords.shape[0]
    n_dof = 2 * n_nodes
    
    K = np.zeros((n_dof, n_dof))
    M = np.zeros((n_dof, n_dof))
    
    # Plane stress D matrix
    E = young_modulus
    nu = poisson_ratio
    factor = E / (1.0 - nu ** 2)
    D_mat = factor * np.array([
        [1.0, nu, 0.0],
        [nu, 1.0, 0.0],
        [0.0, 0.0, (1.0 - nu) / 2.0]
    ])
    
    for e in range(elements.shape[0]):
        n1, n2, n3 = elements[e]
        x = node_coords[[n1, n2, n3], 0]
        y = node_coords[[n1, n2, n3], 1]
        
        # Area
        area = triangle_area(node_coords[[n1, n2, n3]])
        if area < 1e-14:
            continue
        
        # B matrix (strain-displacement)
        # B = [dN1_dx, 0, dN2_dx, 0, dN3_dx, 0
        #      0, dN1_dy, 0, dN2_dy, 0, dN3_dy
        #      dN1_dy, dN1_dx, dN2_dy, dN2_dx, dN3_dy, dN3_dx]
        
        # TODO: Implement B-matrix (strain-displacement) construction for plane-stress T3 element.
        # For linear triangle, dN_dx and dN_dy are constant over the element.
        # B relates nodal displacements [u1, v1, u2, v2, u3, v3] to strains [eps_xx, eps_yy, gamma_xy].
        x1, x2, x3 = x
        y1, y2, y3 = y
        
        dN_dx = np.zeros(3)
        dN_dy = np.zeros(3)
        B = np.zeros((3, 6))
        # HINT: dN_dx[i] = (y_j - y_k) / (2*area) with cyclic permutation (i,j,k) = (0,1,2), (1,2,0), (2,0,1)
        # HINT: B[0, 2*i] = dN_dx[i], B[1, 2*i+1] = dN_dy[i], B[2, 2*i] = dN_dy[i], B[2, 2*i+1] = dN_dx[i]
        
        # Stiffness: K_e = area * h * B^T * D * B
        if np.isscalar(E):
            Ee = E
        else:
            # Average E at nodes
            Ee = np.mean([E[n1], E[n2], E[n3]]) if hasattr(E, '__len__') else E
        
        factor_e = Ee / (1.0 - nu ** 2)
        D_e = factor_e * np.array([
            [1.0, nu, 0.0],
            [nu, 1.0, 0.0],
            [0.0, 0.0, (1.0 - nu) / 2.0]
        ])
        
        Ke = area * material_thickness * B.T @ D_e @ B
        
        # Consistent mass: M_e = (area * h * rho / 12) * [matrix with 2s and 1s]
        Me_local = np.array([
            [2.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 2.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 2.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0, 2.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0, 2.0]
        ]) * (area * material_thickness * density / 12.0)
        
        # Assembly
        dof_map = [2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1, 2 * n3, 2 * n3 + 1]
        for i in range(6):
            for j in range(6):
                K[dof_map[i], dof_map[j]] += Ke[i, j]
                M[dof_map[i], dof_map[j]] += Me_local[i, j]
    
    return K, M
