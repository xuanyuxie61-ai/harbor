"""
Finite Element Method (FEM) Sparse Matrix Assembly Module.

Integrates:
  - 344_exactness: Gaussian quadrature rules (Legendre, Chebyshev, Hermite, Laguerre)
  - 925_pwl_approx_1d: Piecewise linear shape functions (sparse least-squares structure)
  - 1340_triangulation_node_to_element: mesh connectivity processing

Scientific formulas:
  Weak form of diffusion-reaction equation:
    -nabla·(D(x) nabla u) + sigma(x) u = f(x)  in Omega
    u = g_D  on Gamma_D
    D * du/dn = g_N  on Gamma_N

  Element stiffness matrix (local):
    K_e[i,j] = integral_{T_e} D(x) * nabla phi_i · nabla phi_j dx
             + integral_{T_e} sigma(x) * phi_i * phi_j dx

  For linear triangular elements with barycentric coordinates L_i:
    nabla L_i are constant over element, giving:
    K_e^{diff}[i,j] = D_e * Area_e * (nabla L_i · nabla L_j)
    K_e^{react}[i,j] = sigma_e * Area_e / 12 * (1 + delta_{ij})

  Global assembly:
    K[I_e[i], I_e[j]] += K_e[i,j]
    where I_e maps local to global node indices.
"""

import numpy as np
from utils import legendre_monomial_integral, chebyshev1_monomial_integral


def gauss_legendre_nodes_weights(n):
    """
    Return Gauss-Legendre quadrature nodes and weights for [-1, 1].
    For n = 1..10, use tabulated exact values from seed 344_exactness.
    For n > 10, fallback to numpy.polynomial.legendre.leggauss.
    """
    tabulated = {
        1: ([0.0], [2.0]),
        2: ([-0.5773502691896257, 0.5773502691896257], [1.0, 1.0]),
        3: ([-0.7745966692414834, 0.0, 0.7745966692414834],
            [0.5555555555555556, 0.8888888888888888, 0.5555555555555556]),
        4: ([-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526],
            [0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538]),
        5: ([-0.9061798459386640, -0.5384693101056831, 0.0, 0.5384693101056831, 0.9061798459386640],
            [0.2369268850561891, 0.4786286704993665, 0.5688888888888889, 0.4786286704993665, 0.2369268850561891]),
    }
    if n in tabulated:
        return np.array(tabulated[n][0]), np.array(tabulated[n][1])
    try:
        x, w = np.polynomial.legendre.leggauss(n)
        return x, w
    except Exception:
        raise ValueError(f"Cannot compute Gauss-Legendre quadrature for n={n}")


def triangle_quadrature_3():
    """
    3-point Gauss quadrature on the reference triangle
    with vertices (0,0), (1,0), (0,1).
    Exact for polynomials up to degree 2.

    Points and weights:
      p1 = (1/6, 1/6),  w1 = 1/6
      p2 = (2/3, 1/6),  w2 = 1/6
      p3 = (1/6, 2/3),  w3 = 1/6
    """
    pts = np.array([[1.0 / 6.0, 1.0 / 6.0],
                    [2.0 / 3.0, 1.0 / 6.0],
                    [1.0 / 6.0, 2.0 / 3.0]])
    w = np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    return pts, w


def triangle_quadrature_7():
    """
    7-point Gauss quadrature on the reference triangle.
    Exact for polynomials up to degree 5.

    Citation: Dunavant (1985) - commonly used in FEM.
    """
    pts = np.array([
        [1.0 / 3.0, 1.0 / 3.0],
        [0.059715871789770, 0.470142064105115],
        [0.470142064105115, 0.059715871789770],
        [0.470142064105115, 0.470142064105115],
        [0.797426985353087, 0.101286507323456],
        [0.101286507323456, 0.797426985353087],
        [0.101286507323456, 0.101286507323456],
    ])
    w = np.array([
        0.225000000000000,
        0.132394152788506,
        0.132394152788506,
        0.132394152788506,
        0.125939180544827,
        0.125939180544827,
        0.125939180544827,
    ]) * 0.5  # normalize to reference triangle area = 1/2
    return pts, w


def assemble_fem_matrices(nodes, elements, diffusion_func=None, reaction_func=None,
                          source_func=None, quad_order=7):
    """
    Assemble global sparse stiffness matrix K and load vector F for
    the diffusion-reaction equation on a triangular mesh.

    Parameters:
      nodes         : (n_nodes, 2) array
      elements      : (n_elements, 3) array
      diffusion_func: callable(x, y) -> D(x,y), default lambda x,y: 1.0
      reaction_func : callable(x, y) -> sigma(x,y), default lambda x,y: 0.0
      source_func   : callable(x, y) -> f(x,y), default lambda x,y: 1.0
      quad_order    : 3 or 7

    Returns:
      K : (n_nodes, n_nodes) dense stiffness matrix (small problems)
      F : (n_nodes,) load vector
    """
    nodes = np.asarray(nodes, dtype=float)
    elements = np.asarray(elements, dtype=int)
    n_nodes = nodes.shape[0]
    n_elements = elements.shape[0]

    if diffusion_func is None:
        diffusion_func = lambda x, y: 1.0
    if reaction_func is None:
        reaction_func = lambda x, y: 0.0
    if source_func is None:
        source_func = lambda x, y: 1.0

    if quad_order == 3:
        qpts, qw = triangle_quadrature_3()
    else:
        qpts, qw = triangle_quadrature_7()
    nq = len(qw)

    K = np.zeros((n_nodes, n_nodes), dtype=float)
    F = np.zeros(n_nodes, dtype=float)

    # HOLE_1: FEM element stiffness matrix assembly is missing.
    # Implement the element stiffness matrix computation and global assembly.
    # Key scientific formulas:
    #   - Jacobian of affine map from reference triangle to physical element
    #   - Gradient of barycentric coordinates: gradL = inv(J).T @ dL_dxi
    #   - Diffusion stiffness: Ke_diff[i,j] = D_e * Area_e * (gradL_i · gradL_j)
    #   - Reaction stiffness: Ke_react[i,j] = sigma_e * Area_e / 12 * (1 + delta_{ij})
    #   - Load vector via quadrature: Fe += fq * L * w * |detJ|
    #   - Global assembly: K[gi,gj] += Ke[i,j], F[gi] += Fe[i]
    raise NotImplementedError("HOLE_1: Implement FEM element stiffness assembly.")


def apply_dirichlet_bc(K, F, bc_nodes, bc_values):
    """
    Apply Dirichlet boundary conditions by modifying the linear system
    K u = F  ->  K_mod u = F_mod.

    For each constrained node i with value g_i:
      - Set row i of K to 0, K[i,i] = 1
      - Set F[i] = g_i
      - For all j != i: F[j] -= K[j,i] * g_i, then K[j,i] = 0
    """
    K = np.array(K, dtype=float, copy=True)
    F = np.array(F, dtype=float, copy=True)
    n = K.shape[0]
    bc_nodes = np.asarray(bc_nodes, dtype=int)
    bc_values = np.asarray(bc_values, dtype=float)

    for idx in range(len(bc_nodes)):
        i = bc_nodes[idx]
        if i < 0 or i >= n:
            continue
        g = bc_values[idx]
        # Modify load vector
        for j in range(n):
            if j != i:
                F[j] -= K[j, i] * g
                K[j, i] = 0.0
        # Modify stiffness matrix row
        K[i, :] = 0.0
        K[i, i] = 1.0
        F[i] = g
    return K, F


def exactness_test_fem_quadrature(max_degree=5):
    """
    Verify that the triangular quadrature rules integrate monomials
    x^p y^q exactly up to the expected degree.

    Returns True if all tests pass.
    """
    pts3, w3 = triangle_quadrature_3()
    pts7, w7 = triangle_quadrature_7()

    # Reference triangle vertices: (0,0), (1,0), (0,1)
    # Area = 0.5
    # For exactness test, integrate x^p y^q over reference triangle:
    # I_{p,q} = integral_0^1 integral_0^{1-x} x^p y^q dy dx
    #         = p! q! / (p + q + 2)!

    def exact_integral_ref(p, q):
        import math
        return math.factorial(p) * math.factorial(q) / math.factorial(p + q + 2)

    # Map quadrature from reference to physical (identity here)
    all_pass = True
    for p in range(max_degree + 1):
        for q in range(max_degree + 1 - p):
            exact = exact_integral_ref(p, q)
            approx3 = sum(w * (pt[0] ** p) * (pt[1] ** q) for pt, w in zip(pts3, w3))
            approx7 = sum(w * (pt[0] ** p) * (pt[1] ** q) for pt, w in zip(pts7, w7))
            if p + q <= 2:
                if abs(approx3 - exact) > 1e-14:
                    all_pass = False
            if p + q <= 5:
                if abs(approx7 - exact) > 1e-14:
                    all_pass = False
    return all_pass
