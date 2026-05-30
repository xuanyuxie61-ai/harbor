
import numpy as np






def gauss_legendre_1d(order):
    if order == 1:
        x = np.array([0.0])
        w = np.array([2.0])
    elif order == 2:
        x = np.array([-1.0/np.sqrt(3.0), 1.0/np.sqrt(3.0)])
        w = np.array([1.0, 1.0])
    elif order == 3:
        x = np.array([-np.sqrt(3.0/5.0), 0.0, np.sqrt(3.0/5.0)])
        w = np.array([5.0/9.0, 8.0/9.0, 5.0/9.0])
    elif order == 4:
        x = np.array([
            -np.sqrt((3.0 + 2.0*np.sqrt(6.0/5.0)) / 7.0),
            -np.sqrt((3.0 - 2.0*np.sqrt(6.0/5.0)) / 7.0),
             np.sqrt((3.0 - 2.0*np.sqrt(6.0/5.0)) / 7.0),
             np.sqrt((3.0 + 2.0*np.sqrt(6.0/5.0)) / 7.0),
        ])
        w = np.array([
            (18.0 - np.sqrt(30.0)) / 36.0,
            (18.0 + np.sqrt(30.0)) / 36.0,
            (18.0 + np.sqrt(30.0)) / 36.0,
            (18.0 - np.sqrt(30.0)) / 36.0,
        ])
    elif order == 5:
        x = np.array([
            -np.sqrt(5.0 + 2.0*np.sqrt(10.0/7.0)) / 3.0,
            -np.sqrt(5.0 - 2.0*np.sqrt(10.0/7.0)) / 3.0,
            0.0,
             np.sqrt(5.0 - 2.0*np.sqrt(10.0/7.0)) / 3.0,
             np.sqrt(5.0 + 2.0*np.sqrt(10.0/7.0)) / 3.0,
        ])
        w = np.array([
            (322.0 - 13.0*np.sqrt(70.0)) / 900.0,
            (322.0 + 13.0*np.sqrt(70.0)) / 900.0,
            128.0 / 225.0,
            (322.0 + 13.0*np.sqrt(70.0)) / 900.0,
            (322.0 - 13.0*np.sqrt(70.0)) / 900.0,
        ])
    else:
        raise ValueError(f"不支持的求阶 order={order}，仅支持 1-5")
    
    return x, w






def cube_gauss_rule(a, b, order_1d):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    order_1d = np.asarray(order_1d, dtype=int)
    

    nodes_list = []
    weights_list = []
    for dim in range(3):
        x_1d, w_1d = gauss_legendre_1d(order_1d[dim])

        x_mapped = 0.5 * (b[dim] - a[dim]) * x_1d + 0.5 * (a[dim] + b[dim])
        w_mapped = 0.5 * (b[dim] - a[dim]) * w_1d
        nodes_list.append(x_mapped)
        weights_list.append(w_mapped)
    

    n_total = np.prod(order_1d)
    xyz = np.zeros((3, n_total))
    w = np.ones(n_total)
    
    idx = 0
    for i in range(order_1d[0]):
        for j in range(order_1d[1]):
            for k in range(order_1d[2]):
                xyz[0, idx] = nodes_list[0][i]
                xyz[1, idx] = nodes_list[1][j]
                xyz[2, idx] = nodes_list[2][k]
                w[idx] = weights_list[0][i] * weights_list[1][j] * weights_list[2][k]
                idx += 1
    
    return xyz, w


def integrate_over_cube(f, a, b, order_1d=(3, 3, 3)):
    xyz, w = cube_gauss_rule(a, b, order_1d)
    total = 0.0
    for i in range(xyz.shape[1]):
        total += w[i] * f(xyz[0, i], xyz[1, i], xyz[2, i])
    return total


def integrate_dic_inventory_cube(DIC_func, a, b, rho_func, order_1d=(3, 3, 3)):
    def integrand(x, y, z):
        return rho_func(x, y, z) * DIC_func(x, y, z)
    
    return integrate_over_cube(integrand, a, b, order_1d)






def keast_subrule_data(rule_index):
    if rule_index == 1:

        subrules = [{
            'bary': np.array([[0.25, 0.25, 0.25, 0.25]]),
            'weights': np.array([1.0]),
            'suborder': 1,
        }]
    elif rule_index == 2:

        a = 0.5854101966249685
        b = 0.1381966011250105
        subrules = [{
            'bary': np.array([[a, b, b, b], [b, a, b, b], [b, b, a, b], [b, b, b, a]]),
            'weights': np.array([0.25, 0.25, 0.25, 0.25]),
            'suborder': 4,
        }]
    elif rule_index == 3:

        subrules = [{
            'bary': np.array([[0.25, 0.25, 0.25, 0.25]]),
            'weights': np.array([-0.8]),
            'suborder': 1,
        }, {
            'bary': np.array([
                [1.0/6.0, 1.0/6.0, 1.0/6.0, 0.5],
                [1.0/6.0, 1.0/6.0, 0.5, 1.0/6.0],
                [1.0/6.0, 0.5, 1.0/6.0, 1.0/6.0],
                [0.5, 1.0/6.0, 1.0/6.0, 1.0/6.0],
            ]),
            'weights': np.array([0.45, 0.45, 0.45, 0.45]),
            'suborder': 4,
        }]
    elif rule_index == 4:

        return keast_subrule_data(3)
    else:

        return keast_subrule_data(3)
    
    return subrules


def keast_rule(rule_index):
    if rule_index <= 3:
        subrules = keast_subrule_data(rule_index)
    else:

        subrules = keast_subrule_data(3)
    
    xyz_list = []
    w_list = []
    
    for sr in subrules:
        bary = sr['bary']
        weights = sr['weights']
        n_sub = bary.shape[0]
        


        xyz_sub = np.zeros((3, n_sub))
        xyz_sub[0, :] = bary[:, 1]
        xyz_sub[1, :] = bary[:, 2]
        xyz_sub[2, :] = bary[:, 3]
        
        xyz_list.append(xyz_sub)
        w_list.append(weights)
    
    xyz = np.hstack(xyz_list)
    w = np.hstack(w_list)
    

    w = w / np.sum(w) * (1.0 / 6.0)
    
    return xyz, w


def tetrahedron_reference_to_physical(v, xyz_ref):
    J = np.column_stack([v[:, 1] - v[:, 0], v[:, 2] - v[:, 0], v[:, 3] - v[:, 0]])
    xyz_phys = v[:, 0:1] + J @ xyz_ref
    return xyz_phys


def tetrahedron_volume(v):
    J = np.column_stack([v[:, 1] - v[:, 0], v[:, 2] - v[:, 0], v[:, 3] - v[:, 0]])
    return abs(np.linalg.det(J)) / 6.0


def integrate_over_tetrahedron(f, v, rule_index=3):
    xyz_ref, w = keast_rule(rule_index)
    xyz_phys = tetrahedron_reference_to_physical(v, xyz_ref)
    vol = tetrahedron_volume(v)
    
    total = 0.0
    for i in range(xyz_phys.shape[1]):
        total += w[i] * f(xyz_phys[0, i], xyz_phys[1, i], xyz_phys[2, i])
    


    total *= vol / (1.0 / 6.0)
    return total






def hypersphere_surface_area(m):
    if m < 1:
        raise ValueError("维度 m 必须 ≥ 1")
    from math import gamma, pi
    return 2.0 * pi**(m / 2.0) / gamma(m / 2.0)


def hypersphere_uniform_sample(m, n_samples, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    x = np.random.randn(m, n_samples)
    norms = np.linalg.norm(x, axis=0)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return x / norms


def integrate_on_hypersphere(f, m, n_samples, seed=None):
    samples = hypersphere_uniform_sample(m, n_samples, seed)
    total = 0.0
    for i in range(n_samples):
        total += f(samples[:, i])
    
    area = hypersphere_surface_area(m)
    return area * total / n_samples


def parameter_sensitivity_on_sphere(base_params, perturb_scale, f_model, m, n_samples=500):
    samples = hypersphere_uniform_sample(m, n_samples)
    outputs = []
    
    for i in range(n_samples):
        perturbed = base_params + perturb_scale * samples[:, i]

        perturbed = np.maximum(perturbed, 1e-6)
        try:
            val = f_model(perturbed)
        except Exception:
            val = np.nan
        outputs.append(val)
    
    outputs = np.array(outputs)
    valid = outputs[~np.isnan(outputs)]
    
    return {
        'mean': np.mean(valid) if len(valid) > 0 else np.nan,
        'std': np.std(valid) if len(valid) > 0 else np.nan,
        'min': np.min(valid) if len(valid) > 0 else np.nan,
        'max': np.max(valid) if len(valid) > 0 else np.nan,
        'samples': outputs,
    }
