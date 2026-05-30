# -*- coding: utf-8 -*-

import numpy as np



_KRONROD_TABLES = {
    7: {
        'nodes': np.array([
            0.9914553711208126, 0.9491079123427585, 0.8648644233597691,
            0.7415311855993945, 0.5860872354676911, 0.4058451513773972,
            0.2077849550078985, 0.0
        ]),
        'weights_kronrod': np.array([
            0.022935322010529224, 0.06309209262997856, 0.10479001032225018,
            0.14065325971552592, 0.1690047266392679, 0.19035057806478542,
            0.20443294007529889, 0.20948214108472782
        ]),
        'weights_gauss': np.array([
            0.0, 0.1294849661688697, 0.0, 0.2797053914892766,
            0.0, 0.3818300505051189, 0.0, 0.4179591836734694
        ])
    },
    10: {
        'nodes': np.array([
            0.9931285991850949, 0.9639719272779138, 0.9122344282513259,
            0.8391169718222188, 0.7463319064601508, 0.6360536807265150,
            0.5108670019508271, 0.37370608871541955, 0.22778585114164507,
            0.07652652113349734, 0.0
        ]),
        'weights_kronrod': np.array([
            0.017614007139152118, 0.04060142980038694, 0.06267204833410906,
            0.08327674157670475, 0.10193011981724044, 0.11819453196151841,
            0.13168863844917664, 0.14209610931838205, 0.14917298647260374,
            0.15275338713072584, 0.1541516949815435
        ]),
        'weights_gauss': np.array([
            0.0, 0.06667134430868814, 0.0, 0.1494513491505806,
            0.0, 0.21908636251598204, 0.0, 0.26926671930999635,
            0.0, 0.29552422471475287, 0.0
        ])
    },
    15: {
        'nodes': np.array([
            0.9956571630258081, 0.9739065285171717, 0.9301574913557082,
            0.8650633666889845, 0.7808177265864169, 0.6794095682990244,
            0.5627571346686047, 0.4333953941292472, 0.2943928627014602,
            0.14887433898163122, 0.0
        ]),
        'weights_kronrod': np.array([
            0.011694638867371874, 0.032558162307964725, 0.054755896574351995,
            0.07503967481091996, 0.0931254545836976, 0.10938715880229764,
            0.12349197626206585, 0.13470921731147334, 0.14277593857706008,
            0.14773910490133849, 0.1494455540029169
        ]),
        'weights_gauss': np.array([
            0.0, 0.03075324199611727, 0.0, 0.07036604748810812,
            0.0, 0.10715922046717194, 0.0, 0.13957067792615432,
            0.0, 0.16626920581699392, 0.0
        ])
    }
}


def kronrod_rule(n, tol=1e-12):
    if n in _KRONROD_TABLES:
        data = _KRONROD_TABLES[n]
        return data['nodes'].copy(), data['weights_kronrod'].copy(), data['weights_gauss'].copy()


    data = _KRONROD_TABLES[7]
    return data['nodes'].copy(), data['weights_kronrod'].copy(), data['weights_gauss'].copy()


def adaptive_kronrod_integrate(f, a, b, n=7, tol=1e-8, max_iter=20):
    x_std, w_kron, w_gauss = kronrod_rule(n)


    nodes = []
    weights_k = []
    weights_g = []
    for i in range(len(x_std)):
        nodes.append(x_std[i])
        weights_k.append(w_kron[i])
        weights_g.append(w_gauss[i])
        if x_std[i] != 0.0:
            nodes.append(-x_std[i])
            weights_k.append(w_kron[i])
            weights_g.append(w_gauss[i])

    nodes = np.array(nodes, dtype=float)
    weights_k = np.array(weights_k, dtype=float)
    weights_g = np.array(weights_g, dtype=float)

    def integrate_interval(f_int, a_int, b_int, tol_int, depth):
        mid = 0.5 * (a_int + b_int)
        scale = 0.5 * (b_int - a_int)
        x_local = mid + scale * nodes

        try:
            fx = np.array([f_int(xi) for xi in x_local], dtype=float)
        except Exception:
            fx = np.zeros(len(x_local), dtype=float)

        ik = scale * np.sum(weights_k * fx)
        ig = scale * np.sum(weights_g * fx)

        err = abs(ik - ig)

        if err < tol_int or depth >= max_iter:
            return ik
        else:
            m = 0.5 * (a_int + b_int)
            left = integrate_interval(f_int, a_int, m, tol_int * 0.5, depth + 1)
            right = integrate_interval(f_int, m, b_int, tol_int * 0.5, depth + 1)
            return left + right

    return integrate_interval(f, a, b, tol, 0)


def integrate_convection_flux(u_func, v_func, x_nodes, y_nodes, order=7):
    x_std, w_kron, _ = kronrod_rule(order)
    nodes = []
    weights = []
    for i in range(len(x_std)):
        nodes.append(x_std[i])
        weights.append(w_kron[i])
        if x_std[i] != 0.0:
            nodes.append(-x_std[i])
            weights.append(w_kron[i])

    nodes = np.array(nodes, dtype=float)
    weights = np.array(weights, dtype=float)

    result = 0.0
    for xi, wi in zip(nodes, weights):
        for eta, wj in zip(nodes, weights):
            x = 0.5 * (1.0 + xi)
            y = 0.5 * (1.0 + eta)
            try:
                val = u_func(x, y) * v_func(x, y)
                result += wi * wj * val * 0.25
            except Exception:
                result += 0.0
    return result
