
import numpy as np
from math import gamma






def legendre_nodes_weights(n, a=-1.0, b=1.0):
    if n <= 0:
        raise ValueError("order must be positive")


    aj = np.zeros(n)
    bj = np.zeros(n)
    for i in range(1, n + 1):
        bj[i - 1] = i / np.sqrt(4.0 * i * i - 1.0)


    T = np.diag(aj) + np.diag(bj[:-1], 1) + np.diag(bj[:-1], -1)
    eigenvalues, eigenvectors = np.linalg.eigh(T)

    x = eigenvalues
    w = 2.0 * (eigenvectors[0, :] ** 2)


    if not (a == -1.0 and b == 1.0):
        scale = (b - a) / 2.0
        shift = (a + b) / 2.0
        x = scale * x + shift
        w = scale * w

    return x, w


def gauss_legendre_integral(f, a, b, n=64):
    x, w = legendre_nodes_weights(n, a, b)
    return np.sum(w * f(x))






def alpert_log_rule(rule_idx):
    if rule_idx < 1 or rule_idx > 10:
        raise ValueError("rule_idx must be in [1,10]")

    rules = {
        1: (
            np.array([1.591549430918953e-01]),
            np.array([5.0e-01])
        ),
        2: (
            np.array([1.150395811972836e-01, 9.365464527949632e-01]),
            np.array([3.913373788753340e-01, 1.108662621124666e+00])
        ),
        3: (
            np.array([2.379647284118974e-02, 2.935370741501914e-01, 1.023715124251890e+00]),
            np.array([8.795942675593887e-02, 4.989017152913699e-01, 9.131388579526912e-01])
        ),
        4: (
            np.array([2.339013027203800e-02, 2.854764931311984e-01, 1.005403327220700e+00, 1.994970303994294e+00]),
            np.array([8.609736556158105e-02, 4.847019685417959e-01, 9.152988869123725e-01, 1.013901778984250e+00])
        ),
        5: (
            np.array([4.004884194926570e-03, 7.745655373336686e-02, 3.972849993523248e-01,
                      1.075673352915104e+00, 2.003796927111872e+00]),
            np.array([1.671879691147102e-02, 1.636958371447360e-01, 4.981856569770637e-01,
                      8.372266245578912e-01, 9.841730844088381e-01])
        ),
        6: (
            np.array([6.531815708567918e-03, 9.086744584657729e-02, 3.967966533375878e-01,
                      1.027856640525646e+00, 1.945288592909266e+00, 2.980147933889640e+00,
                      3.998861349951123e+00]),
            np.array([2.462194198995203e-02, 1.701315866854178e-01, 4.609256358650077e-01,
                      7.947291148621894e-01, 1.008710414337933e+00, 1.036093649726216e+00,
                      1.004787656533285e+00])
        ),
        7: (
            np.array([1.175089381227308e-03, 1.877034129831289e-02, 9.686468391426860e-02,
                      3.004818668002884e-01, 6.901331557173356e-01, 1.293695738083659e+00,
                      2.090187729798780e+00, 3.016719313149212e+00, 4.001369747872486e+00,
                      5.000025661793423e+00]),
            np.array([4.560746882084207e-03, 3.810606322384757e-02, 1.293864997289512e-01,
                      2.884360381408835e-01, 4.958111914344961e-01, 7.077154600594529e-01,
                      8.741924365285083e-01, 9.661361986515218e-01, 9.957887866078700e-01,
                      9.998665787423845e-01])
        ),
        8: (
            np.array([1.674223682668368e-03, 2.441110095009738e-02, 1.153851297429517e-01,
                      3.345898490480388e-01, 7.329740531807683e-01, 1.332305048525433e+00,
                      2.114358752325948e+00, 3.026084549655318e+00, 4.003166301292590e+00,
                      5.000141170055870e+00, 6.000001002441859e+00]),
            np.array([6.364190780720557e-03, 4.723964143287529e-02, 1.450891158385963e-01,
                      3.021659470785897e-01, 4.984270739715340e-01, 6.971213795176096e-01,
                      8.577295622757315e-01, 9.544136554351155e-01, 9.919938052776484e-01,
                      9.994621875822987e-01, 9.999934408092805e-01])
        ),
        9: (
            np.array([9.305182368545380e-04, 1.373832458434617e-02, 6.630752760779359e-02,
                      1.979971397622003e-01, 4.504313503816532e-01, 8.571888631101634e-01,
                      1.434505229617112e+00, 2.175177834137754e+00, 3.047955068386372e+00,
                      4.004974906813428e+00, 4.998525901820967e+00, 5.999523015116678e+00,
                      6.999963617883990e+00, 7.999999488130134e+00]),
            np.array([3.545060644780164e-03, 2.681514031576498e-02, 8.504092035093420e-02,
                      1.854526216643691e-01, 3.251724374883192e-01, 4.911553747260108e-01,
                      6.622933417369036e-01, 8.137254578840510e-01, 9.235595514944174e-01,
                      9.821609923744658e-01, 1.000047394596121e+00, 1.000909336693954e+00,
                      1.000119534283784e+00, 1.000002835746089e+00])
        ),
        10: (
            np.array([8.371529832014113e-04, 1.239382725542637e-02, 6.009290785739468e-02,
                      1.805991249601928e-01, 4.142832599028031e-01, 7.964747731112430e-01,
                      1.348993882467059e+00, 2.073471660264395e+00, 2.947904939031494e+00,
                      3.928129252248612e+00, 4.957203086563112e+00, 5.986360113977494e+00,
                      6.997957704791519e+00, 7.999888757524622e+00, 8.999998754306120e+00]),
            np.array([3.190919086626234e-03, 2.423621380426338e-02, 7.740135521653088e-02,
                      1.704889420286369e-01, 3.029123478511309e-01, 4.652220834914617e-01,
                      6.401489637096768e-01, 8.051212946181061e-01, 9.362411945698647e-01,
                      1.014359775369075e+00, 1.035167721053657e+00, 1.020308624984610e+00,
                      1.004798397441514e+00, 1.000395017352309e+00, 1.000007149422537e+00])
        ),
    }
    x, w = rules[rule_idx]
    return x.copy(), w.copy()


def alpert_log_integral(f, h, rule_idx=5):
    x_rel, w = alpert_log_rule(rule_idx)
    x = x_rel * h
    return np.sum(w * f(x))






def chebyshev_eval(x, coef):
    coef = np.asarray(coef, dtype=float)
    nc = coef.size
    if nc == 0:
        return np.zeros_like(x)

    x = np.asarray(x, dtype=float)
    scalar_input = False
    if x.ndim == 0:
        x = x.reshape(1)
        scalar_input = True


    x_clip = np.clip(x, -1.0, 1.0)

    x2 = 2.0 * x_clip
    b0 = np.full_like(x_clip, coef[-1])
    b1 = np.zeros_like(x_clip)
    b2 = np.zeros_like(x_clip)

    for i in range(nc - 2, -1, -1):
        b2 = b1
        b1 = b0
        b0 = coef[i] - b2 + x2 * b1

    y = 0.5 * (b0 - b2)
    if scalar_input:
        return y.item()
    return y


def chebyshev_coefficients(f, n):
    j = np.arange(n + 1)
    x = np.cos(np.pi * j / n)
    fx = f(x)
    c = np.zeros(n + 1)
    for k in range(n + 1):
        coef = np.cos(np.pi * k * j / n)
        if k == 0 or k == n:
            c[k] = np.sum(fx * coef) / n
        else:
            c[k] = 2.0 * np.sum(fx * coef) / n
    return c






def _tensor_product_1d(x_list, w_list):
    dim = len(x_list)
    if dim == 1:
        return x_list[0].reshape(-1, 1), w_list[0]

    nodes = x_list[0].reshape(-1, 1)
    weights = w_list[0]
    for d in range(1, dim):
        xi = x_list[d].reshape(-1, 1)
        wi = w_list[d]

        n_new = nodes.shape[0] * xi.shape[0]
        new_nodes = np.zeros((n_new, d + 1))
        new_weights = np.zeros(n_new)
        idx = 0
        for i in range(nodes.shape[0]):
            for j in range(xi.shape[0]):
                new_nodes[idx, :d] = nodes[i, :]
                new_nodes[idx, d] = xi[j, 0]
                new_weights[idx] = weights[i] * wi[j]
                idx += 1
        nodes = new_nodes
        weights = new_weights
    return nodes, weights


def _get_sequences(dim, total):
    if total < dim:
        return np.empty((0, dim), dtype=int)
    if dim == 1:
        return np.array([[total]], dtype=int)
    seqs = []
    for i in range(1, total - dim + 2):
        sub = _get_sequences(dim - 1, total - i)
        for s in sub:
            seqs.append([i] + list(s))
    return np.array(seqs, dtype=int)


def sparse_grid_gauss_legendre(dim, k, f, a=-1.0, b=1.0):
    if dim <= 0 or k <= 0:
        raise ValueError("dim and k must be positive")


    x1d = []
    w1d = []
    n1d = []
    for level in range(1, k + 1):
        n_pts = level
        xi, wi = legendre_nodes_weights(n_pts, a, b)
        x1d.append(xi)
        w1d.append(wi)
        n1d.append(n_pts)

    minq = max(0, k - dim)
    maxq = k - 1

    all_nodes = []
    all_weights = []

    for q in range(minq, maxq + 1):
        bq = ((-1) ** (maxq - q)) * _n_choose_k(dim - 1, dim + q - k)
        seqs = _get_sequences(dim, dim + q)
        for s in seqs:

            levels = s
            xl = [x1d[lv - 1] for lv in levels]
            wl = [w1d[lv - 1] for lv in levels]
            nodes, weights = _tensor_product_1d(xl, wl)
            all_nodes.append(nodes)
            all_weights.append(bq * weights)

    if not all_nodes:
        return 0.0, 0

    nodes = np.vstack(all_nodes)
    weights = np.hstack(all_weights)




    rounded = np.round(nodes, decimals=12)

    order = np.lexsort(rounded.T)
    nodes_sorted = rounded[order]
    weights_sorted = weights[order]

    unique_nodes = [nodes_sorted[0]]
    unique_weights = [weights_sorted[0]]
    for i in range(1, nodes_sorted.shape[0]):
        if np.allclose(nodes_sorted[i], unique_nodes[-1], atol=1e-12):
            unique_weights[-1] += weights_sorted[i]
        else:
            unique_nodes.append(nodes_sorted[i])
            unique_weights.append(weights_sorted[i])

    unique_nodes = np.array(unique_nodes)
    unique_weights = np.array(unique_weights)


    vol = (b - a) ** dim
    if abs(np.sum(unique_weights)) > 1e-15:
        unique_weights = unique_weights / np.sum(unique_weights) * vol

    val = 0.0
    for i in range(unique_nodes.shape[0]):
        val += unique_weights[i] * f(unique_nodes[i])

    return val, unique_nodes.shape[0]


def _n_choose_k(n, k):
    if n < 0 or k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    res = 1
    for i in range(1, k + 1):
        res = res * (n - k + i) // i
    return res
