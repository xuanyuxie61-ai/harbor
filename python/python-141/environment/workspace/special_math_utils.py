
import numpy as np
import cmath
from math import sqrt, pi, exp, log, gcd






def ellipse_perimeter_ramanujan(a, b):
    a = abs(float(a))
    b = abs(float(b))
    if a < b:
        a, b = b, a
    if a == 0:
        return 0.0

    h = ((a - b) / (a + b)) ** 2
    approx = pi * (a + b) * (1.0 + 3.0 * h / (10.0 + sqrt(4.0 - 3.0 * h)))
    return approx


def ellipse_area_matrix(A, r):
    A = np.asarray(A, dtype=np.float64)
    if A.shape != (2, 2):
        raise ValueError("A必须为2x2矩阵")
    detA = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    if detA <= 0:
        raise ValueError("A必须正定(det>0)")
    return pi * r * r / sqrt(detA)


def complete_elliptic_integral_second_kind(k):
    k = float(k)
    if k < 0.0 or k > 1.0:
        raise ValueError("模数k必须在[0,1]区间内")
    if k == 0.0:
        return pi / 2.0
    if k == 1.0:
        return 1.0

    a, b, c = 1.0, sqrt(1.0 - k * k), k
    sum_c2 = 0.0
    two_pow = 1.0
    for _ in range(50):
        if abs(c) < 1e-15:
            break
        two_pow *= 2.0
        sum_c2 += two_pow * c * c
        a_next = (a + b) * 0.5
        b_next = sqrt(a * b)
        c = (a - b) * 0.5
        a, b = a_next, b_next

    return (pi / 2.0) * (a * a - sum_c2) / a






def laguerre_rootfind(f, x0, degree, abserr=1e-12, kmax=100):
    if degree < 2:
        raise ValueError("degree必须≥2")
    x = float(x0)
    beta = 1.0 / (degree - 1.0)
    ierror = 0
    k = 0
    while True:
        fx, dfx, d2fx = f(x)
        if abs(fx) <= abserr:
            break
        k += 1
        if k > kmax:
            ierror = 2
            return x, ierror, k
        z = dfx * dfx - (beta + 1.0) * fx * d2fx
        z = max(z, 0.0)
        bot = beta * dfx + sqrt(z)
        if abs(bot) < 1e-30:
            ierror = 3
            return x, ierror, k
        dx = -(beta + 1.0) * fx / bot
        x += dx
    return x, ierror, k


def heston_characteristic_root(v0, kappa, theta, sigma, rho, u, T):
    def phase_func(x):

        d = cmath.sqrt((rho * sigma * 1j * x - kappa)**2 + sigma**2 * (1j * x + x*x))
        d = complex(d)
        val = kappa * theta / (sigma**2) * ((kappa - rho*sigma*1j*x - d)*T
               - 2.0 * cmath.log((1.0 - ((kappa - rho*sigma*1j*x - d)/(kappa - rho*sigma*1j*x + d))*cmath.exp(-d*T))
               / (1.0 - (kappa - rho*sigma*1j*x - d)/(kappa - rho*sigma*1j*x + d))))

        h = 1e-8
        d_val = (phase_func_real(x+h) - phase_func_real(x-h)) / (2*h)
        dd_val = (phase_func_real(x+h) - 2*phase_func_real(x) + phase_func_real(x-h)) / (h*h)
        return phase_func_real(x), d_val, dd_val

    def phase_func_real(x):
        d = cmath.sqrt((rho * sigma * 1j * x - kappa)**2 + sigma**2 * (1j * x + x*x))
        d = complex(d)
        if abs(d) < 1e-12:
            return 0.0
        g = (kappa - rho*sigma*1j*x - d) / (kappa - rho*sigma*1j*x + d)
        if abs(1.0 - g*cmath.exp(-d*T)) < 1e-12 or abs(1.0 - g) < 1e-12:
            return 0.0
        A = kappa*theta/(sigma**2)*((kappa - rho*sigma*1j*x - d)*T
            - 2.0*cmath.log((1.0 - g*cmath.exp(-d*T))/(1.0 - g)))
        return A.real

    x, ierr, iters = laguerre_rootfind(phase_func, 0.5, 6, abserr=1e-10, kmax=80)
    return x, ierr, iters






def is_prime(n):
    if n < 2:
        return False
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    for p in small_primes:
        if n % p == 0:
            return n == p

    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1

    for a in [2, 325, 9375, 28178, 450775, 9780504, 1795265022]:
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True


def next_prime(n):
    if n <= 2:
        return 2
    if n % 2 == 0:
        n += 1
    while not is_prime(n):
        n += 2
        if n > 1e12:
            raise RuntimeError("无法在给定范围内找到素数")
    return n






def quantile_statistics(data, probs=None):
    data = np.asarray(data, dtype=np.float64)
    if data.size == 0:
        raise ValueError("数据不能为空")
    if probs is None:
        probs = [0.01, 0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99]
    data_sorted = np.sort(data)
    n = len(data_sorted)
    result = {}
    for p in probs:
        idx = int(p * (n - 1))
        result[f'q{int(p*100):02d}'] = data_sorted[idx]

    result['VaR99'] = data_sorted[int(0.01 * (n - 1))]

    tail = data_sorted[:int(0.01 * n) + 1]
    result['CVaR99'] = np.mean(tail)

    result['mean'] = np.mean(data)
    result['std'] = np.std(data, ddof=1)
    result['skewness'] = np.mean(((data - result['mean']) / result['std'])**3) if result['std'] > 0 else 0.0
    result['kurtosis'] = np.mean(((data - result['mean']) / result['std'])**4) if result['std'] > 0 else 0.0

    q25 = result['q25']
    q75 = result['q75']
    iqr = q75 - q25
    lower = q25 - 1.5 * iqr
    upper = q75 + 1.5 * iqr
    outliers = data[(data < lower) | (data > upper)]
    result['outlier_count'] = len(outliers)
    result['outlier_ratio'] = len(outliers) / n
    return result


def box_plot_summary(data):
    return quantile_statistics(data)






def fast_structured_parse(lines, keyword_map):
    result = {key: [] for key in keyword_map}
    current_key = None
    for line in lines:
        line = line.strip().lower()
        if not line:
            continue
        first_word = line.split()[0] if line.split() else ""
        if first_word in keyword_map:
            current_key = first_word
        elif current_key is not None:
            parsed = keyword_map[current_key](line)
            if parsed is not None:
                result[current_key].append(parsed)
    return result


def parse_market_data_csv(text_lines):
    def parse_header(line):
        return line.split(',')

    def parse_row(line):
        parts = line.split(',')
        if len(parts) < 4:
            return None
        try:
            return {
                'strike': float(parts[0]),
                'maturity': float(parts[1]),
                'iv': float(parts[2]),
                'price': float(parts[3])
            }
        except (ValueError, IndexError):
            return None

    return fast_structured_parse(text_lines, {
        'header': parse_header,
        'data': parse_row
    })






class MeshDataManager:

    def __init__(self, dim, nodes, elements=None):
        self.dim = dim
        self.nodes = np.asarray(nodes, dtype=np.float64)
        self.node_num = self.nodes.shape[1] if self.nodes.ndim > 1 else self.nodes.shape[0]
        if elements is not None:
            self.elements = np.asarray(elements, dtype=np.int64)
            self.element_order = self.elements.shape[0]
            self.element_num = self.elements.shape[1]
        else:
            self.elements = None
            self.element_order = 0
            self.element_num = 0

    @staticmethod
    def generate_1d_uniform(x_min, x_max, n_nodes):
        nodes = np.linspace(x_min, x_max, n_nodes)
        elements = np.zeros((2, n_nodes - 1), dtype=np.int64)
        for e in range(n_nodes - 1):
            elements[0, e] = e
            elements[1, e] = e + 1
        return MeshDataManager(1, nodes.reshape(1, -1), elements)

    @staticmethod
    def generate_2d_tensor(x_nodes, y_nodes):
        nx = len(x_nodes)
        ny = len(y_nodes)
        node_num = nx * ny
        nodes = np.zeros((2, node_num), dtype=np.float64)
        idx = 0
        for j in range(ny):
            for i in range(nx):
                nodes[0, idx] = x_nodes[i]
                nodes[1, idx] = y_nodes[j]
                idx += 1

        element_num = (nx - 1) * (ny - 1) * 2
        elements = np.zeros((3, element_num), dtype=np.int64)
        e = 0
        for j in range(ny - 1):
            for i in range(nx - 1):
                n0 = j * nx + i
                n1 = n0 + 1
                n2 = n0 + nx
                n3 = n2 + 1
                elements[:, e] = [n0, n1, n2]
                e += 1
                elements[:, e] = [n1, n3, n2]
                e += 1
        return MeshDataManager(2, nodes, elements)

    def find_boundary_nodes_1d(self):
        if self.dim != 1:
            raise ValueError("仅适用于1D网格")
        return [0, self.node_num - 1]

    def find_boundary_nodes_2d_rect(self, nx, ny):
        if self.dim != 2:
            raise ValueError("仅适用于2D网格")
        boundary = set()
        for j in range(ny):
            for i in range(nx):
                idx = j * nx + i
                if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                    boundary.add(idx)
        return sorted(boundary)

    def export_mesh_dict(self):
        return {
            'dim': self.dim,
            'node_num': self.node_num,
            'nodes': self.nodes.tolist(),
            'element_order': self.element_order,
            'element_num': self.element_num,
            'elements': self.elements.tolist() if self.elements is not None else []
        }
