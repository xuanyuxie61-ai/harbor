
import numpy as np
from math import sqrt, log








_NIEDERREITER_CJ = None
_NIEDERREITER_DIM = None
_NIEDERREITER_NEXTQ = None
_NIEDERREITER_KEY = None


def _calcc2(dim):
    nbits = 31
    maxdim = 20
    
    if dim <= 0 or dim > maxdim:
        raise ValueError(f"Dimension must be between 1 and {maxdim}")
    

    degrees = [1, 2, 3, 3, 4, 4, 5, 5, 5, 5,
               6, 6, 6, 6, 6, 6, 7, 7, 7, 7]
    

    poly = [3, 7, 11, 13, 25, 37, 59, 47, 61, 55,
            87, 91, 103, 115, 117, 173, 199, 185, 227, 217]
    
    cj = np.zeros((dim, nbits), dtype=np.int64)
    
    for i in range(dim):
        deg = degrees[i]
        p = poly[i]
        

        nextq = np.zeros(nbits + 1, dtype=np.int64)
        for j in range(deg):
            nextq[j] = 0
        nextq[deg] = 1
        
        for j in range(deg + 1, nbits + 1):


            nextq[j] = 0
            for k in range(1, deg + 1):
                if (p >> k) & 1:
                    nextq[j] ^= nextq[j - k]
        

        for r in range(nbits):
            cj[i, r] = nextq[r + 1]
    
    return cj


def niederreiter2_init(dim):
    global _NIEDERREITER_CJ, _NIEDERREITER_DIM, _NIEDERREITER_NEXTQ, _NIEDERREITER_KEY
    
    _NIEDERREITER_DIM = dim
    _NIEDERREITER_KEY = -1
    _NIEDERREITER_CJ = _calcc2(dim)
    _NIEDERREITER_NEXTQ = np.zeros(dim, dtype=np.int64)


def niederreiter2_generate(key):
    global _NIEDERREITER_CJ, _NIEDERREITER_DIM, _NIEDERREITER_NEXTQ, _NIEDERREITER_KEY
    
    if _NIEDERREITER_DIM is None:
        raise RuntimeError("Niederreiter generator not initialized")
    
    dim = _NIEDERREITER_DIM
    nbits = 31
    recip = 2.0 ** (-nbits)
    

    if key != _NIEDERREITER_KEY + 1:
        gray = key ^ (key >> 1)
        _NIEDERREITER_NEXTQ = np.zeros(dim, dtype=np.int64)
        r = 0
        while gray > 0:
            if gray & 1:
                for i in range(dim):
                    _NIEDERREITER_NEXTQ[i] ^= _NIEDERREITER_CJ[i, r]
            gray >>= 1
            r += 1
    

    quasi = _NIEDERREITER_NEXTQ.astype(float) * recip
    

    r = 0
    i = key
    while i & 1:
        r += 1
        i >>= 1
    
    if r >= nbits:
        raise RuntimeError("Too many calls to Niederreiter generator")
    

    for i in range(dim):
        _NIEDERREITER_NEXTQ[i] ^= _NIEDERREITER_CJ[i, r]
    
    _NIEDERREITER_KEY = key
    return quasi, key + 1


def generate_niederreiter_sequence(dim, n):
    niederreiter2_init(dim)
    points = np.zeros((n, dim))
    key = 0
    for i in range(n):
        points[i], key = niederreiter2_generate(key)
    return points






def _radical_inverse(n, base):
    result = 0.0
    f = 1.0 / base
    while n > 0:
        result += f * (n % base)
        n //= base
        f /= base
    return result


def hammersley_sequence(i1, i2, m, n_base):
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
              31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
    
    if i1 <= i2:
        i3 = 1
    else:
        i3 = -1
    
    if n_base <= 0:
        n_base = 1
    
    l = abs(i2 - i1) + 1
    points = np.zeros((m, l))
    k = 0
    
    for i in range(i1, i2 + i3, i3):
        points[0, k] = (i % (n_base + 1)) / n_base if n_base > 0 else 0.0
        for j in range(1, m):
            if j - 1 < len(primes):
                points[j, k] = _radical_inverse(i, primes[j - 1])
        k += 1
    
    return points


def estimate_area_qmc(polygon_vertices, bbox, n_samples=5000):
    vertices = np.asarray(polygon_vertices)
    xmin, xmax, ymin, ymax = bbox
    
    if xmax <= xmin or ymax <= ymin or n_samples <= 0:
        return 0.0, 0.0
    

    hammersley = hammersley_sequence(0, n_samples - 1, 2, n_samples - 1)
    

    x = xmin + (xmax - xmin) * hammersley[0, :]
    y = ymin + (ymax - ymin) * hammersley[1, :]
    

    inside_count = 0
    n_vert = len(vertices)
    
    for k in range(n_samples):
        qx, qy = x[k], y[k]
        inside = False
        j = n_vert - 1
        for i in range(n_vert):
            xi, yi = vertices[i]
            xj, yj = vertices[j]
            if ((yi < qy <= yj) or (qy <= yi and yj < qy)):
                if qx <= (xj - xi) * (qy - yi) / (yj - yi) + xi:
                    inside = not inside
            j = i
        if inside:
            inside_count += 1
    
    area_bbox = (xmax - xmin) * (ymax - ymin)
    area_estimate = (inside_count / n_samples) * area_bbox
    
    return area_estimate, area_bbox






def sample_conductivity_parameters(n_samples, method='niederreiter'):
    if method == 'niederreiter':
        points = generate_niederreiter_sequence(3, n_samples)
    else:
        points = hammersley_sequence(0, n_samples - 1, 3, n_samples - 1)
        points = points.T
    

    sigma_f_min, sigma_f_max = 0.2, 0.6
    sigma_t_min, sigma_t_max = 0.02, 0.08
    sigma_n_min, sigma_n_max = 0.01, 0.04
    
    samples = np.zeros((n_samples, 3))
    samples[:, 0] = sigma_f_min + (sigma_f_max - sigma_f_min) * points[:, 0]
    samples[:, 1] = sigma_t_min + (sigma_t_max - sigma_t_min) * points[:, 1]
    samples[:, 2] = sigma_n_min + (sigma_n_max - sigma_n_min) * points[:, 2]
    
    return samples


def compute_discrepancy(points):
    n, dim = points.shape
    if n == 0:
        return 1.0
    

    d2 = 0.0
    for i in range(n):
        for j in range(n):
            prod = 1.0
            for k in range(dim):
                prod *= (1.0 - max(points[i, k], points[j, k]))
            d2 += prod
    
    d2 = d2 / (n ** 2)
    

    term2 = 0.0
    for i in range(n):
        prod = 1.0
        for k in range(dim):
            prod *= (1.0 - points[i, k] ** 2) / 2.0
        term2 += prod
    term2 = term2 * (2.0 ** (1 - dim)) / n
    
    term3 = 3.0 ** (-dim)
    
    l2_disc = sqrt(d2 - term2 + term3)
    return l2_disc
