
import numpy as np

def polygon_moment_polygon(xv, yv, p, q):
    xv = np.asarray(xv, dtype=np.float64)
    yv = np.asarray(yv, dtype=np.float64)
    n = len(xv)
    if n < 3:
        return 0.0


    if xv[0] != xv[-1] or yv[0] != yv[-1]:
        xv = np.append(xv, xv[0])
        yv = np.append(yv, yv[0])
        n += 1

    moment = 0.0
    for i in range(n - 1):
        x0, y0 = xv[i], yv[i]
        x1, y1 = xv[i + 1], yv[i + 1]
        dx = x1 - x0



        if p == 0 and q == 0:
            I = 0.5 * (y0 + y1)
        elif p == 1 and q == 0:
            I = (x0 * y0 + x1 * y1) / 3.0 + (x0 * y1 + x1 * y0) / 6.0
        elif p == 0 and q == 1:
            I = (y0**2 + y0 * y1 + y1**2) / 3.0
        elif p == 2 and q == 0:
            I = ((x0**2 * y0 + x1**2 * y1) / 4.0 +
                 (x0**2 * y1 + x1**2 * y0) / 12.0 +
                 (x0 * x1 * (y0 + y1)) / 6.0)
        elif p == 1 and q == 1:
            I = ((x0 * y0**2 + x1 * y1**2) / 4.0 +
                 (x0 * y1**2 + x1 * y0**2) / 12.0 +
                 (y0 * y1 * (x0 + x1)) / 6.0)
        elif p == 0 and q == 2:
            I = (y0**3 + y0**2 * y1 + y0 * y1**2 + y1**3) / 4.0
        else:

            ns = max(p + q + 1, 5)
            s = np.linspace(0, 1, ns)
            xs = x0 + s * dx
            ys = y0 + s * (y1 - y0)
            I = np.trapezoid(xs**p * ys**q, s)

        moment += dx * I

    return moment


def polygon_area(xv, yv):
    return abs(polygon_moment_polygon(xv, yv, 0, 0))


def polygon_centroid(xv, yv):
    A = polygon_area(xv, yv)
    if abs(A) < 1e-14:
        return 0.0, 0.0
    xbar = polygon_moment_polygon(xv, yv, 1, 0) / A
    ybar = polygon_moment_polygon(xv, yv, 0, 1) / A
    return xbar, ybar


def polygon_second_moments(xv, yv):
    A = polygon_area(xv, yv)
    if abs(A) < 1e-14:
        return 0.0, 0.0, 0.0
    xbar, ybar = polygon_centroid(xv, yv)


    nu_20 = polygon_moment_polygon(xv, yv, 2, 0)
    nu_02 = polygon_moment_polygon(xv, yv, 0, 2)
    nu_11 = polygon_moment_polygon(xv, yv, 1, 1)

    mu_20 = (nu_20 - 2 * xbar * polygon_moment_polygon(xv, yv, 1, 0)
             + xbar**2 * A) / A
    mu_02 = (nu_02 - 2 * ybar * polygon_moment_polygon(xv, yv, 0, 1)
             + ybar**2 * A) / A
    mu_11 = (nu_11 - xbar * polygon_moment_polygon(xv, yv, 0, 1)
             - ybar * polygon_moment_polygon(xv, yv, 1, 0)
             + xbar * ybar * A) / A
    return mu_20, mu_02, mu_11


def polygon_ellipse_parameters(xv, yv):
    mu_20, mu_02, mu_11 = polygon_second_moments(xv, yv)
    trace = mu_20 + mu_02
    det = mu_20 * mu_02 - mu_11**2
    discriminant = np.sqrt(max((mu_20 - mu_02)**2 + 4.0 * mu_11**2, 0.0))
    lambda1 = 0.5 * (trace + discriminant)
    lambda2 = 0.5 * (trace - discriminant)
    lambda1 = max(lambda1, 0.0)
    lambda2 = max(lambda2, 0.0)
    a = 2.0 * np.sqrt(lambda1)
    b = 2.0 * np.sqrt(lambda2)
    if abs(mu_20 - mu_02) > 1e-14:
        theta = 0.5 * np.arctan2(2.0 * mu_11, mu_20 - mu_02)
    else:
        theta = np.pi / 4.0 if mu_11 > 0 else 0.0
    return a, b, theta


def exact_vorticity_integral_over_eddies(eddy_boundaries, vorticity_field, dx, dy):
    Ny, Nx = vorticity_field.shape
    x_grid = np.arange(Nx) * dx
    y_grid = np.arange(Ny) * dy
    integrals = []
    for boundary in eddy_boundaries:

        xv, yv = boundary[:, 0], boundary[:, 1]
        xmin, xmax = np.min(xv), np.max(xv)
        ymin, ymax = np.min(yv), np.max(yv)
        n_samples = 500
        samples_x = np.random.uniform(xmin, xmax, n_samples)
        samples_y = np.random.uniform(ymin, ymax, n_samples)
        inside = np.array([point_in_polygon_exact(sx, sy, boundary)
                           for sx, sy in zip(samples_x, samples_y)])
        if np.sum(inside) == 0:
            integrals.append(0.0)
            continue

        ix = np.clip(np.floor(samples_x[inside] / dx).astype(int), 0, Nx - 1)
        iy = np.clip(np.floor(samples_y[inside] / dy).astype(int), 0, Ny - 1)
        vort_samples = vorticity_field[iy, ix]
        area_box = (xmax - xmin) * (ymax - ymin)
        fraction = np.sum(inside) / n_samples
        area_est = area_box * fraction
        if area_est > 0:
            integrals.append(np.mean(vort_samples) * area_est)
        else:
            integrals.append(0.0)
    return integrals


def point_in_polygon_exact(x, y, poly):
    n = len(poly)
    inside = False
    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y + 1e-14) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside
