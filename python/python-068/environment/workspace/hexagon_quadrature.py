
import numpy as np


def hexagon_stroud_rule1():
    x = np.array([0.0])
    y = np.array([0.0])
    w = np.array([3.0 * np.sqrt(3.0) / 2.0])
    return x, y, w


def hexagon_stroud_rule2():
    a = np.sqrt(5.0 / 12.0)
    x = np.array([a, -a, 0.0, 0.0])
    y = np.array([0.0, 0.0, a, -a])
    w = np.array([3.0 * np.sqrt(3.0) / 8.0] * 4)
    return x, y, w


def hexagon_stroud_rule3():
    x = np.array([0.0, 1.0, 0.5, -0.5, -1.0, -0.5, 0.5])
    y = np.array([0.0, 0.0, np.sqrt(3.0) / 2.0, np.sqrt(3.0) / 2.0,
                  0.0, -np.sqrt(3.0) / 2.0, -np.sqrt(3.0) / 2.0])
    wc = 3.0 * np.sqrt(3.0) / 8.0
    wv = 3.0 * np.sqrt(3.0) / 16.0
    w = np.array([wc, wv, wv, wv, wv, wv, wv])
    return x, y, w


def hexagon_stroud_rule4():
    a = np.sqrt(14.0) / 5.0
    x = np.array([0.0, a, a / 2.0, -a / 2.0, -a, -a / 2.0, a / 2.0])
    y = np.array([0.0, 0.0, a * np.sqrt(3.0) / 2.0, a * np.sqrt(3.0) / 2.0,
                  0.0, -a * np.sqrt(3.0) / 2.0, -a * np.sqrt(3.0) / 2.0])
    wc = 258.0 / 1008.0 * 3.0 * np.sqrt(3.0) / 2.0
    wi = 125.0 / 1008.0 * 3.0 * np.sqrt(3.0) / 2.0
    w = np.array([wc, wi, wi, wi, wi, wi, wi])
    return x, y, w


def hexagon_monomial_integral(p: int, q: int) -> float:
    if p < 0 or q < 0:
        return 0.0
    if p % 2 == 1 or q % 2 == 1:
        return 0.0

    x, y, w = hexagon_stroud_rule4()
    vals = (x ** p) * (y ** q)
    return float(np.sum(w * vals))


def integrate_over_hexagonal_patch(
    field: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    hex_center: tuple[float, float],
    hex_radius: float,
    rule_id: int = 4
) -> float:
    rules = {
        1: hexagon_stroud_rule1,
        2: hexagon_stroud_rule2,
        3: hexagon_stroud_rule3,
        4: hexagon_stroud_rule4,
    }
    xq, yq, wq = rules[rule_id]()


    xq_phys = hex_center[0] + hex_radius * xq
    yq_phys = hex_center[1] + hex_radius * yq


    wq_scaled = wq * (hex_radius ** 2)


    nx, ny = field.shape
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    total = 0.0
    for i in range(len(xq)):
        xi = xq_phys[i]
        yi = yq_phys[i]

        ix = (xi - x_min) / (x_max - x_min) * (nx - 1)
        iy = (yi - y_min) / (y_max - y_min) * (ny - 1)
        ix0 = int(np.floor(np.clip(ix, 0, nx - 2)))
        iy0 = int(np.floor(np.clip(iy, 0, ny - 2)))
        dx = ix - ix0
        dy = iy - iy0

        val = (
            field[ix0, iy0] * (1 - dx) * (1 - dy) +
            field[ix0 + 1, iy0] * dx * (1 - dy) +
            field[ix0, iy0 + 1] * (1 - dx) * dy +
            field[ix0 + 1, iy0 + 1] * dx * dy
        )

        val = max(val, 0.0)
        total += wq_scaled[i] * val

    return float(total)


def compute_hexagonal_patch_metrics(
    fields: dict[str, np.ndarray],
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    hex_centers: list[tuple[float, float]],
    hex_radius: float
) -> dict[str, list[float]]:
    metrics = {name: [] for name in fields.keys()}
    for center in hex_centers:
        for name, field in fields.items():
            val = integrate_over_hexagonal_patch(field, x_coords, y_coords, center, hex_radius)
            metrics[name].append(val)
    return metrics
