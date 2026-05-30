
import numpy as np


def double_grid_nearest(field):
    if field.ndim == 2:
        m, n = field.shape
        field2 = np.zeros((2 * m, 2 * n), dtype=field.dtype)
        for i in range(m):
            for j in range(n):
                field2[2 * i : 2 * i + 2, 2 * j : 2 * j + 2] = field[i, j]
        return field2
    elif field.ndim == 3:
        m, n, p = field.shape
        field2 = np.zeros((2 * m, 2 * n, 2 * p), dtype=field.dtype)
        for i in range(m):
            for j in range(n):
                for k in range(p):
                    field2[
                        2 * i : 2 * i + 2,
                        2 * j : 2 * j + 2,
                        2 * k : 2 * k + 2,
                    ] = field[i, j, k]
        return field2
    else:
        raise ValueError("Only 2D and 3D fields supported.")


def double_grid_bilinear(field):
    if field.ndim != 2:
        raise ValueError("Bilinear doubling only supports 2D fields.")
    m, n = field.shape
    field2 = np.zeros((2 * m - 1, 2 * n - 1), dtype=float)

    for i in range(m):
        for j in range(n):
            field2[2 * i, 2 * j] = field[i, j]


    for i in range(m):
        for j in range(n - 1):
            field2[2 * i, 2 * j + 1] = 0.5 * (field[i, j] + field[i, j + 1])


    for i in range(m - 1):
        for j in range(n):
            field2[2 * i + 1, 2 * j] = 0.5 * (field[i, j] + field[i + 1, j])


    for i in range(m - 1):
        for j in range(n - 1):
            field2[2 * i + 1, 2 * j + 1] = 0.25 * (
                field[i, j] + field[i + 1, j] + field[i, j + 1] + field[i + 1, j + 1]
            )

    return field2


def adaptive_refinement_2d(
    field, threshold, max_level=3, refinement_func="bilinear"
):
    refined_fields = [field.copy()]
    current = field.copy()

    for _ in range(max_level):
        m, n = current.shape
        if m < 3 or n < 3:
            break


        gx = np.zeros_like(current)
        gy = np.zeros_like(current)
        gx[:, :-1] = current[:, 1:] - current[:, :-1]
        gy[:-1, :] = current[1:, :] - current[:-1, :]
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)


        if np.max(grad_mag) < threshold:
            break

        if refinement_func == "bilinear":
            current = double_grid_bilinear(current)
        else:
            current = double_grid_nearest(current)
        refined_fields.append(current)

    return refined_fields


def integrate_field_2d(field, dx, dy):
    return np.trapz(np.trapz(field, dx=dx, axis=0), dx=dy)


def integrate_field_3d(field, dx, dy, dz):
    temp = np.trapz(field, dx=dx, axis=0)
    temp = np.trapz(temp, dx=dy, axis=0)
    return np.trapz(temp, dx=dz)
