# -*- coding: utf-8 -*-

import numpy as np
from typing import Union


class SincInterpolator:

    @staticmethod
    def sincn(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        x = np.asarray(x, dtype=float)
        result = np.ones_like(x)
        mask = np.abs(x) > 1e-15
        if np.any(mask):
            x_mask = x[mask]
            result[mask] = np.sin(np.pi * x_mask) / (np.pi * x_mask)
        return result if result.shape != () else float(result)

    @staticmethod
    def sincn_derivative(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        x = np.asarray(x, dtype=float)
        result = np.zeros_like(x)
        mask = np.abs(x) > 1e-15
        if np.any(mask):
            x_m = x[mask]
            pi_x = np.pi * x_m
            result[mask] = (pi_x * np.cos(pi_x) - np.sin(pi_x)) / (pi_x * x_m)
        return result if result.shape != () else float(result)

    @staticmethod
    def sincn_second_derivative(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        x = np.asarray(x, dtype=float)
        result = np.full_like(x, -np.pi ** 2 / 3.0)
        mask = np.abs(x) > 1e-15
        if np.any(mask):
            x_m = x[mask]
            pi_x = np.pi * x_m
            result[mask] = ((2.0 - pi_x ** 2) * np.sin(pi_x) - 2.0 * pi_x * np.cos(pi_x)) / (pi_x * x_m ** 2)
        return result if result.shape != () else float(result)

    @staticmethod
    def interpolate_1d(x_grid: np.ndarray, f_values: np.ndarray,
                       x_query: np.ndarray, truncation: int = None) -> np.ndarray:
        if len(x_grid) < 2:
            raise ValueError("x_grid 至少需要 2 个点")
        if len(x_grid) != len(f_values):
            raise ValueError("x_grid 和 f_values 长度必须相同")


        h = x_grid[1] - x_grid[0]
        if not np.allclose(np.diff(x_grid), h, rtol=1e-10):
            raise ValueError("x_grid 必须是均匀网格")

        N = len(x_grid)
        if truncation is None:
            truncation = N

        f_interp = np.zeros(len(x_query))

        for qi, xq in enumerate(x_query):

            k0 = int(round((xq - x_grid[0]) / h))
            k_min = max(0, k0 - truncation)
            k_max = min(N - 1, k0 + truncation)

            s = 0.0
            for k in range(k_min, k_max + 1):
                s += f_values[k] * SincInterpolator.sincn((xq - x_grid[k]) / h)
            f_interp[qi] = s

        return f_interp

    @staticmethod
    def interpolate_2d(x_grid: np.ndarray, y_grid: np.ndarray,
                       f_values: np.ndarray, x_query: np.ndarray,
                       y_query: np.ndarray, truncation: int = None) -> np.ndarray:
        if f_values.ndim != 2:
            raise ValueError("f_values 必须为二维数组")

        ny, nx = f_values.shape
        if len(x_grid) != nx or len(y_grid) != ny:
            raise ValueError("网格维度不匹配")

        hx = x_grid[1] - x_grid[0]
        hy = y_grid[1] - y_grid[0]

        if truncation is None:
            truncation = max(nx, ny)

        x_query = np.atleast_1d(x_query)
        y_query = np.atleast_1d(y_query)

        result = np.zeros((len(y_query), len(x_query)))

        for j, yq in enumerate(y_query):
            for i, xq in enumerate(x_query):
                kx0 = int(round((xq - x_grid[0]) / hx))
                ky0 = int(round((yq - y_grid[0]) / hy))

                kx_min = max(0, kx0 - truncation)
                kx_max = min(nx - 1, kx0 + truncation)
                ky_min = max(0, ky0 - truncation)
                ky_max = min(ny - 1, ky0 + truncation)

                s = 0.0
                for ky in range(ky_min, ky_max + 1):
                    sy = SincInterpolator.sincn((yq - y_grid[ky]) / hy)
                    for kx in range(kx_min, kx_max + 1):
                        sx = SincInterpolator.sincn((xq - x_grid[kx]) / hx)
                        s += f_values[ky, kx] * sx * sy
                result[j, i] = s

        return result

    @staticmethod
    def derivative_1d(x_grid: np.ndarray, f_values: np.ndarray,
                      x_query: np.ndarray) -> np.ndarray:
        if len(x_grid) < 2:
            raise ValueError("x_grid 至少需要 2 个点")

        h = x_grid[1] - x_grid[0]
        N = len(x_grid)

        df = np.zeros(len(x_query))
        for qi, xq in enumerate(x_query):
            s = 0.0
            for k in range(N):
                s += f_values[k] * SincInterpolator.sincn_derivative((xq - x_grid[k]) / h) / h
            df[qi] = s

        return df
