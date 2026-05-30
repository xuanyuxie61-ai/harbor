
import numpy as np
from typing import Union


class SpecialFunctions:


    EULER_GAMMA = 0.5772156649015329
    PI_HALF = 1.570796326794897
    EPS = 1.0e-15

    @staticmethod
    def ci(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        scalar_input = np.isscalar(x)
        x = np.atleast_1d(np.asarray(x, dtype=float))
        result = np.empty_like(x)


        zero_mask = np.abs(x) < 1e-18
        result[zero_mask] = -np.inf


        small_mask = (~zero_mask) & (np.abs(x) <= 16.0)
        if np.any(small_mask):
            xs = x[small_mask]
            x2 = xs ** 2
            xr = -0.25 * x2
            val = SpecialFunctions.EULER_GAMMA + np.log(np.abs(xs)) + xr
            for k in range(2, 60):
                xr = -0.5 * xr * (k - 1.0) / (k * k * (2.0 * k - 1.0)) * x2
                val += xr
                if np.all(np.abs(xr) < np.abs(val) * SpecialFunctions.EPS):
                    break
            result[small_mask] = val


        medium_mask = (~zero_mask) & (np.abs(x) > 16.0) & (np.abs(x) <= 32.0)
        if np.any(medium_mask):
            xm = x[medium_mask]
            xabs = np.abs(xm)
            m = np.floor(47.2 + 0.82 * xabs).astype(int)
            m = np.clip(m, 10, 500)

            vals = np.zeros(len(xm))
            for idx in range(len(xm)):
                ma = m[idx]
                xa1 = 0.0
                xa0 = 1.0e-100
                bj = np.zeros(ma)
                for k in range(ma, 0, -1):
                    xa = 4.0 * k * xa0 / xabs[idx] - xa1
                    bj[k - 1] = xa
                    xa1 = xa0
                    xa0 = xa
                xs_sum = bj[0]
                for k in range(2, ma, 2):
                    xs_sum += 2.0 * bj[k]
                if abs(xs_sum) < 1e-18:
                    xs_sum = 1.0
                bj = bj / xs_sum

                xr = 1.0
                xg1 = bj[0]
                for k in range(1, ma):
                    xr = (0.25 * xr * (2.0 * k - 1.0) ** 2
                          / ((k) * (2.0 * k - 1.0) ** 2) * xabs[idx])
                    xg1 += bj[k] * xr

                xr = 1.0
                xg2 = bj[0]
                for k in range(1, ma):
                    xr = (0.25 * xr * (2.0 * k - 5.0) ** 2
                          / ((k) * (2.0 * k - 3.0) ** 2) * xabs[idx])
                    xg2 += bj[k] * xr

                xcs = np.cos(xabs[idx] / 2.0)
                xss = np.sin(xabs[idx] / 2.0)
                vals[idx] = (SpecialFunctions.EULER_GAMMA + np.log(xabs[idx])
                             - xabs[idx] * xss * xg1
                             + 2.0 * xcs * xg2
                             - 2.0 * xcs * xcs)
            result[medium_mask] = vals


        large_mask = (~zero_mask) & (np.abs(x) > 32.0)
        if np.any(large_mask):
            xl = x[large_mask]
            x2 = xl ** 2
            xabs = np.abs(xl)

            xr = 1.0
            xf = 1.0
            for k in range(1, 10):
                xr = -2.0 * xr * k * (2 * k - 1) / x2
                xf += xr

            xr = 1.0 / xabs
            xg = xr.copy()
            for k in range(1, 9):
                xr = -2.0 * xr * (2 * k + 1) * k / x2
                xg += xr

            result[large_mask] = (xf * np.sin(xabs) / xabs
                                  - xg * np.cos(xabs) / xabs)

        if scalar_input:
            return float(result[0])
        return result

    @staticmethod
    def si(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        scalar_input = np.isscalar(x)
        x = np.atleast_1d(np.asarray(x, dtype=float))

        val = x.copy()
        term = x.copy()
        x2 = x ** 2

        for k in range(1, 80):
            term *= -x2 / ((2.0 * k) * (2.0 * k + 1.0) ** 2 / (2.0 * k - 1.0))

            term = -term * x2 / ((2.0 * k) * (2.0 * k + 1.0))

            term = ((-1.0) ** k) * (x ** (2 * k + 1)) / ((2 * k + 1) * np.math.factorial(2 * k + 1))



            break


        result = np.zeros_like(x)
        for i in range(len(x)):
            xv = x[i]
            if abs(xv) < 1e-12:
                result[i] = 0.0
                continue
            s = xv
            term = xv
            for k in range(1, 100):
                term *= -xv * xv / ((2.0 * k) * (2.0 * k + 1.0))
                s += term
                if abs(term) < abs(s) * SpecialFunctions.EPS:
                    break
            result[i] = s

        if scalar_input:
            return float(result[0])
        return result

    @staticmethod
    def normal_cdf(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        from math import erf
        if np.isscalar(x):
            if x < -8.0:
                return 0.0
            if x > 8.0:
                return 1.0
            return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))

        x_arr = np.asarray(x, dtype=float)
        result = np.empty_like(x_arr)


        result[x_arr <= -8.0] = 0.0
        result[x_arr >= 8.0] = 1.0

        mid_mask = (x_arr > -8.0) & (x_arr < 8.0)
        if np.any(mid_mask):
            xm = x_arr[mid_mask]


            result[mid_mask] = SpecialFunctions._hart_cdf(xm)

        return result

    @staticmethod
    def _hart_cdf(x: np.ndarray) -> np.ndarray:


        try:
            from scipy.special import erf
            return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))
        except ImportError:

            abs_x = np.abs(x)
            t = 1.0 / (1.0 + 0.2316419 * abs_x)
            poly = (0.319381530 * t
                    - 0.356563782 * t ** 2
                    + 1.781477937 * t ** 3
                    - 1.821255978 * t ** 4
                    + 1.330274429 * t ** 5)
            pdf = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * abs_x ** 2)
            cdf = 1.0 - pdf * poly
            result = np.where(x < 0, 1.0 - cdf, cdf)
            return result

    @staticmethod
    def black_scholes_delta(S: float, K: float, T: float,
                            r: float, sigma: float,
                            option_type: str = "call") -> float:
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            return 0.0

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        delta = SpecialFunctions.normal_cdf(d1)
        if option_type == "put":
            delta -= 1.0
        return float(delta)

    @staticmethod
    def kernel_ci(x: np.ndarray, h: float) -> np.ndarray:
        if h <= 0:
            raise ValueError("带宽 h 必须为正.")
        u = np.abs(x) / h

        raw = SpecialFunctions.ci(u) + SpecialFunctions.EULER_GAMMA + np.log(u + 1e-18)
        raw = np.maximum(raw, 0.0)

        C = np.trapz(raw, x) if len(x) > 1 else 1.0
        if C > 0:
            raw /= C
        return raw
