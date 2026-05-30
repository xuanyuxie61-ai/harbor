# -*- coding: utf-8 -*-

import numpy as np


class VolumeCorrector:

    def __init__(self, levelset, target_volume=None):
        self.ls = levelset
        if target_volume is None:
            self.target_volume = levelset.compute_volume()
        else:
            self.target_volume = target_volume

    def _volume_after_shift(self, lam):
        phi_shifted = self.ls.phi + lam
        vol = np.sum(phi_shifted < 0) * self.ls.dx * self.ls.dy
        return vol

    def _volume_residual(self, lam):
        return self._volume_after_shift(lam) - self.target_volume

    def correct_volume_brent(self, a=-1.0, b=1.0, tol=1e-10, max_iter=100):
        c = 0.5 * (3.0 - np.sqrt(5.0))
        eps = np.sqrt(np.finfo(float).eps)

        v = a + c * (b - a)
        w = v
        x = v
        e = 0.0
        fx = self._volume_residual(x)
        fv = fx
        fw = fx

        for it in range(max_iter):
            midpoint = 0.5 * (a + b)
            tol1 = eps * abs(x) + tol / 3.0
            tol2 = 2.0 * tol1

            if abs(x - midpoint) <= (tol2 - 0.5 * (b - a)):
                break

            if abs(e) <= tol1:
                if midpoint <= x:
                    e = a - x
                else:
                    e = b - x
                d = c * e
            else:
                r = (x - w) * (fx - fv)
                q = (x - v) * (fx - fw)
                p_val = (x - v) * q - (x - w) * r
                q = 2.0 * (q - r)
                if q > 0:
                    p_val = -p_val
                q = abs(q)
                r = e
                e = d

                if abs(0.5 * q * r) <= abs(p_val) or p_val <= q * (a - x) or q * (b - x) <= p_val:
                    if midpoint <= x:
                        e = a - x
                    else:
                        e = b - x
                    d = c * e
                else:
                    d = p_val / q
                    u = x + d
                    if (u - a) < tol2:
                        d = abs(tol1) * np.sign(midpoint - x)
                    if (b - u) < tol2:
                        d = abs(tol1) * np.sign(midpoint - x)

            if tol1 <= abs(d):
                u = x + d
            elif abs(d) < tol1:
                u = x + abs(tol1) * np.sign(d)
            else:
                u = x + abs(tol1) * np.sign(d)

            fu = self._volume_residual(u)

            if fu <= fx:
                if x <= u:
                    a = x
                else:
                    b = x
                v = w
                fv = fw
                w = x
                fw = fx
                x = u
                fx = fu
            else:
                if u < x:
                    a = u
                else:
                    b = u
                if fu <= fw or w == x:
                    v = w
                    fv = fw
                    w = u
                    fw = fu
                elif fu <= fv or v == x or v == w:
                    v = u
                    fv = fu

        lam_opt = x
        self.ls.phi += lam_opt
        return lam_opt, it + 1

    def correct_volume_simple(self):
        lo, hi = -2.0, 2.0

        for _ in range(5):
            if self._volume_residual(lo) < 0:
                lo *= 2.0
            if self._volume_residual(hi) > 0:
                hi *= 2.0

        for _ in range(80):
            mid = 0.5 * (lo + hi)
            fmid = self._volume_residual(mid)
            if fmid > 0:

                lo = mid
            else:

                hi = mid
        lam_opt = 0.5 * (lo + hi)
        self.ls.phi += lam_opt
        return lam_opt


class ExternalForcing:

    @staticmethod
    def ripple_like_forcing(X, Y, t, A=0.1):
        r2 = X ** 2 + Y ** 2
        r2 = np.clip(r2, 1e-8, None)
        return A * np.sin(t * r2)

    @staticmethod
    def oscillatory_normal_forcing(X, Y, t, A=0.05, omega=2.0, kx=5.0, ky=5.0):
        return A * np.sin(omega * t) * np.sin(kx * X) * np.sin(ky * Y)

    @staticmethod
    def gravitational_forcing(X, Y, g=1.0, angle=0.0):
        return g * (X * np.cos(angle) + Y * np.sin(angle))

    @staticmethod
    def combined_forcing(X, Y, t, A1=0.05, A2=0.03, omega=2.0):
        f1 = ExternalForcing.oscillatory_normal_forcing(X, Y, t, A=A1, omega=omega)
        f2 = ExternalForcing.ripple_like_forcing(X, Y, t, A=A2)
        return f1 + f2
