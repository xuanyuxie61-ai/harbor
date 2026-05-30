
import numpy as np
from typing import List, Tuple


class Polygon2D:

    def __init__(self, vertices: np.ndarray):
        verts = np.asarray(vertices, dtype=complex)
        if verts.ndim == 2 and verts.shape[1] == 2:
            self.complex_vertices = verts[:, 0] + 1j * verts[:, 1]
        else:
            self.complex_vertices = verts
        self.n = len(self.complex_vertices)

        self.xmin = np.min(self.complex_vertices.real)
        self.xmax = np.max(self.complex_vertices.real)
        self.ymin = np.min(self.complex_vertices.imag)
        self.ymax = np.max(self.complex_vertices.imag)

    def contains(self, x: float, y: float, tol: float = 1e-10) -> bool:
        if x < self.xmin - tol or x > self.xmax + tol or y < self.ymin - tol or y > self.ymax + tol:
            return False

        cn = 0
        n = self.n
        xv = self.complex_vertices.real
        yv = self.complex_vertices.imag
        for i in range(n):
            j = (i + 1) % n

            if ((yv[i] <= y < yv[j]) or (yv[j] <= y < yv[i])):

                vt = (y - yv[i]) / (yv[j] - yv[i] + 1e-18)
                x_intersect = xv[i] + vt * (xv[j] - xv[i])
                if x < x_intersect:
                    cn += 1
        return (cn % 2) == 1

    def area(self) -> float:
        xv = self.complex_vertices.real
        yv = self.complex_vertices.imag
        return 0.5 * np.sum(xv[:-1] * yv[1:] - xv[1:] * yv[:-1])

    def centroid(self) -> Tuple[float, float]:
        xv = self.complex_vertices.real
        yv = self.complex_vertices.imag
        a = self.area()
        if abs(a) < 1e-14:
            return (float(np.mean(xv)), float(np.mean(yv)))
        cx = np.sum((xv[:-1] + xv[1:]) * (xv[:-1] * yv[1:] - xv[1:] * yv[:-1])) / (6.0 * a)
        cy = np.sum((yv[:-1] + yv[1:]) * (xv[:-1] * yv[1:] - xv[1:] * yv[:-1])) / (6.0 * a)
        return (cx, cy)


def rotate_complex(z: np.ndarray, theta: float) -> np.ndarray:
    return z * np.exp(1j * theta)


def translate_complex(z: np.ndarray, dx: float, dy: float) -> np.ndarray:
    return z + (dx + 1j * dy)


def reflect_complex(z: np.ndarray, axis: str = "x") -> np.ndarray:
    if axis == "x":
        return np.conj(z)
    elif axis == "y":
        return -np.conj(z)
    return z


class BatteryCellGeometry:

    def __init__(self, total_width: float = 1.0, total_height: float = 0.5,
                 neg_cc_width: float = 0.05, neg_elec_width: float = 0.3,
                 sep_width: float = 0.05, pos_elec_width: float = 0.3,
                 pos_cc_width: float = 0.05):
        self.total_width = total_width
        self.total_height = total_height
        self.neg_cc_width = neg_cc_width
        self.neg_elec_width = neg_elec_width
        self.sep_width = sep_width
        self.pos_elec_width = pos_elec_width
        self.pos_cc_width = pos_cc_width


        h = total_height
        x0 = 0.0
        self.neg_cc = self._make_rect(x0, x0 + neg_cc_width, 0.0, h)
        x0 += neg_cc_width
        self.neg_elec = self._make_rect(x0, x0 + neg_elec_width, 0.0, h)
        x0 += neg_elec_width
        self.separator = self._make_rect(x0, x0 + sep_width, 0.0, h)
        x0 += sep_width
        self.pos_elec = self._make_rect(x0, x0 + pos_elec_width, 0.0, h)
        x0 += pos_elec_width
        self.pos_cc = self._make_rect(x0, x0 + pos_cc_width, 0.0, h)


        tab_w = 0.04
        tab_h = 0.04

        self.neg_tab = self._make_rect(
            neg_cc_width * 0.5 - tab_w * 0.5,
            neg_cc_width * 0.5 + tab_w * 0.5,
            h, h + tab_h
        )

        self.pos_tab = self._make_rect(
            total_width - pos_cc_width * 0.5 - tab_w * 0.5,
            total_width - pos_cc_width * 0.5 + tab_w * 0.5,
            h, h + tab_h
        )

    def _make_rect(self, x1: float, x2: float, y1: float, y2: float) -> Polygon2D:
        verts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=float)
        return Polygon2D(verts)

    def classify_point(self, x: float, y: float) -> str:
        if self.neg_cc.contains(x, y):
            return "neg_cc"
        if self.neg_elec.contains(x, y):
            return "neg_elec"
        if self.separator.contains(x, y):
            return "separator"
        if self.pos_elec.contains(x, y):
            return "pos_elec"
        if self.pos_cc.contains(x, y):
            return "pos_cc"
        if self.neg_tab.contains(x, y):
            return "neg_tab"
        if self.pos_tab.contains(x, y):
            return "pos_tab"
        return "outside"

    def get_all_regions(self) -> List[Tuple[str, Polygon2D]]:
        return [
            ("neg_cc", self.neg_cc),
            ("neg_elec", self.neg_elec),
            ("separator", self.separator),
            ("pos_elec", self.pos_elec),
            ("pos_cc", self.pos_cc),
            ("neg_tab", self.neg_tab),
            ("pos_tab", self.pos_tab),
        ]
