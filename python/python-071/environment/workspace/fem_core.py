# -*- coding: utf-8 -*-

import numpy as np
import math


def basis_mn_t3(t, n, p):
    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)
    if p.ndim == 1:
        p = p.reshape(2, 1)
    elif p.shape[0] != 2 and p.shape[1] == 2:
        p = p.T

    area = t[0, 0] * (t[1, 1] - t[1, 2]) \
         + t[0, 1] * (t[1, 2] - t[1, 0]) \
         + t[0, 2] * (t[1, 0] - t[1, 1])


    if abs(area) < 1e-15:
        area = 1e-15 * np.sign(area) if area != 0 else 1e-15

    phi = np.zeros((3, n), dtype=float)
    dphidx = np.zeros((3, n), dtype=float)
    dphidy = np.zeros((3, n), dtype=float)

    phi[0, :] = (t[0, 2] - t[0, 1]) * (p[1, :] - t[1, 1]) \
              - (t[1, 2] - t[1, 1]) * (p[0, :] - t[0, 1])
    dphidx[0, :] = -(t[1, 2] - t[1, 1])
    dphidy[0, :] =  (t[0, 2] - t[0, 1])

    phi[1, :] = (t[0, 0] - t[0, 2]) * (p[1, :] - t[1, 2]) \
              - (t[1, 0] - t[1, 2]) * (p[0, :] - t[0, 2])
    dphidx[1, :] = -(t[1, 0] - t[1, 2])
    dphidy[1, :] =  (t[0, 0] - t[0, 2])

    phi[2, :] = (t[0, 1] - t[0, 0]) * (p[1, :] - t[1, 0]) \
              - (t[1, 1] - t[1, 0]) * (p[0, :] - t[0, 0])
    dphidx[2, :] = -(t[1, 1] - t[1, 0])
    dphidy[2, :] =  (t[0, 1] - t[0, 0])

    phi /= area
    dphidx /= area
    dphidy /= area

    return phi, dphidx, dphidy


def basis_mn_t6(t, n, p):
    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)
    if p.ndim == 1:
        p = p.reshape(2, 1)

    phi = np.zeros((6, n), dtype=float)
    dphidx = np.zeros((6, n), dtype=float)
    dphidy = np.zeros((6, n), dtype=float)


    def compute_basis(idx, p1, p2, p3, p4, p5, p6):
        gx = (p[0, :] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p[1, :] - p1[1])
        gn = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])
        hx = (p[0, :] - p4[0]) * (p6[1] - p4[1]) - (p6[0] - p4[0]) * (p[1, :] - p4[1])
        hn = (p5[0] - p4[0]) * (p6[1] - p4[1]) - (p6[0] - p4[0]) * (p5[1] - p4[1])

        gn = np.where(np.abs(gn) < 1e-15, 1e-15, gn)
        hn = np.where(np.abs(hn) < 1e-15, 1e-15, hn)

        ph = (gx * hx) / (gn * hn)
        dpx = ((p3[1] - p1[1]) * hx + gx * (p6[1] - p4[1])) / (gn * hn)
        dpy = -((p3[0] - p1[0]) * hx + gx * (p6[0] - p4[0])) / (gn * hn)
        return ph, dpx, dpy


    phi[0, :], dphidx[0, :], dphidy[0, :] = compute_basis(
        0, t[:, 1], t[:, 0], t[:, 2], t[:, 3], t[:, 0], t[:, 5])

    phi[1, :], dphidx[1, :], dphidy[1, :] = compute_basis(
        1, t[:, 0], t[:, 1], t[:, 2], t[:, 4], t[:, 1], t[:, 3])

    phi[2, :], dphidx[2, :], dphidy[2, :] = compute_basis(
        2, t[:, 1], t[:, 2], t[:, 0], t[:, 5], t[:, 2], t[:, 4])

    phi[3, :], dphidx[3, :], dphidy[3, :] = compute_basis(
        3, t[:, 2], t[:, 0], t[:, 1], t[:, 4], t[:, 2], t[:, 3])

    phi[4, :], dphidx[4, :], dphidy[4, :] = compute_basis(
        4, t[:, 0], t[:, 1], t[:, 2], t[:, 3], t[:, 1], t[:, 4])

    phi[5, :], dphidx[5, :], dphidy[5, :] = compute_basis(
        5, t[:, 1], t[:, 2], t[:, 0], t[:, 5], t[:, 2], t[:, 3])

    return phi, dphidx, dphidy


def local_stiffness_matrix_t3(vertices, nu=1.0):
    vertices = np.asarray(vertices, dtype=float)
    area2 = vertices[0, 0] * (vertices[1, 1] - vertices[1, 2]) \
          + vertices[0, 1] * (vertices[1, 2] - vertices[1, 0]) \
          + vertices[0, 2] * (vertices[1, 0] - vertices[1, 1])
    area = abs(area2) * 0.5
    if area < 1e-15:
        area = 1e-15


    _, dphidx, dphidy = basis_mn_t3(vertices, 1, vertices[:, 0:1])

    K = np.zeros((3, 3), dtype=float)
    for i in range(3):
        for j in range(3):
            K[i, j] = nu * area * (dphidx[i, 0] * dphidx[j, 0] + dphidy[i, 0] * dphidy[j, 0])
    return K


def local_mass_matrix_t3(vertices):
    vertices = np.asarray(vertices, dtype=float)
    area2 = vertices[0, 0] * (vertices[1, 1] - vertices[1, 2]) \
          + vertices[0, 1] * (vertices[1, 2] - vertices[1, 0]) \
          + vertices[0, 2] * (vertices[1, 0] - vertices[1, 1])
    area = abs(area2) * 0.5
    if area < 1e-15:
        area = 1e-15
    return (area / 3.0) * np.eye(3, dtype=float)


def wedge01_monomial_integral(e):
    e = np.asarray(e, dtype=int)
    if e[2] == -1:
        return 0.0

    value = 1.0
    k = e[0]
    for i in range(1, e[1] + 1):
        k = k + 1
        value = value * i / k

    k = k + 1
    value = value / k
    k = k + 1
    value = value / k

    if e[2] % 2 == 1:
        value = 0.0
    else:
        value = value * 2.0 / (e[2] + 1)

    return float(value)


def wedge_boundary_layer_integral(nu, delta, order=4):
    theta = 0.0
    for i in range(order + 1):
        for j in range(order + 1 - i):
            e = [i, j, 0]
            coeff = ((-1) ** j) * math.comb(order, i) * math.comb(order - i, j)
            theta += coeff * wedge01_monomial_integral(e)
    theta *= delta * np.sqrt(nu)
    return theta
