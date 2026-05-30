
import numpy as np
from quadrature_rules import integrate_over_triangle


C_AIR = 343.0


def edge_diffraction_coefficient(phi_inc, phi_diff, k, edge_type='hard'):
    if edge_type == 'hard':
        n_edge = 2.0
        Phi = phi_inc + phi_diff

        eps = 0.1

        Phi_mod = Phi % (2.0 * np.pi)
        if abs(Phi_mod) < eps:
            Phi_mod = eps
        if abs(Phi_mod - np.pi) < eps:
            Phi_mod = np.pi + eps
        if abs(Phi_mod - 2.0 * np.pi) < eps:
            Phi_mod = 2.0 * np.pi - eps

        D = -np.exp(-1j * np.pi / 4.0) / (2.0 * n_edge * np.sqrt(2.0 * np.pi * max(k, 1e-10)))
        arg1 = (np.pi + Phi_mod) / (2.0 * n_edge)
        arg2 = (np.pi - Phi_mod) / (2.0 * n_edge)

        tan1 = np.tan(arg1)
        tan2 = np.tan(arg2)
        tan1 = np.where(abs(tan1) < 1e-10, 1e10 * np.sign(tan1) if tan1 != 0 else 1e10, tan1)
        tan2 = np.where(abs(tan2) < 1e-10, 1e10 * np.sign(tan2) if tan2 != 0 else 1e10, tan2)
        term1 = 1.0 / tan1
        term2 = 1.0 / tan2
        D = D * (term1 + term2)

        if np.abs(D) > 100.0:
            D = 100.0 * np.exp(1j * np.angle(D))
        return D
    else:
        return 0.0 + 0.0j


def keller_cone_direction(edge_point, edge_tangent, incident_dir):
    cos_beta = np.dot(incident_dir, edge_tangent)


    beta = np.arccos(np.clip(abs(cos_beta), 0.0, 1.0))

    theta = np.random.uniform(0.0, 2.0 * np.pi)


    if abs(abs(cos_beta) - 1.0) < 1e-10:

        perp1 = np.array([1.0, 0.0, 0.0])
        if abs(edge_tangent[0]) > 0.9:
            perp1 = np.array([0.0, 1.0, 0.0])
    else:
        perp1 = incident_dir - cos_beta * edge_tangent
        perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(edge_tangent, perp1)
    perp2 = perp2 / np.linalg.norm(perp2)


    d_diff = np.cos(beta) * edge_tangent + np.sin(beta) * (np.cos(theta) * perp1 + np.sin(theta) * perp2)

    if np.dot(d_diff, incident_dir) > 0:
        d_diff = np.cos(beta) * edge_tangent - np.sin(beta) * (np.cos(theta) * perp1 + np.sin(theta) * perp2)
    d_diff = d_diff / np.linalg.norm(d_diff)
    return d_diff


def detect_room_edges(surfaces):
    edges = []

    room_edges = [

        (np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 'floor', 'front_wall'),
        (np.array([10.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), 'floor', 'front_wall'),
        (np.array([10.0, 8.0, 0.0]), np.array([-1.0, 0.0, 0.0]), 'floor', 'back_wall'),
        (np.array([0.0, 8.0, 0.0]), np.array([0.0, -1.0, 0.0]), 'floor', 'back_wall'),

        (np.array([0.0, 0.0, 5.0]), np.array([1.0, 0.0, 0.0]), 'ceiling', 'front_wall'),
        (np.array([10.0, 0.0, 5.0]), np.array([0.0, 1.0, 0.0]), 'ceiling', 'front_wall'),
        (np.array([10.0, 8.0, 5.0]), np.array([-1.0, 0.0, 0.0]), 'ceiling', 'back_wall'),
        (np.array([0.0, 8.0, 5.0]), np.array([0.0, -1.0, 0.0]), 'ceiling', 'back_wall'),

        (np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), 'left_wall', 'front_wall'),
        (np.array([10.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), 'right_wall', 'front_wall'),
        (np.array([10.0, 8.0, 0.0]), np.array([0.0, 0.0, 1.0]), 'right_wall', 'back_wall'),
        (np.array([0.0, 8.0, 0.0]), np.array([0.0, 0.0, 1.0]), 'left_wall', 'back_wall'),
    ]
    for point, tangent, s1, s2 in room_edges:
        tangent = tangent / np.linalg.norm(tangent)
        edges.append({'point': point, 'tangent': tangent, 'surf1': s1, 'surf2': s2})
    return edges


def distance_point_to_edge(point, edge_point, edge_tangent):
    diff = point - edge_point
    cross = np.cross(diff, edge_tangent)
    return np.linalg.norm(cross) / np.linalg.norm(edge_tangent)


def diffraction_edge_response(source_pos, receiver_pos, edge, freq):
    k = 2.0 * np.pi * freq / C_AIR
    ep = edge['point']
    et = edge['tangent']


    vec_s = source_pos - ep
    vec_r = receiver_pos - ep
    t_s = np.dot(vec_s, et)
    t_r = np.dot(vec_r, et)

    t_edge = (t_s + t_r) / 2.0
    diff_point = ep + t_edge * et

    d_source = np.linalg.norm(source_pos - diff_point)
    d_receiver = np.linalg.norm(receiver_pos - diff_point)

    if d_source < 1e-3 or d_receiver < 1e-3:
        return 0.0


    inc_dir = (source_pos - diff_point) / d_source
    rec_dir = (receiver_pos - diff_point) / d_receiver


    inc_perp = inc_dir - np.dot(inc_dir, et) * et
    rec_perp = rec_dir - np.dot(rec_dir, et) * et
    inc_norm = np.linalg.norm(inc_perp)
    rec_norm = np.linalg.norm(rec_perp)
    if inc_norm < 1e-14 or rec_norm < 1e-14:
        phi_inc = 0.0
        phi_diff = 0.0
    else:
        inc_perp = inc_perp / inc_norm
        rec_perp = rec_perp / rec_norm
        phi_inc = np.arctan2(inc_perp[1], inc_perp[0])
        phi_diff = np.arctan2(rec_perp[1], rec_perp[0])

    D = edge_diffraction_coefficient(abs(phi_inc), abs(phi_diff), k)


    rho = d_source
    s = d_receiver
    spread = np.sqrt(rho / (s * (rho + s)))
    phase = np.exp(-1j * k * s)
    amplitude = spread * phase * D

    return amplitude


def integrate_diffraction_over_surface(surfaces, source_pos, edge, freq, precision=5):
    total = 0.0
    for name, tris in surfaces.items():
        for i in range(0, len(tris), 3):
            v0, v1, v2 = tris[i], tris[i + 1], tris[i + 2]
            def func(p):
                return np.abs(diffraction_edge_response(source_pos, p, edge, freq))
            val = integrate_over_triangle(func, v0, v1, v2, precision)
            total += val
    return total


def compute_edge_diffraction_field(source_pos, receiver_pos, edges, freq):
    total = 0.0 + 0.0j
    for edge in edges:
        amp = diffraction_edge_response(source_pos, receiver_pos, edge, freq)
        total += amp
    return total
