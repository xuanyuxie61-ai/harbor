
import numpy as np


def triangle_order3_reference_to_physical(triangle_xy, quad_num, quad_xy_ref):
    x1, x2, x3 = triangle_xy[0, 0], triangle_xy[0, 1], triangle_xy[0, 2]
    y1, y2, y3 = triangle_xy[1, 0], triangle_xy[1, 1], triangle_xy[1, 2]

    quad_xy_phys = np.zeros((2, quad_num))
    for q in range(quad_num):
        xi = quad_xy_ref[0, q]
        eta = quad_xy_ref[1, q]
        quad_xy_phys[0, q] = x1 + (x2 - x1) * xi + (x3 - x1) * eta
        quad_xy_phys[1, q] = y1 + (y2 - y1) * xi + (y3 - y1) * eta
    return quad_xy_phys


def triangle_area_2d(t):
    area = 0.5 * abs(
        t[0, 0] * (t[1, 1] - t[1, 2])
        + t[0, 1] * (t[1, 2] - t[1, 0])
        + t[0, 2] * (t[1, 0] - t[1, 1])
    )
    return area



TRIANGLE_QUAD_RULES = {
    "centroid": {
        "points": np.array([[1.0 / 3.0], [1.0 / 3.0]]),
        "weights": np.array([1.0]),
    },
    "order3": {

        "points": np.array([
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ]),
        "weights": np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]),
    },
    "order7": {

        "points": np.array([
            [1.0 / 3.0, 0.059715871789770, 0.797426985353087, 0.142,
             0.935, 0.05, 0.05],
            [1.0 / 3.0, 0.059715871789770, 0.059715871789770, 0.935,
             0.05, 0.05, 0.797426985353087],
        ]),
        "weights": np.array([
            0.225,
            0.132394152788506,
            0.132394152788506,
            0.125939180544827,
            0.125939180544827,
            0.125939180544827,
            0.125939180544827,
        ]),
    },
}


def integrate_over_triangulation(node_xy, triangle_node, quad_fun,
                                  rule_name="order7"):
    rule = TRIANGLE_QUAD_RULES.get(rule_name, TRIANGLE_QUAD_RULES["order7"])
    quad_xy_ref = rule["points"]
    quad_w = rule["weights"]
    quad_num = quad_w.shape[0]

    triangle_num = triangle_node.shape[1]
    quad_value = 0.0
    region_area = 0.0

    for t in range(triangle_num):
        tri_pts = node_xy[:, triangle_node[:, t]]
        tri_area = triangle_area_2d(tri_pts)
        if tri_area < 1e-14:
            continue

        quad_xy_phys = triangle_order3_reference_to_physical(
            tri_pts, quad_num, quad_xy_ref
        )
        f_vals = quad_fun(quad_xy_phys)
        quad_value += tri_area * np.dot(quad_w, f_vals)
        region_area += tri_area

    return quad_value, region_area


def integrate_nodal_over_triangulation(node_xy, triangle_node, nodal_values):
    triangle_num = triangle_node.shape[1]
    quad_value = 0.0
    region_area = 0.0

    for t in range(triangle_num):
        tri_pts = node_xy[:, triangle_node[:, t]]
        tri_area = triangle_area_2d(tri_pts)
        if tri_area < 1e-14:
            continue
        v1 = nodal_values[triangle_node[0, t]]
        v2 = nodal_values[triangle_node[1, t]]
        v3 = nodal_values[triangle_node[2, t]]
        quad_value += tri_area * (v1 + v2 + v3) / 3.0
        region_area += tri_area

    return quad_value, region_area


def test_triangulation_quadrature():

    node_xy = np.array([
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
    ])

    triangle_node = np.array([[0, 0],
                              [1, 3],
                              [3, 2]], dtype=np.int64)

    def f(pts):
        return pts[0, :] ** 2 + pts[1, :] ** 2

    val, area = integrate_over_triangulation(node_xy, triangle_node, f, "order7")

    assert abs(val - 2.0 / 3.0) < 0.01, f"积分值={val}, 期望=2/3"
    assert abs(area - 1.0) < 1e-10, f"面积={area}, 期望=1.0"
    print("triangulation_quadrature 自测试通过")


if __name__ == "__main__":
    test_triangulation_quadrature()
