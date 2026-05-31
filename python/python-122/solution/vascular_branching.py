"""
脑血流动力学 — 血管分支递归生成模块

基于 collatz_recursive（递归序列）的递归思想，构建脑血管树的递归分支模型。
每一级血管按 Murray 定律分叉，并赋予随机的生理变异。

科学背景:
- 脑血管树具有典型的分形结构：从颈内动脉到毛细血管，经历约 15-20 级分支。
- Murray 定律: r_parent^3 = Σ r_child_i^3
- 分支角度由最小能耗原理决定，通常满足:
    cos(θ1) = (r0^4 + r1^4 - r2^4) / (2 r0^2 r1^2)
- 血管长度与直径比 (L/D) 在生理范围内约为 5-20。
- 类似于 Collatz 序列的递归规则，血管分支可建模为:
    若当前血管半径 r > r_capillary: 生成 2 个子分支
    若 r <= r_capillary: 终止（到达毛细血管）
"""

import numpy as np


def collatz_like_branching_rule(r, r_capillary=4e-3, symmetry_prob=0.7):
    """
    类 Collatz 血管分支规则。
    输入当前血管半径 r [mm]，输出是否继续分支及子血管参数。

    规则:
        if r > r_capillary:
            if random() < symmetry_prob:
                对称二分: r_child = r / 2^(1/3)
            else:
                非对称二分: r_child1 = 0.6 * r / 2^(1/3), r_child2 = 0.8 * r / 2^(1/3)
        else:
            终止分支（到达毛细血管水平）
    """
    if r <= r_capillary:
        return None

    if np.random.rand() < symmetry_prob:
        r_child = r / (2.0 ** (1.0 / 3.0))
        return [(r_child, r_child)]
    else:
        r_base = r / (2.0 ** (1.0 / 3.0))
        r1 = 0.6 * r_base
        r2 = np.clip((r ** 3 - r1 ** 3) ** (1.0 / 3.0), 0.3 * r_base, 1.2 * r_base)
        return [(r1, r2)]


def branch_angle_from_murray(r0, r1, r2):
    """
    由 Murray 定律推导分支角度（基于能量最小化）。
    cos(θ1) = (r0^4 + r1^4 - r2^4) / (2 r0^2 r1^2)
    cos(θ2) = (r0^4 + r2^4 - r1^4) / (2 r0^2 r2^2)
    """
    if r0 < 1e-14 or r1 < 1e-14 or r2 < 1e-14:
        return np.pi / 4.0, np.pi / 4.0
    num1 = r0 ** 4 + r1 ** 4 - r2 ** 4
    den1 = 2.0 * r0 ** 2 * r1 ** 2
    num2 = r0 ** 4 + r2 ** 4 - r1 ** 4
    den2 = 2.0 * r0 ** 2 * r2 ** 2
    cos1 = np.clip(num1 / (den1 + 1e-14), -1.0, 1.0)
    cos2 = np.clip(num2 / (den2 + 1e-14), -1.0, 1.0)
    theta1 = np.arccos(cos1)
    theta2 = np.arccos(cos2)
    return theta1, theta2


class VascularBranch:
    """血管分支节点。"""
    def __init__(self, radius, length, start_point, end_point, parent=None, generation=0):
        self.radius = radius
        self.length = length
        self.start_point = np.asarray(start_point, dtype=float)
        self.end_point = np.asarray(end_point, dtype=float)
        self.parent = parent
        self.generation = generation
        self.children = []
        self.flow_rate = 0.0

    def add_child(self, child):
        self.children.append(child)


def generate_vascular_tree_recursive(start_radius=2.5, start_point=(0.0, 0.0, 0.0),
                                      direction=(0.0, 0.0, 1.0), max_generation=15,
                                      r_capillary=4e-3, L_over_D_mean=10.0):
    """
    递归生成三维脑血管树。

    参数:
        start_radius: 起始血管半径 [mm]（如颈内动脉约 2.5 mm）
        start_point: 起始坐标
        direction: 初始生长方向
        max_generation: 最大分支代数
        r_capillary: 毛细血管半径阈值 [mm]
        L_over_D_mean: 平均长径比

    返回:
        root: VascularBranch 根节点
        branches: 所有分支的列表
    """
    root = VascularBranch(
        radius=start_radius,
        length=start_radius * L_over_D_mean,
        start_point=start_point,
        end_point=np.array(start_point) + np.array(direction) * start_radius * L_over_D_mean,
        generation=0
    )
    branches = [root]
    queue = [root]

    while queue:
        current = queue.pop(0)
        if current.generation >= max_generation:
            continue

        result = collatz_like_branching_rule(current.radius, r_capillary=r_capillary)
        if result is None:
            continue

        (r1, r2) = result[0]
        theta1, theta2 = branch_angle_from_murray(current.radius, r1, r2)

        # 随机扰动生长方向
        base_dir = current.end_point - current.start_point
        base_dir = base_dir / (np.linalg.norm(base_dir) + 1e-14)

        # 构造局部坐标系
        if abs(base_dir[2]) < 0.9:
            perp = np.cross(base_dir, np.array([0.0, 0.0, 1.0]))
        else:
            perp = np.cross(base_dir, np.array([0.0, 1.0, 0.0]))
        perp = perp / (np.linalg.norm(perp) + 1e-14)
        perp2 = np.cross(base_dir, perp)
        perp2 = perp2 / (np.linalg.norm(perp2) + 1e-14)

        # 子分支方向（带随机扭转）
        twist1 = np.random.uniform(-0.3, 0.3)
        twist2 = np.random.uniform(-0.3, 0.3)
        dir1 = (np.cos(theta1) * base_dir +
                np.sin(theta1) * (np.cos(twist1) * perp + np.sin(twist1) * perp2))
        dir1 = dir1 / (np.linalg.norm(dir1) + 1e-14)
        dir2 = (np.cos(theta2) * base_dir +
                np.sin(theta2) * (np.cos(twist2) * perp + np.sin(twist2) * perp2))
        dir2 = dir2 / (np.linalg.norm(dir2) + 1e-14)

        L1 = r1 * L_over_D_mean * np.random.uniform(0.8, 1.2)
        L2 = r2 * L_over_D_mean * np.random.uniform(0.8, 1.2)

        child1 = VascularBranch(
            radius=r1, length=L1,
            start_point=current.end_point,
            end_point=current.end_point + dir1 * L1,
            parent=current, generation=current.generation + 1
        )
        child2 = VascularBranch(
            radius=r2, length=L2,
            start_point=current.end_point,
            end_point=current.end_point + dir2 * L2,
            parent=current, generation=current.generation + 1
        )

        current.add_child(child1)
        current.add_child(child2)
        branches.extend([child1, child2])
        queue.extend([child1, child2])

    return root, branches


def tree_statistics(branches):
    """计算血管树的统计信息。"""
    n_branches = len(branches)
    generations = [b.generation for b in branches]
    radii = [b.radius for b in branches]
    lengths = [b.length for b in branches]
    stats = {
        'n_branches': n_branches,
        'max_generation': max(generations) if generations else 0,
        'min_radius': min(radii) if radii else 0.0,
        'max_radius': max(radii) if radii else 0.0,
        'total_length': sum(lengths),
        'mean_radius': np.mean(radii) if radii else 0.0,
    }
    return stats
