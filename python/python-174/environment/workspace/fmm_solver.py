
import numpy as np
from fmm_tree import FMMOctree
from multipole_expansion import MultipoleExpansion
from local_expansion import LocalExpansion
from translation_operators import m2m_translate, m2l_translate, l2l_translate
from nbody_kernel import coulomb_potential_direct


class FMMSolver:

    def __init__(self, points, charges, order=4, max_depth=6, max_particles=20, separation_param=2.0):
        self.points = np.asarray(points, dtype=float)
        self.charges = np.asarray(charges, dtype=float)
        self.N = self.points.shape[0]
        self.order = order
        self.max_depth = max_depth
        self.max_particles = max_particles
        self.separation_param = separation_param


        self.tree = FMMOctree(self.points, self.charges, max_depth, max_particles, order, separation_param)


        self.tree.compute_moments_upward()


        self._compute_local_expansions()

    def _compute_local_expansions(self):
        all_nodes = self.tree.get_all_nodes()

        for node in all_nodes:
            node.local_coeffs_real = []
            node.local_coeffs_imag = []
            for l in range(node.order + 1):
                node.local_coeffs_real.append(np.zeros(l + 1))
                node.local_coeffs_imag.append(np.zeros(l + 1))


        def downward(node):
            if node.parent is not None:

                child_real, child_imag = l2l_translate(
                    node.parent.local_coeffs_real,
                    node.parent.local_coeffs_imag,
                    node.parent.center,
                    node.center,
                    node.order
                )
                for l in range(node.order + 1):
                    m_len = min(len(child_real[l]), len(node.local_coeffs_real[l]))
                    node.local_coeffs_real[l][:m_len] += child_real[l][:m_len]
                    node.local_coeffs_imag[l][:m_len] += child_imag[l][:m_len]


            interaction_list = node.get_interaction_list(all_nodes)
            for src in interaction_list:
                if (len(src.particle_indices) == 0 and src.is_leaf()
                        and np.sum(np.abs(src.multipole_moments_real[0])) < 1e-15):
                    continue
                l_real, l_imag = m2l_translate(
                    src.multipole_moments_real,
                    src.multipole_moments_imag,
                    src.center,
                    node.center,
                    node.order
                )
                for l in range(node.order + 1):
                    m_len = min(len(l_real[l]), len(node.local_coeffs_real[l]))
                    node.local_coeffs_real[l][:m_len] += l_real[l][:m_len]
                    node.local_coeffs_imag[l][:m_len] += l_imag[l][:m_len]

            if not node.is_leaf():
                for child in node.children:
                    downward(child)

        downward(self.tree.root)

    def compute_potential(self):
        potential = np.zeros(self.N)
        leaves = self.tree.get_leaves()
        all_nodes = self.tree.get_all_nodes()

        for leaf in leaves:
            if len(leaf.particle_indices) == 0:
                continue


            neighbors = leaf.get_neighbors(all_nodes)
            neighbor_leaves = set()
            for nb in neighbors:
                if nb.is_leaf():
                    neighbor_leaves.add(id(nb))
                else:
                    for lf in nb.collect_leaves():
                        neighbor_leaves.add(id(lf))
            neighbor_leaves.add(id(leaf))


            neighbor_indices = []
            for lf in leaves:
                if id(lf) in neighbor_leaves:
                    neighbor_indices.extend(lf.particle_indices)
            neighbor_indices = list(set(neighbor_indices))
            neighbor_pts = self.points[neighbor_indices]
            neighbor_chg = self.charges[neighbor_indices]


            for idx in leaf.particle_indices:
                pt = self.points[idx]
                phi = 0.0


                diff = pt - neighbor_pts
                dist = np.linalg.norm(diff, axis=1)
                mask = (dist > 1e-12) & (neighbor_chg != 0.0)
                if np.any(mask):
                    phi += np.sum(neighbor_chg[mask] / dist[mask])









                raise NotImplementedError("Hole_3: 请实现远场多极展开势能评估")

        return potential

    def compute_force(self):
        force = np.zeros((self.N, 3))
        leaves = self.tree.get_leaves()
        all_nodes = self.tree.get_all_nodes()

        for leaf in leaves:
            if len(leaf.particle_indices) == 0:
                continue
            neighbors = leaf.get_neighbors(all_nodes)
            neighbor_indices = []
            for nb in neighbors:
                if nb.is_leaf():
                    neighbor_indices.extend(nb.particle_indices)
                else:
                    for lf in nb.collect_leaves():
                        neighbor_indices.extend(lf.particle_indices)
            neighbor_indices.extend(leaf.particle_indices)
            neighbor_indices = list(set(neighbor_indices))

            if len(neighbor_indices) == 0:
                continue

            neighbor_pts = self.points[neighbor_indices]
            neighbor_chg = self.charges[neighbor_indices]

            for idx in leaf.particle_indices:
                pt = self.points[idx]
                diff = pt - neighbor_pts
                dist = np.linalg.norm(diff, axis=1)
                mask = dist > 1e-12
                if np.any(mask):
                    inv_r3 = 1.0 / (dist[mask] ** 3)
                    f = np.sum((neighbor_chg[mask] * inv_r3)[:, None] * diff[mask], axis=0)
                    force[idx] += f

        return force

    def get_tree_statistics(self):
        all_nodes = self.tree.get_all_nodes()
        leaves = self.tree.get_leaves()
        depths = [node.depth for node in all_nodes]
        n_particles_per_leaf = [len(leaf.particle_indices) for leaf in leaves]
        return {
            "total_nodes": len(all_nodes),
            "total_leaves": len(leaves),
            "max_depth": max(depths) if depths else 0,
            "avg_particles_per_leaf": float(np.mean(n_particles_per_leaf)) if n_particles_per_leaf else 0.0,
            "min_particles_per_leaf": int(np.min(n_particles_per_leaf)) if n_particles_per_leaf else 0,
            "max_particles_per_leaf": int(np.max(n_particles_per_leaf)) if n_particles_per_leaf else 0,
        }
