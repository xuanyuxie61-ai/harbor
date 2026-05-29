"""
Dynamic-Programming Multi-Eddy Tracking
=======================================
Derived from seed project 444_football_dynamic (dynamic programming
for combinatorial score enumeration).

In oceanography, tracking multiple mesoscale eddies over time is
challenging due to merging, splitting, and birth/death events.
We formulate eddy tracking as a dynamic programming problem on a
state graph.

State representation:
    S_t = { eddy_1(t), eddy_2(t), ..., eddy_{N_t}(t) }

Transition cost from eddy i at time t to eddy j at time t+1:
    C(i→j) = w₁·Δd + w₂·ΔA + w₃·Δζ + w₄·(1 − overlap)

where:
    Δd    = |x_i(t) − x_j(t+1)|  (centroid displacement)
    ΔA    = |A_i(t) − A_j(t+1)| / max(A_i, A_j)
    Δζ    = |ζ_i(t) − ζ_j(t+1)| / max(|ζ_i|, |ζ_j|)
    overlap = |Area( eddy_i(t) ∩ eddy_j(t+1) )| / min(A_i, A_j)

The optimal assignment is found via dynamic programming (Viterbi-like)
on the eddy-state lattice.

Mathematically, this is a shortest-path problem on a DAG:
    V_t(j) = min_i [ V_{t−1}(i) + C(i→j) ]
with initialization V_0(i) = 0.
"""

import numpy as np

class EddyTracker:
    """
    Track multiple eddies across time steps using dynamic programming.
    """

    def __init__(self, w_pos=1.0, w_area=0.5, w_vort=0.3, w_overlap=2.0):
        self.w_pos = w_pos
        self.w_area = w_area
        self.w_vort = w_vort
        self.w_overlap = w_overlap

    def _transition_cost(self, eddy_t, eddy_tp1):
        """
        Compute cost between two eddy states.

        eddy dict keys: 'centroid', 'area', 'mean_vorticity'
        """
        dx = eddy_t['centroid'][0] - eddy_tp1['centroid'][0]
        dy = eddy_t['centroid'][1] - eddy_tp1['centroid'][1]
        dist = np.sqrt(dx**2 + dy**2)

        A_t = max(eddy_t['area'], 1e-10)
        A_tp1 = max(eddy_tp1['area'], 1e-10)
        dA = abs(A_t - A_tp1) / max(A_t, A_tp1)

        zt = eddy_t['mean_vorticity']
        ztp1 = eddy_tp1['mean_vorticity']
        dzeta = abs(zt - ztp1) / max(abs(zt), abs(ztp1), 1e-10)

        # Approximate overlap by Gaussian overlap of circular areas
        r_t = np.sqrt(A_t / np.pi)
        r_tp1 = np.sqrt(A_tp1 / np.pi)
        d_centers = dist
        if d_centers < r_t + r_tp1:
            # Simplified overlap estimate
            overlap = 1.0 - d_centers / (r_t + r_tp1)
        else:
            overlap = 0.0

        cost = (self.w_pos * dist / 1e5 +
                self.w_area * dA +
                self.w_vort * dzeta +
                self.w_overlap * (1.0 - overlap))
        return cost

    def track(self, eddy_snapshots):
        """
        Track eddies across a sequence of time snapshots.

        Parameters
        ----------
        eddy_snapshots : list of list of dict
            eddy_snapshots[t] = list of eddy dicts at time t.

        Returns
        -------
        trajectories : list of list
            Each trajectory is a list of (time, eddy_id) pairs.
        """
        T = len(eddy_snapshots)
        if T == 0:
            return []

        # Forward pass: compute minimum cost paths
        # V[t][j] = minimum cost to reach eddy j at time t
        V = []
        backpointer = []

        # Initialize
        N0 = len(eddy_snapshots[0])
        V.append([0.0] * N0)
        backpointer.append([None] * N0)

        for t in range(1, T):
            Nt = len(eddy_snapshots[t])
            Ntm1 = len(eddy_snapshots[t - 1])
            Vt = [np.inf] * Nt
            Bt = [None] * Nt

            for j in range(Nt):
                min_cost = np.inf
                best_i = None
                for i in range(Ntm1):
                    cost = V[t - 1][i] + self._transition_cost(
                        eddy_snapshots[t - 1][i], eddy_snapshots[t][j]
                    )
                    if cost < min_cost:
                        min_cost = cost
                        best_i = i
                Vt[j] = min_cost
                Bt[j] = best_i

            V.append(Vt)
            backpointer.append(Bt)

        # Backward pass: extract trajectories by greedy assignment
        # Start from each eddy at final time and trace back
        trajectories = []
        assigned = set()

        for j in range(len(eddy_snapshots[T - 1])):
            traj = [(T - 1, j)]
            cur_t = T - 1
            cur_j = j
            while cur_t > 0:
                prev_j = backpointer[cur_t][cur_j]
                if prev_j is None:
                    break
                traj.insert(0, (cur_t - 1, prev_j))
                cur_t -= 1
                cur_j = prev_j

            # Only keep trajectories with length ≥ 2
            if len(traj) >= 2:
                trajectories.append(traj)

        return trajectories

    def compute_lifetime_statistics(self, trajectories, eddy_snapshots):
        """
        Compute mean eddy lifetime, propagation speed, and path length.
        """
        stats = {
            'n_tracks': len(trajectories),
            'mean_lifetime_steps': 0.0,
            'mean_speed': 0.0,
            'mean_path_length': 0.0,
        }
        if not trajectories:
            return stats

        lifetimes = []
        speeds = []
        path_lengths = []

        for traj in trajectories:
            lifetime = len(traj)
            lifetimes.append(lifetime)

            path_len = 0.0
            total_dt = 0.0
            for k in range(1, len(traj)):
                t1, i1 = traj[k - 1]
                t2, i2 = traj[k]
                dt = 1.0  # normalized time step
                c1 = eddy_snapshots[t1][i1]['centroid']
                c2 = eddy_snapshots[t2][i2]['centroid']
                d = np.sqrt((c2[0] - c1[0])**2 + (c2[1] - c1[1])**2)
                path_len += d
                total_dt += dt
            path_lengths.append(path_len)
            if total_dt > 0:
                speeds.append(path_len / total_dt)

        stats['mean_lifetime_steps'] = np.mean(lifetimes) if lifetimes else 0.0
        stats['mean_speed'] = np.mean(speeds) if speeds else 0.0
        stats['mean_path_length'] = np.mean(path_lengths) if path_lengths else 0.0
        return stats
