
import numpy as np
from typing import Tuple, Optional
from numeric_utils import seeded_random, safe_divide


class PolymerChain:
    
    def __init__(
        self,
        n_chains: int = 10,
        beads_per_chain: int = 50,
        bond_length: float = 1.0,
        box: np.ndarray = None,
        random_seed: int = 42,
    ):
        if n_chains < 1:
            raise ValueError("n_chains 必须 >= 1")
        if beads_per_chain < 2:
            raise ValueError("beads_per_chain 必须 >= 2")
        if bond_length <= 0:
            raise ValueError("bond_length 必须 > 0")
        
        self.n_chains = n_chains
        self.beads_per_chain = beads_per_chain
        self.bond_length = bond_length
        self.n_total = n_chains * beads_per_chain
        
        if box is None:


            bead_volume = (4.0 / 3.0) * np.pi * (bond_length ** 3)
            total_bead_volume = self.n_total * bead_volume
            target_fraction = 0.65
            box_volume = total_bead_volume / target_fraction
            est_size = box_volume ** (1.0 / 3.0)
            self.box = np.array([est_size, est_size, est_size])
        else:
            self.box = np.array(box, dtype=float)
            if np.any(self.box <= 0):
                raise ValueError("box 尺寸必须 > 0")
        
        self.dim = 3
        self.positions = np.zeros((self.n_total, self.dim))
        self.velocities = np.zeros((self.n_total, self.dim))
        self.forces = np.zeros((self.n_total, self.dim))
        self.masses = np.ones(self.n_total)
        

        self.chain_starts = np.arange(0, self.n_total, self.beads_per_chain)
        

        self._initialize_conformation(random_seed)
        

        self._initialize_velocities(temperature=1.0, random_seed=random_seed + 1)
    
    def _initialize_conformation(self, seed: int):
        rng = np.random.RandomState(seed)
        

        directions = np.array([
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1]
        ], dtype=float)
        
        min_distance = 0.7 * self.bond_length
        
        for c in range(self.n_chains):
            start_idx = self.chain_starts[c]
            

            origin = rng.rand(self.dim) * self.box
            self.positions[start_idx] = origin
            
            for bead in range(1, self.beads_per_chain):
                idx = start_idx + bead
                prev_pos = self.positions[idx - 1].copy()
                

                valid_directions = []
                for d in directions:
                    trial_pos = prev_pos + self.bond_length * d

                    trial_pos = trial_pos % self.box
                    

                    conflict = False
                    for existing in range(idx):
                        dr = trial_pos - self.positions[existing]
                        dr = dr - self.box * np.rint(dr / self.box)
                        dist = np.linalg.norm(dr)
                        if dist < min_distance:
                            conflict = True
                            break
                    
                    if not conflict:
                        valid_directions.append(d)
                
                if len(valid_directions) > 0:
                    chosen = valid_directions[rng.randint(len(valid_directions))]
                    self.positions[idx] = (prev_pos + self.bond_length * chosen) % self.box
                else:

                    theta = rng.uniform(0, 2 * np.pi)
                    phi = rng.uniform(0, np.pi)
                    dx = self.bond_length * np.sin(phi) * np.cos(theta)
                    dy = self.bond_length * np.sin(phi) * np.sin(theta)
                    dz = self.bond_length * np.cos(phi)
                    self.positions[idx] = (prev_pos + np.array([dx, dy, dz])) % self.box
    
    def _initialize_velocities(self, temperature: float, random_seed: int):
        if temperature <= 0:
            raise ValueError("temperature 必须 > 0")
        
        rng = np.random.RandomState(random_seed)
        sigma = np.sqrt(temperature / self.masses)
        
        for d in range(self.dim):
            self.velocities[:, d] = rng.normal(0.0, sigma, self.n_total)
        

        v_cm = np.mean(self.velocities, axis=0)
        self.velocities -= v_cm
    
    def get_chain_positions(self, chain_id: int) -> np.ndarray:
        if chain_id < 0 or chain_id >= self.n_chains:
            raise IndexError("chain_id 越界")
        start = self.chain_starts[chain_id]
        end = start + self.beads_per_chain
        return self.positions[start:end, :].copy()
    
    def radius_of_gyration(self, chain_id: Optional[int] = None) -> float:
        if chain_id is None:
            pos = self.positions
        else:
            pos = self.get_chain_positions(chain_id)
        
        cm = np.mean(pos, axis=0)
        rg_sq = np.mean(np.sum((pos - cm) ** 2, axis=1))
        return float(np.sqrt(max(rg_sq, 0.0)))
    
    def end_to_end_distance(self, chain_id: int) -> float:
        pos = self.get_chain_positions(chain_id)
        dr = pos[-1] - pos[0]

        dr = dr - self.box * np.rint(dr / self.box)
        return float(np.linalg.norm(dr))
    
    def apply_pbc(self):
        self.positions = self.positions % self.box
    
    def kinetic_energy(self) -> float:
        return float(0.5 * np.sum(self.masses[:, np.newaxis] * self.velocities ** 2))
    
    def instantaneous_temperature(self) -> float:
        ndof = 3 * self.n_total - 3
        if ndof <= 0:
            return 0.0
        ke = self.kinetic_energy()
        return float(2.0 * ke / ndof)


def generate_ellipse_cross_section(
    n_points: int,
    semi_axes: Tuple[float, float],
    center: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    if n_points < 1:
        raise ValueError("n_points 必须 >= 1")
    a, b = semi_axes
    if a <= 0 or b <= 0:
        raise ValueError("半轴长度必须 > 0")
    
    cx, cy = center
    

    if a < b:
        h = 2.0 * a / (2.0 * n_points + 1.0)
        ni = n_points
        nj = int(np.ceil(b / a) * n_points)
    else:
        h = 2.0 * b / (2.0 * n_points + 1.0)
        nj = n_points
        ni = int(np.ceil(a / b) * n_points)
    
    points = []
    
    for j in range(nj + 1):
        i = 0
        x = cx
        y = cy + j * h

        if ((x - cx) / a) ** 2 + ((y - cy) / b) ** 2 <= 1.0 + 1e-12:
            points.append([x, y])
        if j > 0:
            y_mirror = 2 * cy - y
            if ((x - cx) / a) ** 2 + ((y_mirror - cy) / b) ** 2 <= 1.0 + 1e-12:
                points.append([x, y_mirror])
        
        while True:
            i += 1
            x = cx + i * h
            ellipse_val = ((x - cx) / a) ** 2 + ((y - cy) / b) ** 2
            if ellipse_val > 1.0:
                break
            

            points.append([x, y])
            points.append([2 * cx - x, y])
            if j > 0:
                points.append([x, 2 * cy - y])
                points.append([2 * cx - x, 2 * cy - y])
    
    arr = np.array(points)
    if arr.size == 0:
        return np.zeros((0, 2))
    return arr
