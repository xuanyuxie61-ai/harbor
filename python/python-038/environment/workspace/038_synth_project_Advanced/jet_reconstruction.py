
import numpy as np
from scipy.spatial.distance import cdist

from adaptive_sampling import HilbertSpatialIndex
from cubature_integrator import integrate_monte_carlo, integrate_adaptive_1d


class PseudoJet:
    
    def __init__(self, px, py, pz, E, index=-1):
        self.p = np.array([px, py, pz, E], dtype=float)
        self.index = index
        self.parents = []
    
    @property
    def pt(self):
        return float(np.sqrt(self.p[0]**2 + self.p[1]**2))
    
    @property
    def px(self):
        return float(self.p[0])
    
    @property
    def py(self):
        return float(self.p[1])
    
    @property
    def pz(self):
        return float(self.p[2])
    
    @property
    def E(self):
        return float(self.p[3])
    
    @property
    def phi(self):
        return float(np.arctan2(self.p[1], self.p[0]))
    
    @property
    def eta(self):
        pt = self.pt
        if pt < 1e-12:
            return 1e6 if self.p[2] > 0 else -1e6
        theta = np.arctan2(pt, self.p[2])
        return -np.log(np.tan(theta / 2.0))
    
    @property
    def mass(self):
        m2 = self.E**2 - np.sum(self.p[:3]**2)
        return float(np.sqrt(max(m2, 0.0)))
    
    def __add__(self, other):
        return PseudoJet(
            self.px + other.px,
            self.py + other.py,
            self.pz + other.pz,
            self.E + other.E,
            index=-1
        )


def deltaR2(p1, p2):
    dphi = np.abs(p1.phi - p2.phi)
    if dphi > np.pi:
        dphi = 2.0 * np.pi - dphi
    deta = p1.eta - p2.eta
    return dphi**2 + deta**2


def cluster_jets(particles, R=0.4, p=-1, pt_min=5.0, use_hilbert=True):
    if len(particles) == 0:
        return []
    

    jets = [PseudoJet(p.px, p.py, p.pz, p.E, index=i)
            for i, p in enumerate(particles)]
    

    hilbert = None
    if use_hilbert and len(jets) > 20:
        hilbert = HilbertSpatialIndex(
            n_bits=8,
            bbox=[(-200.0, 200.0), (-200.0, 200.0), (-200.0, 200.0)]
        )
        for j in jets:
            hilbert.add_point(j.p[:3], j.index)
        hilbert.build_index()
    
    while len(jets) > 0:
        n = len(jets)
        if n == 1:
            break
        

        d_iB = np.array([j.pt**(2*p) if j.pt > 0 else 0.0 for j in jets])
        

        d_ij = np.full((n, n), np.inf)
        for i in range(n):
            for j in range(i + 1, n):
                d_ij[i, j] = min(jets[i].pt, jets[j].pt)**(2*p) * deltaR2(jets[i], jets[j]) / (R**2)
        

        min_dij = np.min(d_ij)
        min_idx = np.unravel_index(np.argmin(d_ij), d_ij.shape)
        min_diB = np.min(d_iB)
        
        if min_diB < min_dij:

            i_remove = np.argmin(d_iB)
            if jets[i_remove].pt < pt_min:
                del jets[i_remove]
            else:

                jet = jets.pop(i_remove)
                if jet.pt >= pt_min:
                    return [jet] + cluster_jets(jets, R, p, pt_min, False)
                else:
                    return cluster_jets(jets, R, p, pt_min, False)
        else:

            i, j = min_idx
            new_jet = jets[i] + jets[j]

            if j > i:
                del jets[j]
                del jets[i]
            else:
                del jets[i]
                del jets[j]
            jets.append(new_jet)
    

    result = [j for j in jets if j.pt >= pt_min]
    return result


def knapsack_optimal_subjets(jet, n_subjets_target=2, R_sub=0.2):



    rng = np.random.default_rng(42)
    n_constituents = min(10, max(3, int(jet.pt / 5.0)))
    
    constituents = []
    for _ in range(n_constituents):
        frac = rng.uniform(0.05, 0.3)
        phi_c = rng.uniform(0.0, 2.0 * np.pi)
        pt_c = frac * jet.pt
        px_c = pt_c * np.cos(phi_c)
        py_c = pt_c * np.sin(phi_c)
        pz_c = frac * jet.pz
        E_c = np.sqrt(px_c**2 + py_c**2 + pz_c**2 + 0.01)
        constituents.append(PseudoJet(px_c, py_c, pz_c, E_c))
    
    target_mass = 91.1876
    
    n = len(constituents)
    best_mass = -1.0
    best_subset = []
    best_diff = 1e9
    

    max_subsets = min(2**n, 512)
    for mask in range(1, max_subsets):
        subset = []
        for i in range(n):
            if mask & (1 << i):
                subset.append(constituents[i])
        
        if len(subset) < 2 or len(subset) > n_subjets_target + 3:
            continue
        
        combined = subset[0]
        for p in subset[1:]:
            combined = combined + p
        
        diff = abs(combined.mass - target_mass)
        if diff < best_diff:
            best_diff = diff
            best_mass = combined.mass
            best_subset = subset
    
    return best_mass, best_subset


def compute_thrust(particles):
    if len(particles) == 0:
        return 0.0, np.array([1.0, 0.0, 0.0])
    
    momenta = np.array([p.p[:3] for p in particles])
    norms = np.linalg.norm(momenta, axis=1)
    total_norm = np.sum(norms)
    if total_norm < 1e-12:
        return 0.0, np.array([1.0, 0.0, 0.0])
    

    best_T = 0.0
    best_axis = np.array([1.0, 0.0, 0.0])
    

    test_dirs = momenta / np.maximum(norms[:, None], 1e-12)
    n_random = min(50, len(particles) * 2)
    rng = np.random.default_rng(123)
    random_dirs = rng.normal(size=(n_random, 3))
    random_dirs /= np.linalg.norm(random_dirs, axis=1)[:, None]
    test_dirs = np.vstack([test_dirs, random_dirs])
    
    for n in test_dirs:
        proj = np.abs(momenta @ n)
        T = np.sum(proj) / total_norm
        if T > best_T:
            best_T = T
            best_axis = n
    
    return best_T, best_axis


def compute_sphericity(particles):
    if len(particles) == 0:
        return {'S': 0.0, 'A': 0.0, 'eigenvalues': np.zeros(3)}
    
    momenta = np.array([p.p[:3] for p in particles])
    p2_sum = np.sum(momenta**2)
    if p2_sum < 1e-12:
        return {'S': 0.0, 'A': 0.0, 'eigenvalues': np.zeros(3)}
    
    S_tensor = np.zeros((3, 3))
    for p_vec in momenta:
        S_tensor += np.outer(p_vec, p_vec)
    S_tensor /= p2_sum
    
    eigvals = np.sort(np.linalg.eigvalsh(S_tensor))[::-1]
    S = 1.5 * (eigvals[1] + eigvals[2])
    A = 1.5 * eigvals[2]
    
    return {'S': S, 'A': A, 'eigenvalues': eigvals}


def compute_jet_broadening(particles, thrust_axis):
    if len(particles) == 0:
        return 0.0
    
    E_vis = sum(p.E for p in particles)
    if E_vis < 1e-12:
        return 0.0
    
    n = thrust_axis / np.linalg.norm(thrust_axis)
    total = 0.0
    for p in particles:
        p_vec = p.p[:3]
        cross = np.linalg.norm(np.cross(p_vec, n))
        total += cross
    
    return total / (2.0 * E_vis)


def test_jet_reconstruction():
    rng = np.random.default_rng(42)
    

    particles = []

    for _ in range(20):
        pt = rng.exponential(10.0)
        deta = rng.normal(0.0, 0.1)
        dphi = rng.normal(0.0, 0.1)
        eta = deta
        phi = dphi
        px = pt * np.cos(phi)
        py = pt * np.sin(phi)
        pz = pt * np.sinh(eta)
        E = np.sqrt(px**2 + py**2 + pz**2 + 0.01)
        particles.append(PseudoJet(px, py, pz, E))
    

    for _ in range(10):
        pt = rng.exponential(2.0)
        eta = rng.uniform(-2.0, 2.0)
        phi = rng.uniform(0.0, 2.0 * np.pi)
        px = pt * np.cos(phi)
        py = pt * np.sin(phi)
        pz = pt * np.sinh(eta)
        E = np.sqrt(px**2 + py**2 + pz**2 + 0.01)
        particles.append(PseudoJet(px, py, pz, E))
    

    jets = cluster_jets(particles, R=0.4, p=-1, pt_min=5.0)
    assert len(jets) >= 1, "Anti-kT should find at least one jet"
    

    jets_kt = cluster_jets(particles, R=0.4, p=1, pt_min=5.0)
    assert len(jets_kt) >= 1
    

    T, axis = compute_thrust(particles)
    assert 0.5 <= T <= 1.0 + 1e-6, f"Thrust out of range: {T}"
    

    spher = compute_sphericity(particles)
    assert 0.0 <= spher['S'] <= 1.0 + 1e-6
    

    B = compute_jet_broadening(particles, axis)
    assert B >= 0.0
    

    if len(jets) > 0:
        mass, subset = knapsack_optimal_subjets(jets[0])
        assert mass >= 0.0
    
    return True


if __name__ == "__main__":
    test_jet_reconstruction()
    print("Jet reconstruction tests passed.")
