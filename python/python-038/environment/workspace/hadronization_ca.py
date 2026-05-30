
import numpy as np
from scipy.special import gamma as gamma_func


class Hadron:
    
    def __init__(self, px, py, pz, E, pid=211, charge=1):
        self.p = np.array([px, py, pz, E], dtype=float)
        self.pid = pid
        self.charge = charge
    
    @property
    def pt(self):
        return float(np.sqrt(self.p[0]**2 + self.p[1]**2))
    
    @property
    def mass(self):
        m2 = self.p[3]**2 - np.sum(self.p[:3]**2)
        return float(np.sqrt(max(m2, 0.0)))
    
    @property
    def eta(self):
        p = self.p
        pt = self.pt
        if pt < 1e-12:
            return 1e6 if p[2] > 0 else -1e6
        theta = np.arctan2(pt, p[2])
        return -np.log(np.tan(theta / 2.0))
    
    @property
    def phi(self):
        return float(np.arctan2(self.p[1], self.p[0]))


def run_cellular_automaton_hadronization(partons, R_cone=0.4, pt_min=0.5,
                                         sigma_kT=0.3, seed=42):
    rng = np.random.default_rng(seed)
    
    if len(partons) == 0:
        return [], []
    

    etas = np.array([p.eta for p in partons])
    phis = np.array([p.phi for p in partons])
    pts = np.array([p.pt for p in partons])
    energies = np.array([p.p[3] for p in partons])
    

    eta_min, eta_max = etas.min() - 0.5, etas.max() + 0.5
    n_cells = max(20, int((eta_max - eta_min) / 0.1))
    d_eta = (eta_max - eta_min) / n_cells
    

    cell_idx = np.clip(((etas - eta_min) / d_eta).astype(int), 0, n_cells - 1)
    occupation = np.zeros(n_cells, dtype=int)
    cell_pt = np.zeros(n_cells, dtype=float)
    cell_E = np.zeros(n_cells, dtype=float)
    
    for i, ci in enumerate(cell_idx):
        occupation[ci] += 1
        cell_pt[ci] += pts[i]
        cell_E[ci] += energies[i]
    



    state = (occupation > 0).astype(int)
    new_state = np.zeros_like(state)
    
    for it in range(3):
        for i in range(n_cells):
            left = state[i - 1] if i > 0 else 0
            self_s = state[i]
            right = state[i + 1] if i < n_cells - 1 else 0

            new_state[i] = left ^ (self_s | right)
        state = new_state.copy()
    

    clusters = []
    in_cluster = False
    current = []
    for i in range(n_cells):
        if state[i] == 1:
            if not in_cluster:
                in_cluster = True
                current = [i]
            else:
                current.append(i)
        else:
            if in_cluster:
                clusters.append(current)
                in_cluster = False
                current = []
    if in_cluster:
        clusters.append(current)
    

    if len(clusters) == 0:
        clusters = [[i] for i in range(n_cells) if occupation[i] > 0]
    

    hadrons = []
    cluster_info = []
    
    for cid, cluster in enumerate(clusters):
        if len(cluster) == 0:
            continue
        
        total_E = sum(cell_E[i] for i in cluster)
        total_pt = sum(cell_pt[i] for i in cluster)
        avg_eta = eta_min + d_eta * np.mean(cluster)
        

        n_partons_in_cluster = sum(occupation[i] for i in cluster)
        n_hadrons = max(2, n_partons_in_cluster)
        n_hadrons = min(n_hadrons, 30)
        

        z_raw = rng.gamma(shape=2.0, scale=1.0, size=n_hadrons)
        z_vals = z_raw / z_raw.sum()
        
        for zh in z_vals:
            if zh < 1e-6:
                continue
            
            E_h = zh * total_E
            if E_h < 0.05:
                continue
            

            phi_h = rng.uniform(0.0, 2.0 * np.pi)
            theta_h = 2.0 * np.arctan(np.exp(-avg_eta))
            theta_h += rng.normal(0.0, 0.15)
            

            px_h = E_h * np.sin(theta_h) * np.cos(phi_h)
            py_h = E_h * np.sin(theta_h) * np.sin(phi_h)
            pz_h = E_h * np.cos(theta_h)
            

            m_h = 0.1396
            p_mag = np.sqrt(max(px_h**2 + py_h**2 + pz_h**2, 1e-9))

            desired_E = E_h
            scale = desired_E / max(p_mag, 1e-9)
            px_h *= scale
            py_h *= scale
            pz_h *= scale
            E_h = np.sqrt(max(px_h**2 + py_h**2 + pz_h**2 + m_h**2, 1e-6))
            
            had = Hadron(px_h, py_h, pz_h, E_h, pid=211, charge=rng.choice([-1, 1]))
            if had.pt >= pt_min:
                hadrons.append(had)
        
        cluster_info.append({
            'cluster_id': cid,
            'cells': cluster,
            'energy': total_E,
            'n_hadrons': n_hadrons
        })
    

    total_parton_E = sum(p.p[3] for p in partons)
    total_hadron_E = sum(h.p[3] for h in hadrons)
    if total_hadron_E > 1e-9 and abs(total_hadron_E - total_parton_E) / total_parton_E > 0.05:
        scale = total_parton_E / total_hadron_E
        for h in hadrons:
            h.p[:3] *= scale
            h.p[3] = np.sqrt(max(np.sum(h.p[:3]**2) + 0.1396**2, 1e-9))
    
    return hadrons, cluster_info


def boundary_word_from_jet(hadrons, n_bins_phi=24, n_bins_eta=20):
    if len(hadrons) == 0:
        return "", 0.0, 0.0
    
    etas = np.array([h.eta for h in hadrons])
    phis = np.array([h.phi for h in hadrons])
    weights = np.array([h.pt for h in hadrons])
    

    phis = np.mod(phis, 2.0 * np.pi)
    

    c_eta = np.average(etas, weights=np.maximum(weights, 1e-12))
    c_phi = np.average(phis, weights=np.maximum(weights, 1e-12))
    

    h_hist, eta_edges, phi_edges = np.histogram2d(
        etas, phis, bins=[n_bins_eta, n_bins_phi],
        weights=weights
    )
    

    threshold = 0.1 * h_hist.max() if h_hist.max() > 0 else 0.0
    active = h_hist > threshold
    


    word = []
    for i in range(n_bins_eta):
        for j in range(n_bins_phi):
            if active[i, j]:

                eta_c = 0.5 * (eta_edges[i] + eta_edges[i + 1])
                phi_c = 0.5 * (phi_edges[j] + phi_edges[j + 1])
                d_eta = eta_c - c_eta
                d_phi = phi_c - c_phi
                angle = np.arctan2(d_phi, d_eta)

                sector = int(np.round((angle + np.pi) / (np.pi / 4.0))) % 8
                word.append(str(sector))
    
    boundary_word = "".join(word)
    return boundary_word, c_eta, c_phi


def test_hadronization():

    class FakeParton:
        def __init__(self, px, py, pz, E):
            self.p = np.array([px, py, pz, E], dtype=float)
            self.flavor = 'q'
        @property
        def pt(self):
            return float(np.sqrt(self.p[0]**2 + self.p[1]**2))
        @property
        def eta(self):
            pt = self.pt
            if pt < 1e-12:
                return 1e6 if self.p[2] > 0 else -1e6
            theta = np.arctan2(pt, self.p[2])
            return -np.log(np.tan(theta / 2.0))
        @property
        def phi(self):
            return float(np.arctan2(self.p[1], self.p[0]))
    
    partons = [FakeParton(10.0*np.cos(i), 10.0*np.sin(i), 5.0*(i-2.5), 12.0)
               for i in range(5)]
    
    hadrons, clusters = run_cellular_automaton_hadronization(partons, seed=42)
    assert len(hadrons) > 0, "Hadronization produced no hadrons"
    assert all(h.pt > 0 for h in hadrons)
    
    word, c_eta, c_phi = boundary_word_from_jet(hadrons)
    assert isinstance(word, str)
    
    return True


if __name__ == "__main__":
    test_hadronization()
    print("Hadronization CA tests passed.")
