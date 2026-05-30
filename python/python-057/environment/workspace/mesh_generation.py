
import numpy as np


class CVT1D:
    
    def __init__(self, n_generators, z_min=-200.0, z_max=0.0,
                 density_type='chebyshev'):
        self.n = n_generators
        self.z_min = z_min
        self.z_max = z_max
        self.density_type = density_type
    
    def density_function(self, z):
        z = np.asarray(z)
        
        if self.density_type == 'uniform':
            return np.ones_like(z)
        elif self.density_type == 'chebyshev':

            s = 2.0 * (z - self.z_min) / (self.z_max - self.z_min) - 1.0
            s = np.clip(s, -0.999, 0.999)
            return 1.0 / np.sqrt(1.0 - s**2)
        elif self.density_type == 'thermocline':

            z_rel = z + 100.0
            return 1.0 + 2.0 * np.exp(-z_rel**2 / 400.0)
        elif self.density_type == 'mixed_layer':

            return 1.0 + 3.0 * np.exp(z / 20.0)
        else:
            return np.ones_like(z)
    
    def lloyd_iteration(self, n_samples=10000, max_iter=50, tol=1.0e-6):

        generators = self._chebyshev_zeros()
        energy_history = []
        
        for it in range(max_iter):

            samples = np.random.uniform(self.z_min, self.z_max, n_samples)
            

            weights = self.density_function(samples)
            

            midpoints = np.sort(generators)
            cell_sums = np.zeros(self.n)
            cell_weights = np.zeros(self.n)
            
            for sample, weight in zip(samples, weights):

                distances = np.abs(sample - generators)
                nearest = np.argmin(distances)
                cell_sums[nearest] += sample * weight
                cell_weights[nearest] += weight
            

            new_generators = np.where(cell_weights > 1.0e-12,
                                      cell_sums / cell_weights,
                                      generators)
            

            new_generators = np.clip(new_generators, self.z_min, self.z_max)
            new_generators = np.sort(new_generators)
            

            energy = self._compute_energy(generators, samples, weights)
            energy_history.append(energy)
            

            displacement = np.max(np.abs(new_generators - generators))
            if displacement < tol:
                break
            
            generators = new_generators
        
        return generators, energy_history
    
    def _chebyshev_zeros(self):
        i = np.arange(self.n)
        theta = np.pi * (2.0 * i + 1.0) / (2.0 * self.n)
        z = 0.5 * (self.z_max + self.z_min) + \
            0.5 * (self.z_max - self.z_min) * np.cos(theta)
        return z
    
    def _compute_energy(self, generators, samples, weights):
        energy = 0.0
        for sample, weight in zip(samples, weights):
            distances = (sample - generators)**2
            nearest = np.min(distances)
            energy += weight * nearest
        return energy / len(samples)


def delaunay_triangulation_2d(points):
    points = np.asarray(points)
    n = len(points)
    
    if n < 3:
        return np.array([])
    


    
    idx = np.argsort(points[:, 0])
    sorted_points = points[idx]
    
    triangles = []
    

    if n >= 3:

        center = np.mean(sorted_points, axis=0)
        

        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        angle_idx = np.argsort(angles)
        

        for i in range(n - 2):
            v0 = angle_idx[0]
            v1 = angle_idx[i + 1]
            v2 = angle_idx[i + 2]
            

            cross = (points[v1, 0] - points[v0, 0]) * (points[v2, 1] - points[v0, 1]) - \
                    (points[v1, 1] - points[v0, 1]) * (points[v2, 0] - points[v0, 0])
            
            if cross > 0:
                triangles.append([v0, v1, v2])
            else:
                triangles.append([v0, v2, v1])
    
    return np.array(triangles, dtype=int)


def triangulate_ocean_domain(x_range=(0, 10000), y_range=(0, 10000),
                              n_points=50):

    nx = int(np.sqrt(n_points))
    ny = nx
    
    x = np.linspace(x_range[0], x_range[1], nx)
    y = np.linspace(y_range[0], y_range[1], ny)
    
    nodes = []
    for i in range(nx):
        for j in range(ny):

            dx = np.random.uniform(-100, 100)
            dy = np.random.uniform(-100, 100)
            nodes.append([x[i] + dx, y[j] + dy])
    
    nodes = np.array(nodes)
    

    for i, node in enumerate(nodes):
        if node[0] < x_range[0] + 50:
            nodes[i, 0] = x_range[0]
        if node[0] > x_range[1] - 50:
            nodes[i, 0] = x_range[1]
        if node[1] < y_range[0] + 50:
            nodes[i, 1] = y_range[0]
        if node[1] > y_range[1] - 50:
            nodes[i, 1] = y_range[1]
    
    triangles = delaunay_triangulation_2d(nodes)
    
    return nodes, triangles
