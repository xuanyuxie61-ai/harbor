
import numpy as np


class HilbertCurve3D:
    
    def __init__(self, r):
        self.r = r
        self.N = 1 << r
        self.max_h = (1 << (3 * r)) - 1
    
    def _transform(self, x, y, z, o, s):


        

        transforms = [
            (0, 1, 2, 0),
            (2, 1, 0, 1),
            (2, 1, 0, 0),
            (0, 1, 2, 1),
            (0, 1, 2, 0),
            (2, 1, 0, 1),
            (2, 1, 0, 0),
            (0, 1, 2, 1),
        ]
        
        perm, flip = transforms[o][:3], transforms[o][3]
        coords = [x, y, z]
        new_coords = [coords[perm[i]] for i in range(3)]
        
        if flip:
            new_coords[0] = self.N - 1 - new_coords[0]
        
        return new_coords[0], new_coords[1], new_coords[2]
    
    def h_to_xyz(self, h):
        h = int(h)
        x, y, z = 0, 0, 0
        
        for i in range(self.r):

            o = h & 0x7
            h >>= 3
            

            if o == 0:
                tx, ty, tz = z, x, y
                x, y, z = tx, ty, tz
                z = (1 << i) - 1 - z
            elif o == 1:
                x += (1 << i)
            elif o == 2:
                y += (1 << i)
                x, y = y, x
            elif o == 3:
                x += (1 << i)
                y += (1 << i)
            elif o == 4:
                z += (1 << i)
                x += (1 << i)
            elif o == 5:
                z += (1 << i)
                x += (1 << i)
                y, z = z, y
                y = (1 << (i + 1)) - 1 - y
            elif o == 6:
                z += (1 << i)
                y += (1 << i)
                x, z = z, x
                x = (1 << (i + 1)) - 1 - x
            elif o == 7:
                x += (1 << i)
                y += (1 << i)
                z += (1 << i)
                x, z = z, x
                z = (1 << (i + 1)) - 1 - z
        

        x = max(0, min(x, self.N - 1))
        y = max(0, min(y, self.N - 1))
        z = max(0, min(z, self.N - 1))
        
        return x, y, z
    
    def xyz_to_h(self, x, y, z):
        x, y, z = int(x), int(y), int(z)
        x = max(0, min(x, self.N - 1))
        y = max(0, min(y, self.N - 1))
        z = max(0, min(z, self.N - 1))
        
        h = 0
        
        for i in range(self.r - 1, -1, -1):
            h <<= 3
            mask = 1 << i
            

            ox = 1 if (x & mask) else 0
            oy = 1 if (y & mask) else 0
            oz = 1 if (z & mask) else 0
            

            if ox == 0 and oy == 0 and oz == 0:
                o = 0
            elif ox == 1 and oy == 0 and oz == 0:
                o = 1
            elif ox == 0 and oy == 1 and oz == 0:
                o = 2
            elif ox == 1 and oy == 1 and oz == 0:
                o = 3
            elif ox == 0 and oy == 0 and oz == 1:
                o = 4
            elif ox == 1 and oy == 0 and oz == 1:
                o = 5
            elif ox == 0 and oy == 1 and oz == 1:
                o = 6
            else:
                o = 7
            
            h |= o
        
        return h
    
    def generate_curve(self):
        n_points = self.N**3
        points = np.zeros((n_points, 3), dtype=int)
        
        for h in range(n_points):
            x, y, z = self.h_to_xyz(h)
            points[h, :] = [x, y, z]
        
        return points
    
    def locality_preservation_index(self, n_samples=1000):
        n_points = self.N**3
        
        if n_samples > n_points * (n_points - 1) // 2:
            n_samples = min(n_samples, 1000)
        
        h1 = np.random.randint(0, n_points, n_samples)
        h2 = np.random.randint(0, n_points, n_samples)
        
        d_h = np.abs(h1 - h2).astype(float)
        d_xyz = np.zeros(n_samples)
        
        for i in range(n_samples):
            x1, y1, z1 = self.h_to_xyz(h1[i])
            x2, y2, z2 = self.h_to_xyz(h2[i])
            d_xyz[i] = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
        

        if np.std(d_h) > 1.0e-12 and np.std(d_xyz) > 1.0e-12:
            lpi = np.corrcoef(d_h, d_xyz)[0, 1]
        else:
            lpi = 0.0
        
        return lpi


def ocean_volume_indexing(depth_levels, lat_levels, lon_levels, r=3):
    N = 1 << r
    

    depth_scale = (N - 1) / max(depth_levels - 1, 1)
    lat_scale = (N - 1) / max(lat_levels - 1, 1)
    lon_scale = (N - 1) / max(lon_levels - 1, 1)
    
    hc = HilbertCurve3D(r)
    
    return hc, depth_scale, lat_scale, lon_scale
