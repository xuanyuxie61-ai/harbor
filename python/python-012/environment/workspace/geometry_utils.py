
import numpy as np


class SampleGeometry:

    def __init__(self, size=100.0, shape='square'):
        self.size = size
        self.shape = shape

    def hexagon_vertices(self):
        L = self.size
        angles = np.linspace(0.0, 2.0 * np.pi, 7)[:-1]
        vertices = np.column_stack((L * np.cos(angles), L * np.sin(angles)))
        return vertices

    def boundary_word_trace(self, word, step_sizes=None):
        if step_sizes is None:
            step_sizes = {c: 1.0 for c in 'ABCDEFGHIJKLabcdefghijkl'}


        angles = {
            'A': 0.0, 'a': 0.0,
            'B': np.pi / 6.0, 'b': np.pi / 6.0,
            'C': np.pi / 3.0, 'c': np.pi / 3.0,
            'D': np.pi / 2.0, 'd': np.pi / 2.0,
            'E': 2.0 * np.pi / 3.0, 'e': 2.0 * np.pi / 3.0,
            'F': 5.0 * np.pi / 6.0, 'f': 5.0 * np.pi / 6.0,
            'G': np.pi, 'g': np.pi,
            'H': 7.0 * np.pi / 6.0, 'h': 7.0 * np.pi / 6.0,
            'I': 4.0 * np.pi / 3.0, 'i': 4.0 * np.pi / 3.0,
            'J': 3.0 * np.pi / 2.0, 'j': 3.0 * np.pi / 2.0,
            'K': 5.0 * np.pi / 3.0, 'k': 5.0 * np.pi / 3.0,
            'L': 11.0 * np.pi / 6.0, 'l': 11.0 * np.pi / 6.0,
        }

        path = [(0.0, 0.0)]
        x, y = 0.0, 0.0

        for char in word:
            if char not in angles:
                continue
            theta = angles[char]
            step = step_sizes.get(char, 1.0)
            x += step * np.cos(theta)
            y += step * np.sin(theta)
            path.append((x, y))

        return np.array(path)

    def tortoise_boundary(self):

        word = ('AAAAAAAADdGEEEEEEDdGEEEEEEEEEEEEEEEEEEEEEECCC'
                'fFFffFFffFEEIfFIIIIJjjJJjjJJjjJJjjJKKKKKKKKKK'
                'KKKKLllLLllLLllLLllL')
        step_sizes = {
            'A': 0.5, 'a': 0.5,
            'B': np.sqrt(3.0) / 3.0, 'b': np.sqrt(3.0) / 6.0,
            'C': 0.5, 'c': 0.5,
            'D': np.sqrt(3.0) / 3.0, 'd': np.sqrt(3.0) / 6.0,
            'E': 0.5, 'e': 0.5,
            'F': np.sqrt(3.0) / 3.0, 'f': np.sqrt(3.0) / 6.0,
            'G': 0.5, 'g': 0.5,
            'H': np.sqrt(3.0) / 3.0, 'h': np.sqrt(3.0) / 6.0,
            'I': 0.5, 'i': 0.5,
            'J': np.sqrt(3.0) / 3.0, 'j': np.sqrt(3.0) / 6.0,
            'K': 0.5, 'k': 0.5,
            'L': np.sqrt(3.0) / 3.0, 'l': np.sqrt(3.0) / 6.0,
        }
        return self.boundary_word_trace(word, step_sizes)

    def point_in_polygon(self, point, polygon):
        x, y = point
        n = len(polygon)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > y) != (yj > y)) and \
               (x < (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi):
                inside = not inside
            j = i
        return inside

    def generate_grid_in_shape(self, nx=50, ny=50):
        if self.shape == 'hexagon':
            vertices = self.hexagon_vertices()
        elif self.shape == 'tortoise':
            vertices = self.tortoise_boundary()
        elif self.shape == 'circle':
            pass
        else:

            x = np.linspace(-self.size, self.size, nx)
            y = np.linspace(-self.size, self.size, ny)
            X, Y = np.meshgrid(x, y)
            points = np.column_stack((X.ravel(), Y.ravel()))
            mask = np.ones((nx, ny), dtype=bool)
            return points, mask


        x = np.linspace(-2.0 * self.size, 2.0 * self.size, nx)
        y = np.linspace(-2.0 * self.size, 2.0 * self.size, ny)
        X, Y = np.meshgrid(x, y)
        mask = np.zeros((nx, ny), dtype=bool)

        for i in range(nx):
            for j in range(ny):
                mask[i, j] = self.point_in_polygon((X[i, j], Y[i, j]), vertices)

        points = np.column_stack((X[mask], Y[mask]))
        return points, mask

    def edge_length(self):
        if self.shape == 'hexagon':
            return 6.0 * self.size
        elif self.shape == 'circle':
            return 2.0 * np.pi * self.size
        elif self.shape == 'square':
            return 8.0 * self.size
        else:

            vertices = self.tortoise_boundary()
            diffs = np.diff(vertices, axis=0)
            return np.sum(np.sqrt(np.sum(diffs ** 2, axis=1)))

    def area(self):
        if self.shape == 'hexagon':
            return 3.0 * np.sqrt(3.0) / 2.0 * self.size ** 2
        elif self.shape == 'circle':
            return np.pi * self.size ** 2
        elif self.shape == 'square':
            return (2.0 * self.size) ** 2
        else:
            vertices = self.tortoise_boundary()
            x = vertices[:, 0]
            y = vertices[:, 1]
            return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
