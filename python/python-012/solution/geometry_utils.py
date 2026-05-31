"""
Sample Geometry and Boundary Definition for TI Nanostructures
==============================================================
Defines the geometric boundaries and shapes for topological insulator
surface state simulations, including:
- Hexagonal sample shapes (relevant for Bi2Se3, Bi2Te3 crystal structure)
- Tortoise-grid boundary tracing (project 1282)
- Ellipsoidal cross-sections

The hexagonal warping of Bi2Te3 is tied to the crystal symmetry (R-3m),
and sample geometries with hexagonal boundaries exhibit unique
edge state distributions.

Boundary word encoding (inspired by tortoise_grid_word, project 1282):
    A = East, C = NNE, E = NNW, G = West, I = SSW, K = SSE
    with intermediate directions for hexagonal symmetry.
"""

import numpy as np


class SampleGeometry:
    """
    Defines sample geometries for TI surface state simulations.
    """

    def __init__(self, size=100.0, shape='square'):
        """
        Parameters
        ----------
        size : float
            Characteristic size in nm.
        shape : str
            'square', 'hexagon', 'rectangle', 'circle'.
        """
        self.size = size
        self.shape = shape

    def hexagon_vertices(self):
        """
        Generate vertices of a regular hexagon.

        For a hexagon of side length L:
            vertices = L * [cos(n*pi/3), sin(n*pi/3)]

        Returns
        -------
        vertices : ndarray
            Shape (6, 2).
        """
        L = self.size
        angles = np.linspace(0.0, 2.0 * np.pi, 7)[:-1]
        vertices = np.column_stack((L * np.cos(angles), L * np.sin(angles)))
        return vertices

    def boundary_word_trace(self, word, step_sizes=None):
        """
        Trace a boundary from a word encoding (inspired by project 1282).

        Directions (hexagonal symmetry, 6 primary + 6 secondary):
            A: 0°,    B: 30°,   C: 60°,   D: 90°
            E: 120°,  F: 150°,  G: 180°,  H: 210°
            I: 240°,  J: 270°,  K: 300°,  L: 330°

        Parameters
        ----------
        word : str
            String of direction letters.
        step_sizes : dict, optional
            Step size for each direction.

        Returns
        -------
        path : ndarray
            Shape (N, 2) path coordinates.
        """
        if step_sizes is None:
            step_sizes = {c: 1.0 for c in 'ABCDEFGHIJKLabcdefghijkl'}

        # Direction angles in radians
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
        """
        Generate the tortoise-grid boundary word from project 1282
        as a sample geometry. This creates a complex non-convex
        boundary useful for studying edge state interference.

        Returns
        -------
        path : ndarray
        """
        # Boundary word from tortoise_grid_word.m
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
        """
        Ray casting algorithm to test if a point is inside a polygon.

        Parameters
        ----------
        point : tuple
            (x, y).
        polygon : ndarray
            Shape (N, 2).

        Returns
        -------
        inside : bool
        """
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
        """
        Generate a grid of points inside the sample shape.

        Parameters
        ----------
        nx, ny : int

        Returns
        -------
        points : ndarray
            Shape (N, 2) points inside the shape.
        mask : ndarray
            Shape (nx, ny) boolean mask.
        """
        if self.shape == 'hexagon':
            vertices = self.hexagon_vertices()
        elif self.shape == 'tortoise':
            vertices = self.tortoise_boundary()
        elif self.shape == 'circle':
            pass
        else:
            # Square
            x = np.linspace(-self.size, self.size, nx)
            y = np.linspace(-self.size, self.size, ny)
            X, Y = np.meshgrid(x, y)
            points = np.column_stack((X.ravel(), Y.ravel()))
            mask = np.ones((nx, ny), dtype=bool)
            return points, mask

        # For polygon shapes
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
        """
        Compute the total boundary length.

        Returns
        -------
        length : float
        """
        if self.shape == 'hexagon':
            return 6.0 * self.size
        elif self.shape == 'circle':
            return 2.0 * np.pi * self.size
        elif self.shape == 'square':
            return 8.0 * self.size
        else:
            # Approximate from polygon
            vertices = self.tortoise_boundary()
            diffs = np.diff(vertices, axis=0)
            return np.sum(np.sqrt(np.sum(diffs ** 2, axis=1)))

    def area(self):
        """
        Compute the sample area using the shoelace formula.

        Returns
        -------
        area : float
        """
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
