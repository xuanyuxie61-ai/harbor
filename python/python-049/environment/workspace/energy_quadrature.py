#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


class EnergyQuadrature:
    
    def __init__(self, x_grid, y_grid):
        self.x_grid = x_grid
        self.y_grid = y_grid
        self.dx = x_grid[1] - x_grid[0]
        self.dy = y_grid[1] - y_grid[0]
    
    def compute_total_energy(self, eta, u, v, h_bathy, rho=1025.0, g=9.81):






















        E_total = 0.0
        
        return E_total
    
    def compute_energy_flux(self, eta, u, v, h_bathy, rho=1025.0, g=9.81):
        H = h_bathy + eta
        flux_x = rho * g * eta * H * u
        flux_y = rho * g * eta * H * v
        

        flux_left = np.sum(flux_x[:, 0]) * self.dy
        flux_right = np.sum(flux_x[:, -1]) * self.dy
        flux_bottom = np.sum(flux_y[0, :]) * self.dx
        flux_top = np.sum(flux_y[-1, :]) * self.dx
        
        return flux_left, flux_right, flux_bottom, flux_top
    



    
    def square_monomial_integral(self, exponents):
        e1, e2 = exponents
        
        if e1 < 0 or e2 < 0:
            raise ValueError("指数必须非负")
        
        integral = 1.0 / ((e1 + 1) * (e2 + 1))
        return integral
    
    def test_square_quadrature(self, max_degree=5):
        errors = {}
        for m in range(max_degree + 1):
            for n in range(max_degree + 1):
                if m + n <= max_degree:
                    exact = self.square_monomial_integral((m, n))

                    numerical = self._numerical_square_integral(m, n)
                    error = abs(numerical - exact)
                    errors[(m, n)] = error
        return errors
    
    def _numerical_square_integral(self, m, n):
        N = 100
        x = np.linspace(0, 1, N)
        y = np.linspace(0, 1, N)
        dx = 1.0 / (N - 1)
        dy = 1.0 / (N - 1)
        
        integral = 0.0
        for yi in y:
            for xi in x:
                integral += (xi**m) * (yi**n) * dx * dy
        
        return integral
    



    
    def triangle_symmetric_quadrature(self, f, triangle_vertices, rule_degree=0):
        v1, v2, v3 = triangle_vertices
        

        if rule_degree == 0:

            xi_nodes = np.array([1.0/3.0])
            eta_nodes = np.array([1.0/3.0])
            weights = np.array([0.5])
        else:

            xi_nodes = np.array([1.0/3.0])
            eta_nodes = np.array([1.0/3.0])
            weights = np.array([0.5])
        

        integral = 0.0
        for i in range(len(weights)):

            s = np.array([xi_nodes[i], eta_nodes[i]])
            t = self._simplex_to_triangle(v1, v2, v3, s)
            

            J = abs((v2[0]-v1[0])*(v3[1]-v1[1]) - (v3[0]-v1[0])*(v2[1]-v1[1]))
            
            integral += weights[i] * f(t[0], t[1]) * J
        
        return integral
    
    def _simplex_to_triangle(self, v1, v2, v3, s):
        t = v1 * (1.0 - s[0] - s[1]) + v2 * s[0] + v3 * s[1]
        return t
    
    def triangle_symmetric_quadrature_test(self):
        triangle = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        f = lambda x, y: 1.0
        return self.triangle_symmetric_quadrature(f, triangle, rule_degree=0)
    



    
    def hermite_integral_exact(self, n, option=1):
        if n < 0:
            return 0.0
        
        if n % 2 == 1:
            return 0.0
        
        if option == 1:

            return self._double_factorial(n - 1) * np.sqrt(np.pi) / (2.0 ** (n / 2.0))
        else:
            return self._double_factorial(n - 1) * np.sqrt(np.pi) / (2.0 ** (n / 2.0))
    
    def _double_factorial(self, n):
        if n < 1:
            return 1.0
        
        value = 1.0
        while n > 1:
            value *= n
            n -= 2
        return value
    
    def gauss_hermite_quadrature(self, f, n_points):

        from numpy.polynomial.hermite import hermgauss
        
        x, w = hermgauss(n_points)
        result = np.sum(w * f(x))
        return result
    
    def test_hermite_exactness(self, max_degree=9, n_points=10):
        errors = {}
        
        for degree in range(max_degree + 1):
            exact = self.hermite_integral_exact(degree, option=1)
            

            f = lambda x: x ** degree
            numerical = self.gauss_hermite_quadrature(f, n_points)
            
            if exact == 0.0:
                error = abs(numerical)
            else:
                error = abs(numerical - exact) / abs(exact)
            
            errors[degree] = error
        
        return errors
    
    def compute_energy_by_triangle_quadrature(self, eta, u, v, h_bathy,
                                               rho=1025.0, g=9.81):
        E_total = 0.0
        
        for j in range(len(self.y_grid) - 1):
            for i in range(len(self.x_grid) - 1):

                p1 = np.array([self.x_grid[i], self.y_grid[j]])
                p2 = np.array([self.x_grid[i+1], self.y_grid[j]])
                p3 = np.array([self.x_grid[i], self.y_grid[j+1]])
                p4 = np.array([self.x_grid[i+1], self.y_grid[j+1]])
                

                def energy_density(x, y):

                    xi = (x - self.x_grid[i]) / self.dx
                    eta_val = (y - self.y_grid[j]) / self.dy
                    
                    e00 = 0.5 * rho * g * eta[j, i]**2
                    e10 = 0.5 * rho * g * eta[j, i+1]**2
                    e01 = 0.5 * rho * g * eta[j+1, i]**2
                    e11 = 0.5 * rho * g * eta[j+1, i+1]**2
                    
                    val = (1-xi)*(1-eta_val)*e00 + xi*(1-eta_val)*e10 + \
                          (1-xi)*eta_val*e01 + xi*eta_val*e11
                    return val
                

                E_total += self.triangle_symmetric_quadrature(
                    energy_density, np.array([p1, p2, p3]), rule_degree=0
                )
                

                E_total += self.triangle_symmetric_quadrature(
                    energy_density, np.array([p2, p4, p3]), rule_degree=0
                )
        
        return E_total
