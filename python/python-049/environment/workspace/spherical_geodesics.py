#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


class SphericalGeodesics:
    
    def __init__(self, earth_radius=6371e3):
        self.R = earth_radius
        
        if earth_radius <= 0:
            raise ValueError("地球半径必须为正")
    
    def haversine_distance(self, lat1, lon1, lat2, lon2):

        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        

        a = np.sin(dphi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0)**2

        a = min(1.0, max(0.0, a))
        c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
        
        d = self.R * c
        return d
    
    def spherical_cosine_distance(self, lat1, lon1, lat2, lon2):
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        dlambda = np.radians(lon2 - lon1)
        
        cos_c = np.sin(phi1) * np.sin(phi2) + np.cos(phi1) * np.cos(phi2) * np.cos(dlambda)

        cos_c = np.clip(cos_c, -1.0, 1.0)
        c = np.arccos(cos_c)
        
        return self.R * c
    
    def compute_grid_distances(self, x_grid, y_grid, epicenter_lat, epicenter_lon):
        nx = len(x_grid)
        ny = len(y_grid)
        distances = np.zeros((ny, nx))
        
        phi_epicenter = np.radians(epicenter_lat)
        
        for j in range(ny):
            for i in range(nx):

                delta_lambda = x_grid[i] / (self.R * np.cos(phi_epicenter))
                delta_phi = y_grid[j] / self.R
                
                lat = epicenter_lat + np.degrees(delta_phi)
                lon = epicenter_lon + np.degrees(delta_lambda)
                
                distances[j, i] = self.haversine_distance(
                    epicenter_lat, epicenter_lon, lat, lon
                )
        
        return distances
    
    def compute_travel_time(self, distances, depth, g=9.81):
        wave_speed = np.sqrt(g * np.maximum(depth, 1.0))
        travel_time = distances / wave_speed
        return travel_time
    
    def sample_sphere_distance_statistics(self, n_samples=1000):
        distances = np.zeros(n_samples)
        
        for i in range(n_samples):

            p = self._sample_unit_sphere_positive()
            q = self._sample_unit_sphere_positive()
            distances[i] = np.linalg.norm(p - q)
        
        mu = np.mean(distances)
        if n_samples > 1:
            var = np.sum((distances - mu)**2) / (n_samples - 1)
        else:
            var = 0.0
        
        return mu, var
    
    def _sample_unit_sphere_positive(self):

        xyz = np.random.randn(3)
        xyz = xyz / np.linalg.norm(xyz)
        

        if xyz[2] < 0:
            xyz[2] = -xyz[2]
        
        return xyz
    
    def cartesian_to_geographic(self, x, y, z):
        r = np.sqrt(x**2 + y**2 + z**2)
        r = max(r, 1e-12)
        
        lat = np.degrees(np.arcsin(np.clip(z / r, -1.0, 1.0)))
        lon = np.degrees(np.arctan2(y, x))
        
        return lat, lon
    
    def geographic_to_cartesian(self, lat, lon):
        phi = np.radians(lat)
        lambda_ = np.radians(lon)
        
        x = self.R * np.cos(phi) * np.cos(lambda_)
        y = self.R * np.cos(phi) * np.sin(lambda_)
        z = self.R * np.sin(phi)
        
        return np.array([x, y, z])
