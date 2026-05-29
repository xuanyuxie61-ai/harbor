#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 spherical_geodesics.py
 
 融合种子项目：
   - 1125_sphere_positive_distance：球面上随机点距离统计
 
 科学功能：
   球面测地距离计算。海啸在地球上长距离传播时，
   地球曲率效应不可忽略。本模块计算震中与各网格点之间的
   大圆距离（测地距离），用于海啸传播时间校正和能量衰减分析。
 
 核心物理公式：
 
   1) Haversine 公式（大圆距离）：
      
      给定两点 (φ₁, λ₁) 和 (φ₂, λ₂)，其中 φ 为纬度，λ 为经度：
      
      a = sin²(Δφ/2) + cos(φ₁)·cos(φ₂)·sin²(Δλ/2)
      c = 2·atan2(√a, √(1-a))
      d = R·c
      
      其中 R 为地球半径，d 为测地距离。
   
   2) 球面余弦定理：
      cos(c) = sin(φ₁)·sin(φ₂) + cos(φ₁)·cos(φ₂)·cos(Δλ)
      d = R·arccos(cos(c))
      
   3) 海啸在球面上的传播时间：
      T(φ, λ) = d(φ, λ) / √(g·h(φ, λ))
      
      其中 h(φ, λ) 为当地水深。
   
   4) 球面坐标与笛卡尔坐标转换：
      x = R·cos(φ)·cos(λ)
      y = R·cos(φ)·sin(λ)
      z = R·sin(φ)
      
   5) 欧氏距离与测地距离关系（小角度近似）：
      d_great_circle ≈ R · |p₁ - p₂| / R = |p₁ - p₂|
      
      但对于大距离，两者差异显著。
"""

import numpy as np


class SphericalGeodesics:
    """
    球面测地距离计算器。
    """
    
    def __init__(self, earth_radius=6371e3):
        """
        Parameters
        ----------
        earth_radius : float
            地球平均半径（m）
        """
        self.R = earth_radius
        
        if earth_radius <= 0:
            raise ValueError("地球半径必须为正")
    
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """
        使用 Haversine 公式计算两点间大圆距离。
        
        Parameters
        ----------
        lat1, lon1 : float
            点 1 的纬度和经度（度）
        lat2, lon2 : float
            点 2 的纬度和经度（度）
            
        Returns
        -------
        d : float
            测地距离（m）
        """
        # 角度转弧度
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        
        # Haversine 公式
        a = np.sin(dphi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0)**2
        # 边界处理：防止浮点误差导致 a > 1
        a = min(1.0, max(0.0, a))
        c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
        
        d = self.R * c
        return d
    
    def spherical_cosine_distance(self, lat1, lon1, lat2, lon2):
        """
        使用球面余弦定理计算大圆距离。
        """
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        dlambda = np.radians(lon2 - lon1)
        
        cos_c = np.sin(phi1) * np.sin(phi2) + np.cos(phi1) * np.cos(phi2) * np.cos(dlambda)
        # 边界处理
        cos_c = np.clip(cos_c, -1.0, 1.0)
        c = np.arccos(cos_c)
        
        return self.R * c
    
    def compute_grid_distances(self, x_grid, y_grid, epicenter_lat, epicenter_lon):
        """
        计算网格点到震中的球面距离。
        
        将笛卡尔坐标 (x, y) 近似为局部球面坐标：
          经度偏移：Δλ ≈ x / (R·cos(φ_epicenter))
          纬度偏移：Δφ ≈ y / R
        
        Parameters
        ----------
        x_grid, y_grid : ndarray
            相对于震中的局部笛卡尔坐标（m）
        epicenter_lat, epicenter_lon : float
            震中纬度和经度（度）
            
        Returns
        -------
        distances : ndarray
            球面距离场，形状 (ny, nx)，单位 m
        """
        nx = len(x_grid)
        ny = len(y_grid)
        distances = np.zeros((ny, nx))
        
        phi_epicenter = np.radians(epicenter_lat)
        
        for j in range(ny):
            for i in range(nx):
                # 局部坐标到经纬度偏移
                delta_lambda = x_grid[i] / (self.R * np.cos(phi_epicenter))
                delta_phi = y_grid[j] / self.R
                
                lat = epicenter_lat + np.degrees(delta_phi)
                lon = epicenter_lon + np.degrees(delta_lambda)
                
                distances[j, i] = self.haversine_distance(
                    epicenter_lat, epicenter_lon, lat, lon
                )
        
        return distances
    
    def compute_travel_time(self, distances, depth, g=9.81):
        """
        计算海啸传播时间。
        
        T = d / √(g·h)
        
        Parameters
        ----------
        distances : ndarray
            距离场（m）
        depth : ndarray
            水深场（m）
        g : float
            重力加速度
            
        Returns
        -------
        travel_time : ndarray
            传播时间场（s）
        """
        wave_speed = np.sqrt(g * np.maximum(depth, 1.0))
        travel_time = distances / wave_speed
        return travel_time
    
    def sample_sphere_distance_statistics(self, n_samples=1000):
        """
        球面上随机点对距离统计（融合 sphere_positive_distance_stats）。
        
        计算单位球面上 n 对随机点之间的欧氏距离统计量，
        用于验证球面距离算法的数值精度。
        
        Parameters
        ----------
        n_samples : int
            采样对数
            
        Returns
        -------
        mu : float
            平均距离
        var : float
            距离方差
        """
        distances = np.zeros(n_samples)
        
        for i in range(n_samples):
            # 在单位球面上均匀随机采样
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
        """
        在单位球面正半球上均匀随机采样一个点。
        
        来源于 sphere_positive_sample 的思想。
        """
        # 使用高斯分布生成各向同性点，然后归一化
        xyz = np.random.randn(3)
        xyz = xyz / np.linalg.norm(xyz)
        
        # 限制到正半球（z > 0）
        if xyz[2] < 0:
            xyz[2] = -xyz[2]
        
        return xyz
    
    def cartesian_to_geographic(self, x, y, z):
        """
        笛卡尔坐标转地理坐标。
        """
        r = np.sqrt(x**2 + y**2 + z**2)
        r = max(r, 1e-12)
        
        lat = np.degrees(np.arcsin(np.clip(z / r, -1.0, 1.0)))
        lon = np.degrees(np.arctan2(y, x))
        
        return lat, lon
    
    def geographic_to_cartesian(self, lat, lon):
        """
        地理坐标转笛卡尔坐标。
        """
        phi = np.radians(lat)
        lambda_ = np.radians(lon)
        
        x = self.R * np.cos(phi) * np.cos(lambda_)
        y = self.R * np.cos(phi) * np.sin(lambda_)
        z = self.R * np.sin(phi)
        
        return np.array([x, y, z])
