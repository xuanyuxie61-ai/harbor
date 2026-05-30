#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import norm


class TruncatedNormalRatingModel:
    
    def __init__(self, mu=3.0, sigma=1.2, a=1.0, b=5.0):
        self.mu = float(mu)
        self.sigma = max(float(sigma), 1e-6)
        self.a = float(a)
        self.b = float(b)
        

        self.alpha = (self.a - self.mu) / self.sigma
        self.beta = (self.b - self.mu) / self.sigma
        

        self.alpha_cdf = norm.cdf(self.alpha)
        self.beta_cdf = norm.cdf(self.beta)
        self.alpha_pdf = norm.pdf(self.alpha)
        self.beta_pdf = norm.pdf(self.beta)
        

        self.Z = self.beta_cdf - self.alpha_cdf
        if self.Z < 1e-12:
            self.Z = 1e-12
    
    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        z = (x - self.mu) / self.sigma
        density = norm.pdf(z) / (self.sigma * self.Z)

        density = np.where((x >= self.a) & (x <= self.b), density, 0.0)
        return density
    
    def mean(self):
        return self.mu + self.sigma * (self.alpha_pdf - self.beta_pdf) / self.Z
    
    def variance(self):
        term1 = 1.0
        term2 = (self.alpha * self.alpha_pdf - self.beta * self.beta_pdf) / self.Z
        term3 = ((self.alpha_pdf - self.beta_pdf) / self.Z) ** 2
        return self.sigma ** 2 * (term1 + term2 - term3)
    
    def sample(self, size=1):
        rng = np.random.RandomState(42)
        size = int(size)
        if size <= 0:
            return np.array([])
        
        u = rng.rand(size)
        xi_cdf = self.alpha_cdf + u * (self.beta_cdf - self.alpha_cdf)
        xi = norm.ppf(xi_cdf)
        samples = self.mu + self.sigma * xi
        

        samples = np.clip(samples, self.a, self.b)
        return samples
    
    def expected_rating(self, predicted_mean, clip=True):

        old_mu = self.mu
        self.mu = predicted_mean
        self.alpha = (self.a - self.mu) / self.sigma
        self.beta = (self.b - self.mu) / self.sigma
        self.alpha_cdf = norm.cdf(self.alpha)
        self.beta_cdf = norm.cdf(self.beta)
        self.alpha_pdf = norm.pdf(self.alpha)
        self.beta_pdf = norm.pdf(self.beta)
        self.Z = max(self.beta_cdf - self.alpha_cdf, 1e-12)
        
        e = self.mean()
        

        self.mu = old_mu
        self.alpha = (self.a - self.mu) / self.sigma
        self.beta = (self.b - self.mu) / self.sigma
        self.alpha_cdf = norm.cdf(self.alpha)
        self.beta_cdf = norm.cdf(self.beta)
        self.alpha_pdf = norm.pdf(self.alpha)
        self.beta_pdf = norm.pdf(self.beta)
        self.Z = max(self.beta_cdf - self.alpha_cdf, 1e-12)
        
        if clip:
            e = np.clip(e, self.a, self.b)
        return float(e)
