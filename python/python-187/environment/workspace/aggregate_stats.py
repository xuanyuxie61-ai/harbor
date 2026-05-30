#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


class AggregateStatistics:
    
    def __init__(self):
        pass
    
    def group_users_by_embedding(self, embeddings, n_clusters=4):
        X = np.asarray(embeddings, dtype=float)
        n = X.shape[0]
        k = min(n_clusters, n)
        
        rng = np.random.RandomState(42)

        centroids = [X[rng.randint(n)]]
        for _ in range(1, k):
            dists = np.array([min(np.linalg.norm(x - c)**2 for c in centroids) for x in X])
            probs = dists / (dists.sum() + 1e-12)
            centroids.append(X[rng.choice(n, p=probs)])
        
        centroids = np.array(centroids)
        

        labels = np.zeros(n, dtype=int)
        for _ in range(30):

            new_labels = np.array([
                int(np.argmin([np.linalg.norm(X[i] - centroids[j]) for j in range(k)]))
                for i in range(n)
            ])
            

            for j in range(k):
                members = X[new_labels == j]
                if len(members) > 0:
                    centroids[j] = members.mean(axis=0)
            
            if np.array_equal(labels, new_labels):
                break
            labels = new_labels
        
        return labels
    
    def compute_group_statistics(self, R_matrix, group_labels):
        R = np.array(R_matrix, dtype=float)
        labels = np.asarray(group_labels)
        
        groups = {}
        unique_labels = np.unique(labels)
        
        for gid in unique_labels:
            mask = labels == gid
            ratings = R[mask, :].flatten()
            ratings = ratings[~np.isnan(ratings)]
            
            if len(ratings) == 0:
                groups[int(gid)] = {
                    'mean': 3.0, 'std': 0.0,
                    'min': 1.0, 'max': 5.0,
                    'count': 0
                }
                continue
            
            mean_val = float(np.mean(ratings))
            if len(ratings) > 1:
                std_val = float(np.std(ratings, ddof=1))
            else:
                std_val = 0.0
            
            groups[int(gid)] = {
                'mean': mean_val,
                'std': std_val,
                'min': float(np.min(ratings)),
                'max': float(np.max(ratings)),
                'count': int(len(ratings))
            }
        
        return groups
    
    def running_aggregate(self, values):
        stats = {}
        for key, val in values:
            val = float(val)
            if key not in stats:
                stats[key] = {
                    'sum': val,
                    'count': 1,
                    'min': val,
                    'max': val
                }
            else:
                s = stats[key]
                s['sum'] += val
                s['count'] += 1
                s['min'] = min(s['min'], val)
                s['max'] = max(s['max'], val)
        
        for key in stats:
            s = stats[key]
            s['mean'] = s['sum'] / max(s['count'], 1)
        
        return stats
    
    def average_embeddings(self, embeddings):
        E = np.asarray(embeddings, dtype=float)
        if E.size == 0:
            return np.array([])
        return E.mean(axis=0)
