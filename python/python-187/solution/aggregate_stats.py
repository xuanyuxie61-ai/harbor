#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aggregate_stats.py
==================

基于种子项目 347_faces_average 和 118_brc_naive 的聚合统计模块。

科学背景
--------
在推荐系统中，聚合统计用于：
1. 用户分组画像：将相似用户聚类后，计算每组的统计特征
2. 全局/局部均值填充：作为基线估计
3. 异常检测：通过组内标准差识别异常评分

聚合公式:
    对于组 G 中的评分集合 {r_i}:
        μ_G = (1/|G|) Σ r_i
        σ_G² = (1/(|G|-1)) Σ (r_i - μ_G)²
        min_G = min{r_i}
        max_G = max{r_i}
        
在图像平均（faces_average）的启发下，我们将多个用户的潜向量
通过 K-means 聚类进行"平均"，形成组代表（centroid）。

BRC（Billion Record Challenge）风格聚合:
    高效流式处理大量记录，维护运行统计量:
        count_{new} = count_{old} + 1
        sum_{new}   = sum_{old} + x
        mean_{new}  = sum_{new} / count_{new}
        min_{new}   = min(min_{old}, x)
        max_{new}   = max(max_{old}, x)
"""

import numpy as np


class AggregateStatistics:
    """
    聚合统计引擎。
    """
    
    def __init__(self):
        pass
    
    def group_users_by_embedding(self, embeddings, n_clusters=4):
        """
        使用 K-means 风格的 Lloyd 算法将用户按潜向量分组。
        
        算法:
            1. 随机初始化质心
            2. 迭代直到收敛:
               a. 分配步：每个点分配到最近质心
               b. 更新步：质心 = 组内均值
               
        边界保护:
            - n_clusters > n_points 时自动减少
        """
        X = np.asarray(embeddings, dtype=float)
        n = X.shape[0]
        k = min(n_clusters, n)
        
        rng = np.random.RandomState(42)
        # K-means++ 初始化
        centroids = [X[rng.randint(n)]]
        for _ in range(1, k):
            dists = np.array([min(np.linalg.norm(x - c)**2 for c in centroids) for x in X])
            probs = dists / (dists.sum() + 1e-12)
            centroids.append(X[rng.choice(n, p=probs)])
        
        centroids = np.array(centroids)
        
        # Lloyd 迭代
        labels = np.zeros(n, dtype=int)
        for _ in range(30):
            # 分配
            new_labels = np.array([
                int(np.argmin([np.linalg.norm(X[i] - centroids[j]) for j in range(k)]))
                for i in range(n)
            ])
            
            # 更新
            for j in range(k):
                members = X[new_labels == j]
                if len(members) > 0:
                    centroids[j] = members.mean(axis=0)
            
            if np.array_equal(labels, new_labels):
                break
            labels = new_labels
        
        return labels
    
    def compute_group_statistics(self, R_matrix, group_labels):
        """
        计算每组的聚合统计量。
        
        返回:
            dict: group_id -> {mean, std, min, max, count}
            
        边界保护:
            - 忽略 NaN 值
            - 单元素组标准差设为 0
        """
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
        """
        BRC 风格的流式聚合。
        
        输入:
            values : iterable of (key, value)
            
        返回:
            stats  : dict of key -> {sum, mean, min, max, count}
        """
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
        """
        平均一组嵌入向量（受 faces_average 启发）。
        
        公式:
            e_avg = (1/N) Σ e_i
            
        边界保护:
            - 空输入返回零向量
        """
        E = np.asarray(embeddings, dtype=float)
        if E.size == 0:
            return np.array([])
        return E.mean(axis=0)
