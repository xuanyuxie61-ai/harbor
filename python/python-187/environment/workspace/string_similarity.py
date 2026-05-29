#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
string_similarity.py
====================

基于种子项目 669_levenshtein_matrix 的字符串相似性引擎。

科学背景
--------
在推荐系统中，冷启动问题（cold-start）是一个核心挑战：
新物品缺乏交互历史，无法通过协同过滤获得推荐。
利用物品的文本元数据（标题、描述、标签）进行内容-based
相似性匹配是经典解决方案。

Levenshtein 编辑距离:
    两个字符串 s（长度 m）和 t（长度 n）之间的最小编辑操作数：
    - 插入（insertion）
    - 删除（deletion）
    - 替换（substitution）
    
动态规划递推:
    d[i,j] = min(
        d[i-1, j]   + 1,    # 删除 s[i]
        d[i, j-1]   + 1,    # 插入 t[j]
        d[i-1, j-1] + cost  # 替换或匹配
    )
    cost = 0 if s[i-1] == t[j-1] else 1
    
    初值:
        d[0,j] = j,  d[i,0] = i
        
相似度转换:
    sim(s,t) = 1 - d[m,n] / max(m, n)
    
在中文场景下，每个字符作为一个 token。
"""

import numpy as np


class StringSimilarityEngine:
    """
    Levenshtein 编辑距离相似性引擎。
    """
    
    def __init__(self):
        pass
    
    def levenshtein_distance(self, s, t):
        """
        计算字符串 s 和 t 之间的 Levenshtein 编辑距离。
        
        算法:
            动态规划，时间复杂度 O(m·n)，空间复杂度 O(m·n)。
            
        边界保护:
            - 空字符串距离为另一字符串长度
            - 输入非字符串时尝试转换为字符串
        """
        s = str(s)
        t = str(t)
        m, n = len(s), len(t)
        
        if m == 0:
            return n
        if n == 0:
            return m
        
        # 使用完整矩阵以支持后续回溯（如需要）
        d = np.zeros((m + 1, n + 1), dtype=int)
        
        for i in range(m + 1):
            d[i, 0] = i
        for j in range(n + 1):
            d[0, j] = j
        
        for j in range(1, n + 1):
            for i in range(1, m + 1):
                substitution_cost = 0 if s[i - 1] == t[j - 1] else 1
                d[i, j] = min(
                    d[i - 1, j] + 1,      # 删除
                    min(
                        d[i, j - 1] + 1,  # 插入
                        d[i - 1, j - 1] + substitution_cost  # 替换/匹配
                    )
                )
        
        return int(d[m, n])
    
    def similarity(self, s, t):
        """
        将编辑距离转换为归一化相似度 [0, 1]。
        
        公式:
            sim = 1 - d / max(len(s), len(t))
            
        边界保护:
            - 两字符串均为空时返回 1.0
        """
        dist = self.levenshtein_distance(s, t)
        maxlen = max(len(str(s)), len(str(t)))
        if maxlen == 0:
            return 1.0
        return 1.0 - dist / maxlen
    
    def compute_similarity_matrix(self, strings):
        """
        计算字符串列表的成对相似度矩阵。
        
        返回:
            S[i,j] = similarity(strings[i], strings[j])
            
        优化:
            对称矩阵，只计算上三角。
        """
        n = len(strings)
        if n == 0:
            return np.array([])
        
        S = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                sim = self.similarity(strings[i], strings[j])
                S[i, j] = sim
                S[j, i] = sim
        
        return S
    
    def find_similar_items(self, query, candidates, top_k=3):
        """
        为查询字符串在候选集中找到最相似的 top_k 个。
        
        返回:
            list of (idx, similarity_score)
        """
        sims = [(i, self.similarity(query, cand))
                for i, cand in enumerate(candidates)]
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:top_k]
