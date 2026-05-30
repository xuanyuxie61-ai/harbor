#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


class StringSimilarityEngine:
    
    def __init__(self):
        pass
    
    def levenshtein_distance(self, s, t):
        s = str(s)
        t = str(t)
        m, n = len(s), len(t)
        
        if m == 0:
            return n
        if n == 0:
            return m
        

        d = np.zeros((m + 1, n + 1), dtype=int)
        
        for i in range(m + 1):
            d[i, 0] = i
        for j in range(n + 1):
            d[0, j] = j
        
        for j in range(1, n + 1):
            for i in range(1, m + 1):
                substitution_cost = 0 if s[i - 1] == t[j - 1] else 1
                d[i, j] = min(
                    d[i - 1, j] + 1,
                    min(
                        d[i, j - 1] + 1,
                        d[i - 1, j - 1] + substitution_cost
                    )
                )
        
        return int(d[m, n])
    
    def similarity(self, s, t):
        dist = self.levenshtein_distance(s, t)
        maxlen = max(len(str(s)), len(str(t)))
        if maxlen == 0:
            return 1.0
        return 1.0 - dist / maxlen
    
    def compute_similarity_matrix(self, strings):
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
        sims = [(i, self.similarity(query, cand))
                for i, cand in enumerate(candidates)]
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:top_k]
