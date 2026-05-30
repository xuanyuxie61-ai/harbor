#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from itertools import permutations


class MatrixChainOptimizer:
    
    def __init__(self):
        pass
    
    def catalan_number(self, n):
        from math import factorial
        
        if n < 0:
            return 0
        
        return factorial(2 * n) // (factorial(n + 1) * factorial(n))
    
    def matrix_multiply_cost(self, dims, i, j, k):
        return dims[i - 1] * dims[j] * dims[k]
    
    def pivot_sequence_to_cost(self, n_mats, pivot_sequence, dims):


        mat_dims = [(dims[i], dims[i + 1]) for i in range(n_mats)]
        
        total_cost = 0
        


        available = list(range(n_mats))
        
        for pivot in pivot_sequence:
            if pivot >= len(available) - 1:
                continue
            
            left_idx = available[pivot]
            right_idx = available[pivot + 1]
            
            left_dim = mat_dims[left_idx]
            right_dim = mat_dims[right_idx]
            

            cost = left_dim[0] * left_dim[1] * right_dim[1]
            total_cost += cost
            

            new_dim = (left_dim[0], right_dim[1])
            

            new_idx = min(left_idx, right_idx)
            mat_dims[new_idx] = new_dim
            

            removed_idx = max(left_idx, right_idx)
            mat_dims[removed_idx] = (0, 0)
            

            available = [i for i in range(n_mats) if mat_dims[i][0] > 0]
        
        return total_cost
    
    def find_optimal_chain(self, dims):
        n_dims = len(dims)
        n_mats = n_dims - 1
        
        if n_mats == 1:
            return 0, []
        
        if any(d <= 0 for d in dims):
            return 0, list(range(n_mats - 1, 0, -1))
        
        min_cost = float('inf')
        optimal_order = []
        


        for perm in permutations(range(n_mats - 1)):
            cost = self.pivot_sequence_to_cost(n_mats, list(perm), dims)
            
            if cost < min_cost:
                min_cost = cost
                optimal_order = list(perm)
        
        return min_cost, optimal_order
    
    def optimal_matrix_power(self, M, power):
        if power == 0:
            return np.eye(M.shape[0]), 0
        if power == 1:
            return M.copy(), 0
        
        n_multiplies = 0
        

        result = np.eye(M.shape[0])
        base = M.copy()
        p = power
        
        while p > 0:
            if p % 2 == 1:
                result = result @ base
                n_multiplies += 1
            base = base @ base
            n_multiplies += 1
            p //= 2
        


        
        return result, n_multiplies
    
    def compute_state_transition_chain(self, state_matrices):
        if len(state_matrices) == 0:
            return None, 0
        if len(state_matrices) == 1:
            return state_matrices[0].copy(), 0
        

        dims = [state_matrices[0].shape[0]]
        for M in state_matrices:
            dims.append(M.shape[1])
        

        min_cost, optimal_order = self.find_optimal_chain(dims)
        


        result = state_matrices[0].copy()
        for i in range(1, len(state_matrices)):
            result = result @ state_matrices[i]
        
        return result, min_cost
