#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial import Delaunay


class TriangulationBoundaryDetector:
    
    def __init__(self):
        pass
    
    def detect_boundary_edges(self, triangles):
        triangles = np.asarray(triangles, dtype=int)
        n_tri = triangles.shape[0]
        
        if n_tri == 0:
            return np.array([]).reshape(0, 2)
        

        edge_num = 3 * n_tri
        edges = np.zeros((edge_num, 3), dtype=int)
        
        edges[0:n_tri, 0:2] = triangles[:, 0:2]
        edges[n_tri:2*n_tri, 0] = triangles[:, 1]
        edges[n_tri:2*n_tri, 1] = triangles[:, 2]
        edges[2*n_tri:3*n_tri, 0] = triangles[:, 2]
        edges[2*n_tri:3*n_tri, 1] = triangles[:, 0]
        

        edges[:, 2] = (edges[:, 0] < edges[:, 1]).astype(int)
        

        e1 = np.minimum(edges[:, 0], edges[:, 1])
        e2 = np.maximum(edges[:, 0], edges[:, 1])
        edges[:, 0] = e1
        edges[:, 1] = e2
        

        sort_idx = np.lexsort((edges[:, 1], edges[:, 0]))
        edges = edges[sort_idx]
        

        boundary_edges = []
        e = 0
        while e < edge_num:
            if e == edge_num - 1:

                be = edges[e]
                e += 1
            else:
                if edges[e, 0] == edges[e+1, 0] and edges[e, 1] == edges[e+1, 1]:

                    e += 2
                    continue
                else:
                    be = edges[e]
                    e += 1
            

            if be[2] == 1:
                boundary_edges.append([be[0], be[1]])
            else:
                boundary_edges.append([be[1], be[0]])
        
        return np.array(boundary_edges, dtype=int)
    
    def construct_boundary_path(self, boundary_edges):
        if boundary_edges.size == 0:
            return []
        
        n_be = boundary_edges.shape[0]
        used = np.zeros(n_be, dtype=bool)
        paths = []
        
        while not np.all(used):

            start_idx = np.where(~used)[0][0]
            path = [boundary_edges[start_idx, 0]]
            current = boundary_edges[start_idx, 1]
            used[start_idx] = True
            path.append(current)
            

            while True:
                found = False
                for i in range(n_be):
                    if used[i]:
                        continue
                    if boundary_edges[i, 0] == current:
                        current = boundary_edges[i, 1]
                        used[i] = True
                        path.append(current)
                        found = True
                        break
                if not found:
                    break
                if current == path[0]:
                    break
            
            paths.append(path)
        
        return paths
    
    def detect_boundary(self, points_2d):
        points = np.asarray(points_2d, dtype=float)
        n = points.shape[0]
        
        if n < 3:
            return set(range(n))
        
        try:
            tri = Delaunay(points)
            triangles = tri.simplices
        except Exception:

            boundary_nodes = set()
            for dim in range(2):
                boundary_nodes.add(int(np.argmin(points[:, dim])))
                boundary_nodes.add(int(np.argmax(points[:, dim])))
            return boundary_nodes
        
        boundary_edges = self.detect_boundary_edges(triangles)
        if boundary_edges.size == 0:
            return set()
        
        boundary_nodes = set(boundary_edges.flatten())
        return boundary_nodes
