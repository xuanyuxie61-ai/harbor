
import numpy as np


class OceanRegionGraph:
    
    def __init__(self):
        self.node_xy = []
        self.node_attrs = {}
        self.edges = []
        self.n_nodes = 0
        self.n_edges = 0
    
    def add_node(self, x, y, **attrs):
        node_id = self.n_nodes
        self.node_xy.append((x, y))
        for key, val in attrs.items():
            if key not in self.node_attrs:
                self.node_attrs[key] = [None] * self.n_nodes
            self.node_attrs[key].append(val)

        for key in self.node_attrs:
            while len(self.node_attrs[key]) < self.n_nodes + 1:
                self.node_attrs[key].append(None)
        self.n_nodes += 1
        return node_id
    
    def add_edge(self, i, j, weight=1.0):
        if i < 0 or i >= self.n_nodes or j < 0 or j >= self.n_nodes:
            raise ValueError("节点编号越界")
        if i == j:
            return

        if i > j:
            i, j = j, i
        self.edges.append((i, j, float(weight)))
        self.n_edges += 1
    
    def build_adjacency(self):

        adj_dict = {i: [] for i in range(self.n_nodes)}
        for i, j, w in self.edges:
            adj_dict[i].append((j, w))
            adj_dict[j].append((i, w))
        
        edge_pointer = np.zeros(self.n_nodes + 1, dtype=int)
        edge_data = []
        edge_weights = []
        
        for i in range(self.n_nodes):
            neighbors = sorted(adj_dict[i], key=lambda t: t[0])
            edge_pointer[i] = len(edge_data)
            for j, w in neighbors:
                edge_data.append(j)
                edge_weights.append(w)
        
        edge_pointer[self.n_nodes] = len(edge_data)
        
        return edge_pointer, np.array(edge_data, dtype=int), np.array(edge_weights)
    
    def connected_components(self):
        edge_pointer, edge_data, _ = self.build_adjacency()
        visited = [False] * self.n_nodes
        components = []
        
        def dfs(node, component):
            visited[node] = True
            component.append(node)
            start = edge_pointer[node]
            end = edge_pointer[node + 1]
            for idx in range(start, end):
                neighbor = edge_data[idx]
                if not visited[neighbor]:
                    dfs(neighbor, component)
        
        for i in range(self.n_nodes):
            if not visited[i]:
                comp = []
                dfs(i, comp)
                components.append(comp)
        
        return components
    
    def dijkstra_shortest_path(self, source):
        edge_pointer, edge_data, edge_weights = self.build_adjacency()
        

        dist = np.full(self.n_nodes, np.inf)
        prev = np.full(self.n_nodes, -1, dtype=int)
        dist[source] = 0.0
        
        unvisited = set(range(self.n_nodes))
        
        while unvisited:

            u = min(unvisited, key=lambda x: dist[x])
            unvisited.remove(u)
            
            if dist[u] == np.inf:
                break
            
            start = edge_pointer[u]
            end = edge_pointer[u + 1]
            for idx in range(start, end):
                v = edge_data[idx]
                w = edge_weights[idx]
                if w > 1e-15:
                    d = dist[u] + 1.0 / w
                    if d < dist[v]:
                        dist[v] = d
                        prev[v] = u
        
        return dist, prev
    
    def to_grf_string(self):
        edge_pointer, edge_data, _ = self.build_adjacency()
        lines = []
        for i in range(self.n_nodes):
            x, y = self.node_xy[i]
            start = edge_pointer[i]
            end = edge_pointer[i + 1]
            neighbors = edge_data[start:end]
            line = f"{i+1}  {x:.6f}  {y:.6f}  " + "  ".join(str(n+1) for n in neighbors)
            lines.append(line)
        return "\n".join(lines)
    
    @classmethod
    def from_grf_string(cls, grf_str):
        graph = cls()
        lines = grf_str.strip().split("\n")
        

        temp_nodes = []
        temp_edges = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            node_idx = int(parts[0]) - 1
            x = float(parts[1])
            y = float(parts[2])
            neighbors = [int(p) - 1 for p in parts[3:]]
            temp_nodes.append((node_idx, x, y))
            temp_edges.append(neighbors)
        

        temp_nodes.sort(key=lambda t: t[0])
        for idx, x, y in temp_nodes:
            graph.add_node(x, y)
        

        for idx, neighbors in enumerate(temp_edges):
            for nb in neighbors:
                if idx < nb:
                    graph.add_edge(idx, nb)
        
        return graph


def create_ocean_basin_graph(n_regions=12, basin_radius=500.0, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    graph = OceanRegionGraph()
    

    region_types = ['shelf', 'upwelling', 'eddy', 'abyssal']
    type_colors = {'shelf': 0, 'upwelling': 1, 'eddy': 2, 'abyssal': 3}
    
    for i in range(n_regions):
        angle = 2.0 * np.pi * i / n_regions
        r = basin_radius * (0.3 + 0.7 * np.random.rand())
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        
        rtype = region_types[i % len(region_types)]
        area = 1000.0 + 5000.0 * np.random.rand()
        dic_inv = 1e6 + 5e6 * np.random.rand()
        
        graph.add_node(x, y,
                       region_type=rtype,
                       area=area,
                       DIC_inventory=dic_inv)
    

    for i in range(n_regions):
        j = (i + 1) % n_regions
        dist = np.linalg.norm(np.array(graph.node_xy[i]) - np.array(graph.node_xy[j]))
        weight = max(0.1, 1000.0 / (dist + 1.0))
        graph.add_edge(i, j, weight)
        

        if np.random.rand() < 0.3:
            k = np.random.randint(0, n_regions)
            if k != i and k != j:
                dist2 = np.linalg.norm(np.array(graph.node_xy[i]) - np.array(graph.node_xy[k]))
                weight2 = max(0.1, 500.0 / (dist2 + 1.0))
                graph.add_edge(i, k, weight2)
    
    return graph


def carbon_transport_path_analysis(graph, source_region, sink_region):
    dist, prev = graph.dijkstra_shortest_path(source_region)
    
    if dist[sink_region] == np.inf:
        return {
            'path_exists': False,
            'path_length': np.inf,
            'path_nodes': [],
        }
    

    path = []
    node = sink_region
    while node != -1:
        path.append(node)
        node = prev[node]
    path = path[::-1]
    
    return {
        'path_exists': True,
        'path_length': dist[sink_region],
        'path_nodes': path,
    }
