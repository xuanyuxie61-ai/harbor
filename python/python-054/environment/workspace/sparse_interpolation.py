
import numpy as np


def shepard_interp_nd(data_coords, data_values, p, query_coords):
    m, nd = data_coords.shape
    _, ni = query_coords.shape
    
    if nd == 0:
        return np.zeros(ni)
    
    interp_values = np.zeros(ni)
    
    for q in range(ni):
        xq = query_coords[:, q]
        

        diff = data_coords - xq[:, np.newaxis]
        dists = np.sqrt(np.sum(diff**2, axis=0))
        

        min_dist = np.min(dists)
        if min_dist < 1e-12:
            idx = np.argmin(dists)
            interp_values[q] = data_values[idx]
            continue
        

        if p == 0.0:
            weights = np.ones(nd) / nd
        else:
            weights = dists**(-p)

            w_max = np.max(weights)
            if w_max > 1e30:
                weights = np.where(weights > 1e-30, weights, 0.0)
            w_sum = np.sum(weights)
            if w_sum < 1e-30:
                weights = np.ones(nd) / nd
            else:
                weights = weights / w_sum
        
        interp_values[q] = np.dot(weights, data_values)
    
    return interp_values


def shepard_interp_3d_ocean(data_lons, data_lats, data_depths, data_values,
                            query_lons, query_lats, query_depths, p=2.0,
                            lon_scale=1.0, lat_scale=1.0, depth_scale=1.0):
    nd = len(data_lons)
    ni = len(query_lons)
    
    data_coords = np.zeros((3, nd))
    data_coords[0, :] = data_lons / lon_scale
    data_coords[1, :] = data_lats / lat_scale
    data_coords[2, :] = data_depths / depth_scale
    
    query_coords = np.zeros((3, ni))
    query_coords[0, :] = query_lons / lon_scale
    query_coords[1, :] = query_lats / lat_scale
    query_coords[2, :] = query_depths / depth_scale
    
    return shepard_interp_nd(data_coords, data_values, p, query_coords)


def kriging_like_shepard_residual(data_coords, data_values, trend_func, p=2.0):

    nd = data_coords.shape[1]
    trend_vals = np.zeros(nd)
    for i in range(nd):
        trend_vals[i] = trend_func(data_coords[:, i])
    
    residuals = data_values - trend_vals
    
    def interp_func(query_coords):
        ni = query_coords.shape[1]
        trend_at_query = np.zeros(ni)
        for q in range(ni):
            trend_at_query[q] = trend_func(query_coords[:, q])
        
        residual_interp = shepard_interp_nd(data_coords, residuals, p, query_coords)
        return trend_at_query + residual_interp
    
    return interp_func


def cross_validate_shepard(data_coords, data_values, p_values=[1.0, 2.0, 3.0, 4.0]):
    nd = data_coords.shape[1]
    if nd < 3:
        return {'best_p': 2.0, 'rmse_by_p': {2.0: 0.0}}
    
    rmse_by_p = {}
    
    for p in p_values:
        errors = []
        for i in range(nd):

            mask = np.ones(nd, dtype=bool)
            mask[i] = False
            train_coords = data_coords[:, mask]
            train_vals = data_values[mask]
            
            query = data_coords[:, i:i+1]
            pred = shepard_interp_nd(train_coords, train_vals, p, query)
            errors.append((pred[0] - data_values[i])**2)
        
        rmse = np.sqrt(np.mean(errors))
        rmse_by_p[p] = rmse
    
    best_p = min(rmse_by_p, key=rmse_by_p.get)
    return {'best_p': best_p, 'rmse_by_p': rmse_by_p}


def generate_sparse_ocean_observations(n_points=50, depth_range=(0, 4000),
                                        lat_range=(20, 60), lon_range=(-80, -20),
                                        seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    lons = np.random.uniform(lon_range[0], lon_range[1], n_points)
    lats = np.random.uniform(lat_range[0], lat_range[1], n_points)
    depths = np.random.uniform(depth_range[0], depth_range[1], n_points)
    

    DIC_surf = 2000.0
    delta_DIC = 300.0
    z_scale = 800.0
    DIC = DIC_surf + delta_DIC * (1.0 - np.exp(-depths / z_scale))
    DIC += np.random.normal(0, 20, n_points)
    

    TA = DIC + 100.0 + np.random.normal(0, 15, n_points)
    

    T_surf = 20.0
    T = T_surf * np.exp(-depths / 200.0) + 2.0 + np.random.normal(0, 0.5, n_points)
    T = np.clip(T, -2.0, 35.0)
    

    S = 34.5 + 0.5 * np.sin(np.radians(lats)) + np.random.normal(0, 0.2, n_points)
    S = np.clip(S, 30.0, 38.0)
    
    return {
        'lons': lons,
        'lats': lats,
        'depths': depths,
        'DIC': DIC,
        'TA': TA,
        'T': T,
        'S': S,
    }
