
import numpy as np


class Observation:
    def __init__(self, obs_type, lon, lat, value, timestamp, error_var):
        self.type = obs_type
        self.lon = lon
        self.lat = lat
        self.value = value
        self.timestamp = timestamp
        self.error_var = max(error_var, 1e-6)


class ObservationAggregator:
    def __init__(self):
        self.observations = []
    
    def add_observation(self, obs):
        self.observations.append(obs)
    
    def group_by_type(self):
        groups = {}
        for obs in self.observations:
            if obs.type not in groups:
                groups[obs.type] = []
            groups[obs.type].append(obs)
        return groups
    
    def aggregate_group(self, obs_list):
        if len(obs_list) == 0:
            return None, None, 0
        
        weights = np.array([1.0 / obs.error_var for obs in obs_list])
        values = np.array([obs.value for obs in obs_list])
        
        total_weight = np.sum(weights)
        if total_weight < 1e-12:
            return np.mean(values), np.var(values), len(obs_list)
        
        mean_val = np.sum(weights * values) / total_weight
        combined_var = 1.0 / total_weight
        
        return mean_val, combined_var, len(obs_list)
    
    def summarize_all_groups(self):
        groups = self.group_by_type()
        summary = {}
        for obs_type, obs_list in groups.items():
            mean_val, var, n = self.aggregate_group(obs_list)
            summary[obs_type] = {
                'mean': mean_val,
                'variance': var,
                'count': n,
                'min': min(obs.value for obs in obs_list),
                'max': max(obs.value for obs in obs_list)
            }
        return summary


def gaspari_cohn_localization(distance, localization_length):
    z = distance / localization_length
    
    if isinstance(z, np.ndarray):
        rho = np.zeros_like(z)
        mask1 = z <= 1.0
        mask2 = (z > 1.0) & (z <= 2.0)
        
        rho[mask1] = (-0.25 * z[mask1]**5 + 0.5 * z[mask1]**4
                      + 0.625 * z[mask1]**3 - (5.0/3.0) * z[mask1]**2 + 1.0)
        
        rho[mask2] = ((1.0/12.0) * z[mask2]**5 - 0.5 * z[mask2]**4
                      + 0.625 * z[mask2]**3 + (5.0/3.0) * z[mask2]**2
                      - 5.0 * z[mask2] + 4.0 - 2.0/(3.0 * z[mask2]))
        return rho
    else:
        if z <= 1.0:
            return (-0.25 * z**5 + 0.5 * z**4 + 0.625 * z**3
                    - (5.0/3.0) * z**2 + 1.0)
        elif z <= 2.0:
            return ((1.0/12.0) * z**5 - 0.5 * z**4 + 0.625 * z**3
                    + (5.0/3.0) * z**2 - 5.0 * z + 4.0 - 2.0/(3.0 * z))
        else:
            return 0.0


def ensemble_kalman_filter_update(ensemble_states, observations, observation_operator,
                                   obs_errors, localization_length=500.0):
    n_ens, state_dim = ensemble_states.shape
    n_obs = len(observations)
    

    x_mean = np.mean(ensemble_states, axis=0)
    

    X_prime = ensemble_states - x_mean[np.newaxis, :]
    


    

    Y_f = np.dot(ensemble_states, observation_operator.T)
    y_mean = np.mean(Y_f, axis=0)
    Y_prime = Y_f - y_mean[np.newaxis, :]
    

    R = np.diag(obs_errors**2)
    





    
    P_HT = np.dot(X_prime.T, Y_prime) / (n_ens - 1.0)
    HPH_T = np.dot(Y_prime.T, Y_prime) / (n_ens - 1.0)
    


    for i in range(n_obs):
        for j in range(n_obs):
            if i != j:
                dist = 0.0
                loc = gaspari_cohn_localization(dist, localization_length)
                HPH_T[i, j] *= loc
    
    innov_cov = HPH_T + R
    
    try:
        inv_innov = np.linalg.inv(innov_cov)
    except np.linalg.LinAlgError:
        inv_innov = np.linalg.pinv(innov_cov)
    
    K = np.dot(P_HT, inv_innov)
    

    analysis_states = np.zeros_like(ensemble_states)
    for i in range(n_ens):
        innovation = observations - Y_f[i, :]
        analysis_states[i, :] = ensemble_states[i, :] + np.dot(K, innovation)
    
    return analysis_states


def generate_synthetic_observations(true_state, obs_types=None):
    if obs_types is None:
        obs_types = ['satellite_wind', 'dropsonde', 'radar', 'buoy']
    
    aggregator = ObservationAggregator()
    

    for obs_type in obs_types:
        n_obs = np.random.randint(3, 8)
        for _ in range(n_obs):

            if obs_type == 'satellite_wind':
                value = true_state[3] + np.random.normal(0, 10.0)
                error = 15.0
            elif obs_type == 'dropsonde':
                value = true_state[2] + np.random.normal(0, 3.0)
                error = 5.0
            elif obs_type == 'radar':
                value = true_state[2] + np.random.normal(0, 5.0)
                error = 8.0
            else:
                value = true_state[2] + np.random.normal(0, 8.0)
                error = 12.0
            
            obs = Observation(
                obs_type=obs_type,
                lon=true_state[0] + np.random.normal(0, 0.5),
                lat=true_state[1] + np.random.normal(0, 0.5),
                value=value,
                timestamp=0.0,
                error_var=error**2
            )
            aggregator.add_observation(obs)
    
    return aggregator
