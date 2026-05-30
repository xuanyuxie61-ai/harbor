
import numpy as np


def ensemble_kalman_filter(ensemble_forecast, observations, observation_models,
                           observation_noise_std=1.0, inflation_factor=1.02):
    n_ensemble, n_state = ensemble_forecast.shape
    n_obs = len(observations)

    x_mean = np.mean(ensemble_forecast, axis=0)
    X_pert = ensemble_forecast - x_mean
    X_pert *= inflation_factor


    H_ensemble = np.zeros((n_ensemble, n_obs), dtype=np.float64)
    for j, model in enumerate(observation_models):
        for i in range(n_ensemble):
            T_local = ensemble_forecast[i, model.location]
            H_ensemble[i, j] = model.forward_model(T_local)


    Y_pert = np.zeros((n_ensemble, n_obs), dtype=np.float64)
    for j in range(n_obs):
        Y_pert[:, j] = observations[j] + np.random.normal(0.0, observation_noise_std, n_ensemble)

    y_mean = np.mean(Y_pert, axis=0)
    H_mean = np.mean(H_ensemble, axis=0)


    PHt = X_pert.T @ (H_ensemble - H_mean) / (n_ensemble - 1.0)
    HPHt = (H_ensemble - H_mean).T @ (H_ensemble - H_mean) / (n_ensemble - 1.0)
    R = np.eye(n_obs) * (observation_noise_std**2)
    HPHT_plus_R = HPHt + R


    reg = 1e-6 * np.trace(HPHT_plus_R) / max(n_obs, 1)
    HPHT_plus_R += reg * np.eye(n_obs)

    try:
        K = PHt @ np.linalg.inv(HPHT_plus_R)
    except np.linalg.LinAlgError:
        K = PHt @ np.linalg.pinv(HPHT_plus_R)

    ensemble_analysis = np.zeros_like(ensemble_forecast)
    for i in range(n_ensemble):
        innovation = Y_pert[i] - H_ensemble[i]
        ensemble_analysis[i] = ensemble_forecast[i] + K @ innovation

    return np.clip(ensemble_analysis, 200.0, 350.0)


def compute_analysis_rmse(ensemble, truth):
    mean = np.mean(ensemble, axis=0)
    return float(np.sqrt(np.mean((mean - truth)**2)))


def compute_spread(ensemble):
    return np.std(ensemble, axis=0)


def assimilate_paleoclimate_data(ensemble, T_truth, proxy_models, t, vertices,
                                  observation_noise_std=0.8, inflation_factor=1.01):
    observations = np.array([
        model.sample_observation(T_truth[model.location])
        for model in proxy_models
    ])

    analysis = ensemble_kalman_filter(
        ensemble, observations, proxy_models,
        observation_noise_std=observation_noise_std,
        inflation_factor=inflation_factor
    )

    diagnostics = {
        'rmse_prior': compute_analysis_rmse(ensemble, T_truth),
        'rmse_post': compute_analysis_rmse(analysis, T_truth),
        'spread_prior': float(np.mean(compute_spread(ensemble))),
        'spread_post': float(np.mean(compute_spread(analysis))),
        'observations': observations
    }
    return analysis, diagnostics
