
import numpy as np


def path_cost(n, distance, p):
    cost = 0.0
    i1 = n - 1
    for i2 in range(n):
        cost += distance[p[i1], p[i2]]
        i1 = i2
    return cost


def path_greedy(n, distance, start):
    p = np.zeros(n, dtype=int)
    p[0] = start
    d = distance.copy()
    d[:, start] = np.inf
    np.fill_diagonal(d, np.inf)

    from_city = start
    for j in range(1, n):
        to_city = int(np.argmin(d[from_city, :]))
        p[j] = to_city
        d[:, to_city] = np.inf
        from_city = to_city
    return p


def tsp_greedy_solver(coordinates):
    coordinates = np.asarray(coordinates, dtype=float)
    n = coordinates.shape[0]
    if n < 4:

        return np.arange(n), 0.0


    diff = coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]
    distance = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(distance, 0.0)

    best_cost = np.inf
    best_p = np.arange(n)

    for start in range(min(n, 20)):
        p = path_greedy(n, distance, start)
        cost = path_cost(n, distance, p)
        if cost < best_cost:
            best_cost = cost
            best_p = p

    return best_p, best_cost


def cvt_sensor_iterate(sensors, region_box, n_samples_per_sensor=500,
                       density_func=None):
    sensors = np.asarray(sensors, dtype=float)
    n, dim = sensors.shape
    sample_num = n_samples_per_sensor * n

    samples = np.zeros((sample_num, dim), dtype=float)
    for d in range(dim):
        samples[:, d] = region_box[d, 0] + np.random.rand(sample_num) * (
            region_box[d, 1] - region_box[d, 0])


    if density_func is not None:
        weights = np.array([density_func(samples[i, :]) for i in range(sample_num)])
        weights = np.clip(weights, 0.0, None)
        if np.sum(weights) > 0.0:

            probs = weights / np.sum(weights)
            indices = np.random.choice(sample_num, size=sample_num, p=probs)
            samples = samples[indices, :]


    diff = samples[:, np.newaxis, :] - sensors[np.newaxis, :, :]
    dists = np.sum(diff ** 2, axis=2)
    nearest = np.argmin(dists, axis=1)

    sensors_new = np.zeros_like(sensors)
    counts = np.zeros(n, dtype=int)

    for j in range(sample_num):
        idx = nearest[j]
        sensors_new[idx, :] += samples[j, :]
        counts[idx] += 1

    for j in range(n):
        if counts[j] > 0:
            sensors_new[j, :] /= counts[j]
        else:
            sensors_new[j, :] = sensors[j, :]

    avg_move = np.mean(np.sqrt(np.sum((sensors_new - sensors) ** 2, axis=1)))
    return sensors_new, avg_move


def optimize_sensor_array(n_sensors, region_box, it_max=100, tol=1e-5,
                          density_func=None, return_tsp=True):
    dim = region_box.shape[0]

    sensors = np.zeros((n_sensors, dim), dtype=float)
    for d in range(dim):
        sensors[:, d] = region_box[d, 0] + np.random.rand(n_sensors) * (
            region_box[d, 1] - region_box[d, 0])


    for it in range(it_max):
        sensors_new, avg_move = cvt_sensor_iterate(
            sensors, region_box, n_samples_per_sensor=500,
            density_func=density_func)
        sensors = sensors_new
        if avg_move < tol:
            break


    sample_num = 5000
    samples = np.zeros((sample_num, dim), dtype=float)
    for d in range(dim):
        samples[:, d] = region_box[d, 0] + np.random.rand(sample_num) * (
            region_box[d, 1] - region_box[d, 0])

    diff = samples[:, np.newaxis, :] - sensors[np.newaxis, :, :]
    dists = np.sum(diff ** 2, axis=2)
    min_dists = np.min(dists, axis=1)
    cvt_energy = float(np.mean(min_dists))

    result = {
        'sensors': sensors,
        'cvt_energy': cvt_energy,
        'iterations': it + 1
    }

    if return_tsp:
        tsp_path, tsp_cost = tsp_greedy_solver(sensors)
        result['tsp_path'] = tsp_path
        result['tsp_cost'] = tsp_cost

    return result


class SensorArray:

    def __init__(self, positions, sensitivity=1.0, noise_level=0.01):
        self.positions = np.asarray(positions, dtype=float)
        self.n_sensors = self.positions.shape[0]
        self.dim = self.positions.shape[1]
        self.sensitivity = float(sensitivity)
        self.noise_level = float(noise_level)

    def measure(self, true_field_func):
        measurements = np.zeros(self.n_sensors, dtype=float)
        for i in range(self.n_sensors):
            val = true_field_func(self.positions[i, :])
            if not np.isfinite(val):
                val = 0.0
            noise = np.random.randn() * self.noise_level * max(abs(val), 1.0)
            measurements[i] = self.sensitivity * val + noise
        return measurements

    def reconstruction_mse(self, true_field_func, reconstructed_field_func):
        mse = 0.0
        for i in range(self.n_sensors):
            true_val = true_field_func(self.positions[i, :])
            recon_val = reconstructed_field_func(self.positions[i, :])
            if np.isfinite(true_val) and np.isfinite(recon_val):
                mse += (true_val - recon_val) ** 2
        return mse / self.n_sensors
