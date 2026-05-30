
import numpy as np


def _validate_positive_scalar(value, name):
    if not np.isscalar(value) or value <= 0:
        raise ValueError(f"{name} must be a positive scalar")


class PINNNetwork:

    def __init__(self, input_dim=2, hidden_dims=None, output_dim=1,
                 activation='gaussian_rbf', rbf_scale=1.0, seed=42):
        if hidden_dims is None:
            hidden_dims = [64, 64, 64, 64]
        if not isinstance(hidden_dims, (list, tuple)) or len(hidden_dims) == 0:
            raise ValueError("hidden_dims must be a non-empty list/tuple")
        if input_dim < 1 or output_dim < 1:
            raise ValueError("input_dim and output_dim must be positive integers")

        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.output_dim = output_dim
        self.activation_name = activation
        self.rbf_scale = float(rbf_scale)
        self.rng = np.random.default_rng(seed)


        self.weights = []
        self.biases = []
        dims = [input_dim] + self.hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            fan_in, fan_out = dims[i], dims[i + 1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            W = self.rng.normal(0.0, std, size=(fan_out, fan_in))
            b = np.zeros(fan_out)
            self.weights.append(W)
            self.biases.append(b)

        self.n_layers = len(self.weights)
        self._cache = {}

    def activation(self, z):
        if self.activation_name == 'gaussian_rbf':

            return np.exp(-0.5 * (z ** 2) / (self.rbf_scale ** 2 + 1e-12))
        elif self.activation_name == 'squircle':


            abs_z = np.abs(z)
            sqrt_abs = np.sqrt(abs_z + 1e-12)
            blend = sqrt_abs / (1.0 + sqrt_abs)
            return np.tanh(z) * blend
        elif self.activation_name == 'tanh':
            return np.tanh(z)
        else:
            raise ValueError(f"Unknown activation: {self.activation_name}")

    def activation_derivative(self, z):
        if self.activation_name == 'gaussian_rbf':

            phi = self.activation(z)
            return -z / (self.rbf_scale ** 2 + 1e-12) * phi
        elif self.activation_name == 'squircle':

            dz = 1e-6
            return (self.activation(z + dz) - self.activation(z - dz)) / (2.0 * dz)
        elif self.activation_name == 'tanh':

            t = np.tanh(z)
            return 1.0 - t ** 2
        else:
            raise ValueError(f"Unknown activation: {self.activation_name}")

    def forward(self, X, store_cache=False):
        if X.ndim != 2 or X.shape[1] != self.input_dim:
            raise ValueError(f"X must have shape (n_samples, {self.input_dim})")

        h = X.T
        activations = [h]
        z_values = []

        for i in range(self.n_layers - 1):
            z = self.weights[i] @ h + self.biases[i][:, np.newaxis]
            z_values.append(z)
            h = self.activation(z)
            activations.append(h)


        z = self.weights[-1] @ h + self.biases[-1][:, np.newaxis]
        z_values.append(z)
        h = z
        activations.append(h)

        if store_cache:
            self._cache = {'activations': activations, 'z_values': z_values, 'X': X}

        return h.T

    def predict(self, X):
        return self.forward(X, store_cache=False)

    def parameter_count(self):
        return sum(W.size for W in self.weights) + sum(b.size for b in self.biases)

    def get_params_flat(self):
        return np.concatenate([W.ravel() for W in self.weights] +
                              [b.ravel() for b in self.biases])

    def set_params_flat(self, params):
        idx = 0

        for i in range(self.n_layers):
            W_size = self.weights[i].size
            self.weights[i] = params[idx:idx + W_size].reshape(self.weights[i].shape)
            idx += W_size
        for i in range(self.n_layers):
            b_size = self.biases[i].size
            self.biases[i] = params[idx:idx + b_size].reshape(self.biases[i].shape)
            idx += b_size

    def finite_difference_derivatives(self, X, var_idx):
        h = 1e-5
        X_plus = X.copy()
        X_minus = X.copy()
        X_plus[:, var_idx] += h
        X_minus[:, var_idx] -= h
        u_plus = self.forward(X_plus, store_cache=False)
        u_minus = self.forward(X_minus, store_cache=False)
        return (u_plus - u_minus) / (2.0 * h)

    def second_derivative(self, X, var_idx):
        h = 1e-4
        X_plus = X.copy()
        X_minus = X.copy()
        X_plus[:, var_idx] += h
        X_minus[:, var_idx] -= h
        u_plus = self.forward(X_plus)
        u_center = self.forward(X)
        u_minus = self.forward(X_minus)
        return (u_plus - 2.0 * u_center + u_minus) / (h ** 2)

    def fourth_derivative(self, X, var_idx):




        raise NotImplementedError("Hole 2: fourth_derivative not implemented")
