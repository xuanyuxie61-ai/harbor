"""
pinn_network.py
===============
Physics-Informed Neural Network architecture with custom activation functions.

The network approximates the solution u_\theta(t, x) of the KS PDE.
Architecture: fully-connected feedforward network with residual connections
and RBF-inspired Gaussian activations.

For a scalar output u(t,x) with inputs z = [t, x], the L-layer network is:

    h^{(0)} = z
    h^{(l)} = \sigma( W^{(l)} h^{(l-1)} + b^{(l)} ),   l = 1, ..., L-1
    u_\theta = W^{(L)} h^{(L-1)} + b^{(L)}

with\sigma being a composite activation combining:
  - Gaussian RBF kernel: \phi_g(r) = exp(-0.5 r^2 / r_0^2)
  - Squircle-inspired nonlinearities for periodic structure capture
  - tanh for smooth gradient flow

The total parameter vector \theta concatenates all (W^{(l)}, b^{(l)}).
"""

import numpy as np


def _validate_positive_scalar(value, name):
    if not np.isscalar(value) or value <= 0:
        raise ValueError(f"{name} must be a positive scalar")


class PINNNetwork:
    """
    Fully-connected neural network for PINN regression.

    Parameters
    ----------
    input_dim : int
        Dimension of input features (e.g., 2 for (t, x)).
    hidden_dims : list of int
        Number of neurons in each hidden layer.
    output_dim : int
        Dimension of output (e.g., 1 for scalar u).
    activation : str
        'gaussian_rbf', 'squircle', or 'tanh'.
    rbf_scale : float
        Scale parameter r_0 for Gaussian RBF activations.
    seed : int
        Random seed for weight initialization.
    """

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

        # Xavier/Glorot initialization: W ~ N(0, sqrt(2 / (fan_in + fan_out)))
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
        """
        Element-wise activation function.

        For 'gaussian_rbf':
            \phi(z) = exp(-0.5 * z^2 / r_0^2)

        For 'squircle':
            Uses a periodic generalization of sine/cosine via squircle ODE
            parametrization:
            \phi_s(z) = sign(z) * |z|^{s-1}  with s=4 (super-elliptic)
            Here we blend it with tanh for stability:
            \phi_s(z) = tanh(z) * |z|^{0.5} / (1 + |z|^{0.5})

        For 'tanh': standard hyperbolic tangent.
        """
        if self.activation_name == 'gaussian_rbf':
            # Gaussian RBF kernel: exp(-0.5 * z^2 / r0^2)
            return np.exp(-0.5 * (z ** 2) / (self.rbf_scale ** 2 + 1e-12))
        elif self.activation_name == 'squircle':
            # Squircle-inspired: super-elliptic smooth blending
            # \phi(z) = tanh(z) * sqrt(|z|) / (1 + sqrt(|z|))
            abs_z = np.abs(z)
            sqrt_abs = np.sqrt(abs_z + 1e-12)
            blend = sqrt_abs / (1.0 + sqrt_abs)
            return np.tanh(z) * blend
        elif self.activation_name == 'tanh':
            return np.tanh(z)
        else:
            raise ValueError(f"Unknown activation: {self.activation_name}")

    def activation_derivative(self, z):
        """
        Derivative of activation function w.r.t. its argument.
        """
        if self.activation_name == 'gaussian_rbf':
            # d/dz exp(-0.5 z^2 / r0^2) = -z/r0^2 * exp(-0.5 z^2 / r0^2)
            phi = self.activation(z)
            return -z / (self.rbf_scale ** 2 + 1e-12) * phi
        elif self.activation_name == 'squircle':
            # Approximate derivative via finite differences for robustness
            dz = 1e-6
            return (self.activation(z + dz) - self.activation(z - dz)) / (2.0 * dz)
        elif self.activation_name == 'tanh':
            # d/dz tanh(z) = 1 - tanh^2(z)
            t = np.tanh(z)
            return 1.0 - t ** 2
        else:
            raise ValueError(f"Unknown activation: {self.activation_name}")

    def forward(self, X, store_cache=False):
        """
        Forward pass.

        Parameters
        ----------
        X : ndarray, shape (n_samples, input_dim)
            Input matrix.
        store_cache : bool
            If True, store intermediate activations for backprop.

        Returns
        -------
        Y : ndarray, shape (n_samples, output_dim)
            Network output.
        """
        if X.ndim != 2 or X.shape[1] != self.input_dim:
            raise ValueError(f"X must have shape (n_samples, {self.input_dim})")

        h = X.T  # shape (input_dim, n_samples)
        activations = [h]
        z_values = []

        for i in range(self.n_layers - 1):
            z = self.weights[i] @ h + self.biases[i][:, np.newaxis]
            z_values.append(z)
            h = self.activation(z)
            activations.append(h)

        # Final linear layer
        z = self.weights[-1] @ h + self.biases[-1][:, np.newaxis]
        z_values.append(z)
        h = z  # linear output
        activations.append(h)

        if store_cache:
            self._cache = {'activations': activations, 'z_values': z_values, 'X': X}

        return h.T  # shape (n_samples, output_dim)

    def predict(self, X):
        """Alias for forward without cache."""
        return self.forward(X, store_cache=False)

    def parameter_count(self):
        """Return total number of trainable parameters."""
        return sum(W.size for W in self.weights) + sum(b.size for b in self.biases)

    def get_params_flat(self):
        """Flatten all parameters into a 1D vector."""
        return np.concatenate([W.ravel() for W in self.weights] +
                              [b.ravel() for b in self.biases])

    def set_params_flat(self, params):
        """Restore parameters from a flat vector."""
        idx = 0
        # Match get_params_flat: all weights first, then all biases
        for i in range(self.n_layers):
            W_size = self.weights[i].size
            self.weights[i] = params[idx:idx + W_size].reshape(self.weights[i].shape)
            idx += W_size
        for i in range(self.n_layers):
            b_size = self.biases[i].size
            self.biases[i] = params[idx:idx + b_size].reshape(self.biases[i].shape)
            idx += b_size

    def finite_difference_derivatives(self, X, var_idx):
        """
        Compute partial derivative of network output w.r.t. input variable
        var_idx using central finite differences.

        For a neural network u_\theta(t, x), the derivative is:
            \partial u / \partial z_{var_idx} = (u(z + h*e) - u(z - h*e)) / (2h)

        Parameters
        ----------
        X : ndarray, shape (n_samples, input_dim)
        var_idx : int
            Index of input variable to differentiate w.r.t.

        Returns
        -------
        du_dvar : ndarray, shape (n_samples, output_dim)
        """
        h = 1e-5
        X_plus = X.copy()
        X_minus = X.copy()
        X_plus[:, var_idx] += h
        X_minus[:, var_idx] -= h
        u_plus = self.forward(X_plus, store_cache=False)
        u_minus = self.forward(X_minus, store_cache=False)
        return (u_plus - u_minus) / (2.0 * h)

    def second_derivative(self, X, var_idx):
        """
        Compute second partial derivative using finite differences:
            \partial^2 u / \partial z_{var_idx}^2 =
                (u(z+h) - 2u(z) + u(z-h)) / h^2
        """
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
        """
        Compute fourth partial derivative using 5-point stencil:
            f^{(4)} \approx (f_{-2} - 4f_{-1} + 6f_0 - 4f_{+1} + f_{+2}) / h^4
        """
        # TODO (Hole 2): Implement the 5-point central difference stencil
        # for the fourth partial derivative of the network output.
        # HINT: Use step size h=2e-3, coefficients [1, -4, 6, -4, 1],
        # and shifts [-2, -1, 0, 1, 2] applied to input coordinate var_idx.
        raise NotImplementedError("Hole 2: fourth_derivative not implemented")
