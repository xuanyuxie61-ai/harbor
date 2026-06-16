"""
surrogate_model.py
==================
A physics-informed surrogate model for the reconnection rate, inspired
by discrete normalizing flow architectures.

Maps seed project:
  - 1069_andrew-cr_discrete_flow_models: transformer / autoregressive
    architecture for modeling probability distributions

Physical motivation:
  Full resistive MHD simulations of reconnection are expensive. A
  surrogate model can predict the reconnection rate R(eta, k, S, beta)
  much faster, enabling rapid parameter exploration.

Architecture:
  The model takes plasma parameters as input and outputs the normalized
  reconnection rate via a sequence of learned affine transformations
  (simplified normalizing flow):

    z_0 = normalize(input_params)
    z_{l+1} = W_l z_l + b_l + attention-weighted nonlinear correction
    R = sigmoid(z_L)

  The attention mechanism allows the model to focus on the most relevant
  parameter combinations (e.g., eta*S coupling for the reconnection rate).

This is a reduced-dimension model: it operates on 5 input parameters
(beta, eta_norm, d_i_norm, gamma, S) and outputs a scalar reconnection rate.

Training data comes from the physics-based stability analysis and
dispersion relations in stability_analysis.py.
"""

import numpy as np


class ReconnectionSurrogate:
    """
    Lightweight autoregressive surrogate for the reconnection rate.

    Uses a simplified attention-based architecture (inspired by the
    discrete flow model of 1069) with 2 layers and 16 hidden units.
    """

    def __init__(self, n_input=5, n_hidden=16, n_layers=2, n_heads=2):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.n_heads = n_heads

        # Initialize weights (Xavier initialization)
        rng = np.random.RandomState(42)
        self.weights = []
        self.biases = []

        d_in = n_input
        for l in range(n_layers):
            d_out = n_hidden
            scale = np.sqrt(2.0 / (d_in + d_out))
            W = rng.randn(d_in, d_out) * scale
            b = np.zeros(d_out)
            self.weights.append(W)
            self.biases.append(b)
            d_in = d_out

        # Output projection
        scale = np.sqrt(2.0 / (n_hidden + 1))
        self.W_out = rng.randn(n_hidden, 1) * scale
        self.b_out = np.zeros(1)

        # Attention parameters
        self.W_q = rng.randn(n_hidden, n_hidden) * 0.1
        self.W_k = rng.randn(n_hidden, n_hidden) * 0.1
        self.W_v = rng.randn(n_hidden, n_hidden) * 0.1

        # Normalization parameters (from training data)
        self.input_mean = None
        self.input_std = None

    def normalize_input(self, x):
        """Normalize input parameters to zero mean, unit variance."""
        if self.input_mean is None:
            return x
        std = np.atleast_1d(self.input_std)
        std_safe = np.where(np.abs(std) < 1e-8, 1e-8, std)
        return (x - self.input_mean) / std_safe

    def set_normalization(self, mean, std):
        """Set input normalization from training data statistics."""
        self.input_mean = np.array(mean)
        self.input_std = np.array(std)

    def attention_layer(self, h):
        """
        Simplified self-attention on the hidden state.
        For a single input vector, this reduces to a learned nonlinear
        modulation (query=key=value=h).
        """
        q = h @ self.W_q  # (n_hidden,)
        k = h @ self.W_k  # (n_hidden,)
        v = h @ self.W_v  # (n_hidden,)

        # Attention weights (scaled dot-product)
        scale = np.sqrt(self.n_hidden)
        attn = np.sum(q * k) / scale
        attn_weight = 1.0 / (1.0 + np.exp(-np.clip(attn, -10, 10)))

        # Apply attention to values
        return attn_weight * v

    def forward(self, x):
        """
        Forward pass: input_params -> reconnection_rate

        Input:
            x: array of shape (n_input,) or (batch, n_input)
        Output:
            rate: scalar or (batch,) array of reconnection rates
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
            squeeze = True
        else:
            squeeze = False

        # Normalize
        h = self.normalize_input(x)

        # Hidden layers with GELU activation
        for l in range(self.n_layers):
            h = h @ self.weights[l] + self.biases[l]
            # GELU activation (approximate)
            h = h * 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi)
                                           * (h + 0.044715 * h ** 3)))
            # Attention modulation
            if h.shape[0] == 1:
                attn_out = self.attention_layer(h[0])
                h[0] += 0.1 * attn_out

        # Output layer with sigmoid (rate must be in [0, 1])
        out = h @ self.W_out + self.b_out
        rate = 1.0 / (1.0 + np.exp(-out[0, 0]))  # sigmoid

        if squeeze:
            return float(rate)
        return rate

    def train_on_physics(self, n_samples=200):
        """
        Generate training data from the physics-based tearing mode
        growth rate and reconnection scaling laws, then fit the
        surrogate by least-squares.

        This is a simplified "training" that adjusts the weights
        to match known physics scaling.
        """
        from stability_analysis import tearing_growth_rate

        rng = np.random.RandomState(123)

        # Generate parameter samples
        params = []
        targets = []

        for _ in range(n_samples):
            eta = 10.0 ** rng.uniform(-5, -2)
            S = 1.0 / max(eta, 1e-15)
            beta = rng.uniform(0.01, 0.5)
            d_i = rng.uniform(0.001, 0.1)
            gamma = 5.0 / 3.0

            # Physics target: reconnection rate from tearing mode
            k_opt = S ** (-0.25)  # most unstable wavenumber
            gamma_max = tearing_growth_rate(k_opt, S)

            # Normalize to [0, 1] range
            rate_target = min(gamma_max * 10.0, 1.0)

            params.append([beta, eta, d_i, gamma, np.log10(S)])
            targets.append(rate_target)

        params = np.array(params)
        targets = np.array(targets)

        # Set normalization
        self.set_normalization(np.mean(params, axis=0),
                               np.std(params, axis=0) + 1e-8)

        # Simple gradient descent fitting
        lr = 0.01
        for epoch in range(100):
            # Forward pass for all samples
            total_loss = 0.0
            for i in range(n_samples):
                pred = self.forward(params[i])
                err = pred - targets[i]
                total_loss += err ** 2

                # Simplified gradient update (numerical gradient)
                eps = 1e-4
                for l in range(self.n_layers):
                    for ii in range(min(self.weights[l].shape[0], 3)):
                        for jj in range(min(self.weights[l].shape[1], 3)):
                            self.weights[l][ii, jj] += eps
                            pred_plus = self.forward(params[i])
                            self.weights[l][ii, jj] -= 2 * eps
                            pred_minus = self.forward(params[i])
                            self.weights[l][ii, jj] += eps

                            grad = (pred_plus - pred_minus) / (2 * eps) * err
                            self.weights[l][ii, jj] -= lr * grad

            if epoch % 20 == 0:
                pass  # silent training

    def predict_rate(self, plasma, k_pert=None):
        """
        Predict the reconnection rate for given plasma parameters.

        Input:
            plasma: SolarCoronaPlasma object
            k_pert: perturbation wavenumber (optional)
        """
        params = np.array([
            plasma.beta,
            plasma.eta_normalized,
            plasma.d_i_normalized,
            plasma.gamma_ad,
            np.log10(max(plasma.lundquist, 1.0)),
        ])
        return self.forward(params)


def validate_surrogate(n_test=20):
    """
    Validate the surrogate model against physics predictions.

    Returns:
        max_error: maximum absolute error over test set
        mean_error: mean absolute error
    """
    model = ReconnectionSurrogate()
    model.train_on_physics(n_samples=100)

    rng = np.random.RandomState(456)
    errors = []

    for _ in range(n_test):
        eta = 10.0 ** rng.uniform(-4, -2)
        S = 1.0 / eta
        beta = rng.uniform(0.01, 0.3)
        d_i = rng.uniform(0.005, 0.05)

        from stability_analysis import tearing_growth_rate
        k_opt = S ** (-0.25)
        gamma_physics = tearing_growth_rate(k_opt, S)
        rate_physics = min(gamma_physics * 10.0, 1.0)

        params = np.array([beta, eta, d_i, 5.0 / 3.0, np.log10(S)])
        rate_surrogate = model.forward(params)

        errors.append(abs(rate_surrogate - rate_physics))

    return {
        'max_error': max(errors),
        'mean_error': np.mean(errors),
        'n_test': n_test,
    }
