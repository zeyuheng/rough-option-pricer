from __future__ import annotations

import numpy as np


class MCDropoutRegressor:
    """Lightweight NumPy MC-dropout neural regressor.

    The model is intentionally small and dependency-light. It trains with dropout
    masks and keeps dropout active at prediction time, so repeated forward passes
    form an approximate predictive distribution.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: tuple[int, ...] = (64, 32),
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        seed: int = 42,
    ) -> None:
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        layer_sizes = (input_dim, *hidden_sizes, 1)
        self.weights = [
            self.rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out))
            for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:], strict=True)
        ]
        self.biases = [np.zeros((1, fan_out)) for fan_out in layer_sizes[1:]]

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0.0)

    @staticmethod
    def _relu_grad(x: np.ndarray) -> np.ndarray:
        return (x > 0.0).astype(float)

    def _forward(
        self,
        x: np.ndarray,
        training: bool,
    ) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        activations = [x]
        preactivations: list[np.ndarray] = []
        dropout_masks: list[np.ndarray] = []

        current = x
        keep_prob = 1.0 - self.dropout
        for weight, bias in zip(self.weights[:-1], self.biases[:-1], strict=True):
            z = current @ weight + bias
            a = self._relu(z)
            if training and self.dropout > 0:
                mask = (self.rng.random(a.shape) < keep_prob).astype(float) / keep_prob
                a = a * mask
            else:
                mask = np.ones_like(a)
            preactivations.append(z)
            dropout_masks.append(mask)
            activations.append(a)
            current = a

        output = current @ self.weights[-1] + self.biases[-1]
        activations.append(output)
        return output, activations, preactivations, dropout_masks

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int = 300,
        batch_size: int = 64,
    ) -> None:
        y = y.reshape(-1, 1)
        n_samples = len(x)
        for _ in range(epochs):
            order = self.rng.permutation(n_samples)
            for start in range(0, n_samples, batch_size):
                batch_idx = order[start : start + batch_size]
                xb = x[batch_idx]
                yb = y[batch_idx]
                pred, activations, preactivations, masks = self._forward(xb, training=True)
                grad = 2.0 * (pred - yb) / len(xb)

                grad_weights: list[np.ndarray] = []
                grad_biases: list[np.ndarray] = []
                grad_w = activations[-2].T @ grad
                grad_b = grad.sum(axis=0, keepdims=True)
                grad_hidden = grad @ self.weights[-1].T
                grad_weights.append(grad_w)
                grad_biases.append(grad_b)

                for layer in range(len(self.weights) - 2, -1, -1):
                    grad_hidden *= masks[layer]
                    grad_hidden *= self._relu_grad(preactivations[layer])
                    grad_w = activations[layer].T @ grad_hidden
                    grad_b = grad_hidden.sum(axis=0, keepdims=True)
                    grad_weights.append(grad_w)
                    grad_biases.append(grad_b)
                    if layer > 0:
                        grad_hidden = grad_hidden @ self.weights[layer].T

                grad_weights.reverse()
                grad_biases.reverse()
                for i in range(len(self.weights)):
                    self.weights[i] -= self.learning_rate * grad_weights[i]
                    self.biases[i] -= self.learning_rate * grad_biases[i]

    def predict_distribution(self, x: np.ndarray, n_samples: int = 100) -> np.ndarray:
        preds = []
        for _ in range(n_samples):
            pred, *_ = self._forward(x, training=True)
            preds.append(pred.ravel())
        return np.vstack(preds)

    def predict(self, x: np.ndarray, n_samples: int = 100) -> np.ndarray:
        return np.mean(self.predict_distribution(x, n_samples=n_samples), axis=0)

    def predict_interval(
        self, x: np.ndarray, n_samples: int = 100, alpha: float = 0.05
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        samples = self.predict_distribution(x, n_samples)
        mean = np.mean(samples, axis=0)
        lower = np.quantile(samples, alpha / 2, axis=0)
        upper = np.quantile(samples, 1 - alpha / 2, axis=0)
        return mean, lower, upper

