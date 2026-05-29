from __future__ import annotations

import numpy as np


class MCDropoutRegressor:
    """Torch MC-dropout regressor for uncertainty-aware meta-pricing."""

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: tuple[int, ...] = (128, 64),
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        seed: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.seed = seed
        self.model = None

    def _build(self):
        import torch
        from torch import nn

        torch.manual_seed(self.seed)
        layers: list[nn.Module] = []
        previous = self.input_dim
        for size in self.hidden_sizes:
            layers.extend([nn.Linear(previous, size), nn.ReLU(), nn.Dropout(self.dropout)])
            previous = size
        layers.append(nn.Linear(previous, 1))
        return nn.Sequential(*layers)

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int = 100, batch_size: int = 128) -> None:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        self.model = self._build()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        loss_fn = torch.nn.MSELoss()
        dataset = TensorDataset(
            torch.tensor(x, dtype=torch.float32), torch.tensor(y[:, None], dtype=torch.float32)
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.model.train()
        for _ in range(epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = loss_fn(self.model(xb), yb)
                loss.backward()
                optimizer.step()

    def predict_distribution(self, x: np.ndarray, n_samples: int = 100) -> np.ndarray:
        import torch

        if self.model is None:
            raise RuntimeError("model must be fitted before prediction")
        self.model.train()
        xt = torch.tensor(x, dtype=torch.float32)
        preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                preds.append(self.model(xt).cpu().numpy().ravel())
        return np.vstack(preds)

    def predict_interval(
        self, x: np.ndarray, n_samples: int = 100, alpha: float = 0.05
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        samples = self.predict_distribution(x, n_samples)
        mean = np.mean(samples, axis=0)
        lower = np.quantile(samples, alpha / 2, axis=0)
        upper = np.quantile(samples, 1 - alpha / 2, axis=0)
        return mean, lower, upper

