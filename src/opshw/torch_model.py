import torch
from torch import nn


class DeepMLP(nn.Module):
    """Глубокая полносвязная нейросеть для демонстрации динамики градиентов и профилирования."""

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        output_dim: int = 1,
        num_layers: int = 4,
        activation: str = "sigmoid",
        use_batch_norm: bool = False,
    ) -> None:
        super().__init__()
        self.activation_name = activation.lower()
        self.use_batch_norm = use_batch_norm

        layers: list[nn.Module] = []
        current_dim = input_dim

        for i in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            if self.activation_name == "sigmoid":
                layers.append(nn.Sigmoid())
            elif self.activation_name == "relu":
                layers.append(nn.ReLU())
            elif self.activation_name == "tanh":
                layers.append(nn.Tanh())
            else:
                raise ValueError(
                    f"Unsupported activation: {activation}. Choose 'sigmoid', 'relu', or 'tanh'."
                )

            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
