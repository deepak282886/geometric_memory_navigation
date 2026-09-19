"""MLP classifier head over ``[position ; navigation_vector]`` -> 4 options.

REPLACED BY GEOMETRIC ENDING SCORING - see ``score_example`` in
``src/eval/evaluate.py``.

TASK 7 trained this head on ``[ctx position ; nav vector]`` and it plateaued at
~26% val accuracy: the correct HellaSwag ending is a property of the *endings*,
which never reached the head, so the task was unlearnable from those features
(measured: every ctx-only predictor sits at 0.22-0.28 ~ chance).

The head is kept for reference (and for the TASK 9 ablations that retrain it),
but inference now scores each candidate ending geometrically instead.
"""

from __future__ import annotations

import torch
import torch.nn as nn

POSITION_DIM = 256
NAVIGATION_DIM = 256
INPUT_DIM = POSITION_DIM + NAVIGATION_DIM  # 512
HIDDEN_DIM_1 = 256
HIDDEN_DIM_2 = 128
NUM_CLASSES = 4
DROPOUT = 0.1


class ClassifierHead(nn.Module):
    """512 -> 256 -> 128 -> 4, with ReLU + dropout on the hidden layers."""

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dim_1: int = HIDDEN_DIM_1,
        hidden_dim_2: int = HIDDEN_DIM_2,
        num_classes: int = NUM_CLASSES,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.num_classes = num_classes

        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape ``(batch, num_classes)`` - no softmax."""
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        return self.fc3(x)

    def count_parameters(self) -> int:
        """Number of trainable parameters in this head."""
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def __repr__(self) -> str:
        return (
            f"ClassifierHead({self.input_dim} -> {self.hidden_dim_1} -> "
            f"{self.hidden_dim_2} -> {self.num_classes}, dropout={self.dropout.p})"
        )