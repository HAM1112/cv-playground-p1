"""CellNet: a tiny CNN that labels a 64x64 grayscale cell crop as empty (0) or circle (1)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .board import CROP_SIZE

LABELS = ("empty", "circle")
DEFAULT_MODEL_PATH = Path(
    os.environ.get(
        "GRIDCHECK_MODEL",
        Path(__file__).resolve().parents[2] / "models" / "cellnet.pt",
    )
)


def _block(cin: int, cout: int) -> nn.Sequential:
    # GroupNorm rather than BatchNorm: no running statistics, so the model behaves
    # the same in eval mode even after very short training runs or small fine-tunes.
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.GroupNorm(4, cout),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class CellNet(nn.Module):
    def __init__(self, n_classes: int = len(LABELS)) -> None:
        super().__init__()
        self.features = nn.Sequential(_block(1, 16), _block(16, 32), _block(32, 64))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, 1, H, W) in [0, 1]
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


def preprocess(crops: np.ndarray) -> torch.Tensor:
    """uint8 (N, H, W) -> float tensor (N, 1, H, W) in [0, 1]."""
    if crops.ndim == 2:
        crops = crops[None]
    x = torch.from_numpy(np.ascontiguousarray(crops)).float().div_(255.0)
    return x.unsqueeze(1)


def pick_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_model(model: CellNet, path: str | os.PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "crop_size": CROP_SIZE, "labels": list(LABELS)},
        path,
    )


def load_model(path: str | os.PathLike | None = None, device: str | torch.device | None = None) -> CellNet:
    path = Path(path) if path else DEFAULT_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model at {path}. Run `uv run gridcheck train` to create it "
            "(or `git checkout -- models/cellnet.pt` to restore the committed one)."
        )
    dev = pick_device(str(device)) if isinstance(device, str) else (device or pick_device())
    ckpt = torch.load(path, map_location=dev)
    model = CellNet(n_classes=len(ckpt.get("labels", LABELS)))
    model.load_state_dict(ckpt["state_dict"])
    model.to(dev).eval()
    return model
