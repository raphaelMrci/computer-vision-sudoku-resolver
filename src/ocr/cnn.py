from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from torch import nn


TARGET_SIZE = 32
DEFAULT_CNN_CHECKPOINT = Path("models/ocr_cnn.pt")
MIN_CONFIDENCE = 0.55


class OCRCNN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


@dataclass
class _LoadedModel:
    model: OCRCNN
    device: torch.device


_MODEL_CACHE: Optional[_LoadedModel] = None


def _preprocess_cell(cell_image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(cell_image, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    pad_y = max(1, int(h * 0.2))
    pad_x = max(1, int(w * 0.2))
    roi = gray[pad_y : h - pad_y, pad_x : w - pad_x]
    if roi.size == 0:
        roi = gray

    roi = cv2.GaussianBlur(roi, (3, 3), 0)
    roi = cv2.resize(roi, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
    roi = roi.astype(np.float32) / 255.0
    return roi


def _load_model(checkpoint_path: Path = DEFAULT_CNN_CHECKPOINT) -> Optional[_LoadedModel]:
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    if not checkpoint_path.exists():
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OCRCNN(num_classes=10).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.eval()
    _MODEL_CACHE = _LoadedModel(model=model, device=device)
    return _MODEL_CACHE


@torch.no_grad()
def read_digit_with_confidence_cnn(
    cell_image: np.ndarray,
    checkpoint_path: Path = DEFAULT_CNN_CHECKPOINT,
) -> Tuple[int, float]:
    loaded = _load_model(checkpoint_path=checkpoint_path)
    if loaded is None:
        return 0, 0.0

    x = _preprocess_cell(cell_image)
    tensor = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(loaded.device)
    logits = loaded.model(tensor)
    probs = torch.softmax(logits, dim=1)[0]
    conf, idx = torch.max(probs, dim=0)
    label = int(idx.item())
    confidence = float(conf.item())

    if label == 0:
        return 0, confidence
    if confidence < MIN_CONFIDENCE:
        return 0, confidence
    return label, confidence
