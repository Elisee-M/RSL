"""Sign-language alphabet prediction using a local Hugging Face model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "asl-efficientnet-b0"

REQUIRED_FILES = ("config.json", "model.safetensors")


class ASLPredictor:
    """Runs inference with a local image-classification model."""

    def __init__(self, model_dir: Path | str = MODEL_DIR) -> None:
        self.model_dir = Path(model_dir)
        self._validate_model_dir()
        self.processor = AutoImageProcessor.from_pretrained(self.model_dir)
        self.model = AutoModelForImageClassification.from_pretrained(self.model_dir)
        self.model.eval()

    def _validate_model_dir(self) -> None:
        """Raise a clear error if the required model files are missing."""
        missing = [
            name for name in REQUIRED_FILES if not (self.model_dir / name).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing model files in {self.model_dir}: {', '.join(missing)}. "
                "Copy the Hugging Face model files (config.json, model.safetensors) "
                "into models/asl-efficientnet-b0/ and try again."
            )

    @torch.no_grad()
    def predict(self, frame: np.ndarray) -> tuple[str, float]:
        """Predict the letter for a single RGB frame.

        Args:
            frame: An RGB image as a NumPy array (H, W, 3).

        Returns:
            A tuple of (predicted label, confidence score in [0, 1]).
        """
        inputs = self.processor(images=frame, return_tensors="pt")
        logits = self.model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)
        best_index = int(torch.argmax(probabilities, dim=-1).item())
        confidence = float(probabilities[0, best_index].item())
        label = self.model.config.id2label[best_index]
        return label, confidence
