"""Entry point for the rsl-alphabet webcam recognition system."""

import sys

from src.camera import WebcamPredictor
from src.predictor import ASLPredictor


def main() -> None:
    try:
        predictor = ASLPredictor()
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded model from {predictor.model_dir}")
    print("Press Q in the camera window to quit.")

    try:
        WebcamPredictor(predictor).run()
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
