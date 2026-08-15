"""Webcam capture with live sign-language prediction."""

from __future__ import annotations

import cv2

from src.predictor import ASLPredictor

WINDOW_NAME = "RSL Alphabet - Sign Language Recognition"


class WebcamPredictor:
    """Runs the predictor on live webcam frames."""

    def __init__(self, predictor: ASLPredictor, camera_index: int = 0) -> None:
        self.predictor = predictor
        self.capture = cv2.VideoCapture(camera_index)
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open webcam (camera index {camera_index}).")

    def run(self) -> None:
        """Capture frames, predict, and display results until Q is pressed."""
        try:
            while True:
                ok, frame = self.capture.read()
                if not ok:
                    print("Could not read a frame from the webcam.")
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                label, confidence = self.predictor.predict(rgb)
                print(f"Prediction: {label} (confidence: {confidence:.2%})", flush=True)

                self._draw_prediction(frame, label, confidence)
                cv2.imshow(WINDOW_NAME, frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            self.capture.release()
            cv2.destroyAllWindows()

    def _draw_prediction(self, frame, label: str, confidence: float) -> None:
        """Overlay the predicted letter and confidence on the frame."""
        text = f"{label} ({confidence:.2%})"
        cv2.putText(
            frame,
            text,
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
