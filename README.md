# rsl-alphabet

Real-time sign-language alphabet recognition from a webcam using a pretrained
Hugging Face EfficientNet-B0 image-classification model.


> **Important:** The current model recognizes the **American Sign Language (ASL)
> alphabet**, not Rwanda Sign Language (RSL). It is used only as an initial
> baseline. RSL-specific fine-tuning and training are planned for a later stage.

## Current scope (V1)

- Capture frames from the default webcam with OpenCV.
- Classify each frame with a local pretrained EfficientNet-B0 model
  (`AutoImageProcessor` + `AutoModelForImageClassification`).
- Display the predicted letter and confidence on the camera window.
- Print the prediction to the terminal.
- Press `Q` to quit.

Out of scope for V1 (planned later): web app, database, API, authentication,
frontend, training/fine-tuning, MediaPipe, and object detection.

## Project structure

```
rsl-alphabet/
├── models/
│   └── asl-efficientnet-b0/      # Hugging Face model files (not committed)
│       └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── camera.py                 # webcam capture + live display
│   └── predictor.py              # model loading + inference
├── .gitignore
├── requirements.txt
├── README.md
└── main.py                       # entry point
```

## Installation

1. Create and activate a virtual environment:

   ```
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   source .venv/bin/activate     # Linux/macOS
   ```

2. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

## Placing the Hugging Face model

Download an ASL alphabet EfficientNet-B0 model (for example one of the
`ASL_Alphabet_EfficientNet_B0` variants on the Hugging Face Hub) and copy its
files into:

```
models/asl-efficientnet-b0/
```

Expected files:

- `config.json`
- `model.safetensors`
- `preprocessor_config.json`

The model weights are ignored by Git and will never be committed.

## Running

```
python main.py
```

A camera window will open showing your live feed with the predicted letter and
confidence overlaid. The prediction is also printed to the terminal.

To quit: press `Q` in the camera window (or close the window). The camera is
released automatically.

## Roadmap

- V1 (this version): ASL alphabet baseline with a pretrained model.
- Later: collect Rwanda Sign Language data, fine-tune the model on RSL, and add
  more advanced features such as hand tracking (MediaPipe) and detection.
