# BCG Smart Glasses — Phone-side AI

Object detection + spoken guidance for the BCG (Bone-Conduction) Smart Glasses.

The glasses' camera streams over WiFi to a phone/laptop, where an AI model detects
objects in real time, works out **where** each object is (left / ahead / right), and
**speaks** it out loud — so a visually-impaired user hears, e.g., *"person on your left"*.
The voice is meant to play through the glasses' bone-conduction transducer (for now it
plays on the laptop speaker as a stand-in).

## How it works

```
Glasses camera (ESP32-S3 + OV2640)
        │  MJPEG stream over WiFi
        ▼
Phone / laptop  ──►  YOLO26-small (OpenVINO)  ──►  detect objects
        │                                          + pick closest one
        │                                          + decide direction
        ▼
Offline TTS (voice cue)  ──►  "person on your left"
        │  (in the final product, routed back to the glasses transducer)
        ▼
   User hears it
```

## Files

| File | What it does |
|------|--------------|
| `src/live_detection.py` | **Main program** — live detection on the glasses/webcam feed, direction, and offline voice output |
| `src/diagnose_glasses.py` | Step-by-step diagnostic to check the WiFi connection to the glasses camera |
| `notebooks/stage1_detection.ipynb` | Stage-1 detection experiments |
| `notebooks/stage1_detection_output.ipynb` | Same notebook with the result images saved in it |

## Run it

```bash
# 1. set up a virtual environment and install deps
python -m venv bcg_ai_env
bcg_ai_env\Scripts\activate          # Windows
pip install ultralytics opencv-python requests pyttsx3 pypiwin32 openvino

# 2. run live detection
python src/live_detection.py
```

Inside `src/live_detection.py`, set the source at the top:

- `SOURCE = "webcam"` — test on the laptop's own camera (no glasses needed)
- `SOURCE = "glasses"` — connect the laptop to the glasses' WiFi (`ESP32-CAM`,
  password `12345678`) first, then run

Controls in the video window: **`q`** = quit, **`s`** = save the current frame.

## Performance

YOLO26-small is exported to **OpenVINO** and runs on the laptop's Intel Arc iGPU
(`device="intel:gpu"`) for smooth real-time speed (~29 FPS). A background thread always
grabs the newest frame so there's no buffer lag, and the voice runs in its own thread so
the video never freezes.

`small` was chosen over a bigger model because the final product runs on a **phone**, not
a powerful laptop — the real accuracy gains will come from **fine-tuning** this small model
on real glasses footage, not from shipping a heavier model.

## Status

- ✅ Live detection working on the glasses feed
- ✅ Direction (left / ahead / right) from the object's position
- ✅ Offline voice output (no internet needed)
- ⬜ Fine-tune the model on real glasses footage
- ⬜ Increase camera resolution (trades off FPS)
- ⬜ Add distance ("how far") for closeness-based alerts
- ⬜ Firmware integration — run camera + audio together on the glasses (hardware-side work)

## Notes

The large generated files are **not** in this repo (by design): the virtual environment,
model weights (`.pt`), the exported `*_openvino_model/` folders, and output images. Anyone
cloning the repo creates these locally via the steps above.
