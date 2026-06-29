"""
BCG Stage 1 — Live object detection prototype (FAST / SMOOTH version).

Three things make this smooth instead of laggy:
  1. iGPU via OpenVINO  -> YOLO runs on the Intel Arc iGPU (much faster than CPU)
  2. Threaded reader     -> always grabs the LATEST frame, drops old ones (no delay)
  3. imgsz 640           -> smaller detection size = faster

It also SPEAKS what it sees and where (left / ahead / right) using offline TTS
(pyttsx3 / Windows SAPI5).  The voice plays through the laptop's default audio
output for now -- that's a stand-in for the glasses' bone-conduction transducer
(the real product routes this audio to the transducer via the ESP32 + amplifier).

Sources:
  - "webcam"  -> laptop's own camera (test without the glasses)
  - "glasses" -> the smart-glasses camera stream over WiFi (the real thing)

Controls (while the video window is focused):
  q -> quit     s -> save the current annotated frame to outputs/

Run:  python src/live_detection.py
"""

import os
import time
import threading
import subprocess
import numpy as np
import cv2
import requests
import pyttsx3
from ultralytics import YOLO

# ----------------------------------------------------------------------
# 1. CONFIG  -- the things you normally change
# ----------------------------------------------------------------------

SOURCE = "glasses"          # "glasses" or "webcam"

GLASSES_URL = "http://192.168.4.1:8080"   # same URL that shows the stream in a browser

# Where YOLO runs.  "intel:gpu" = Arc iGPU (fast).  "cpu" = fallback (slow).
DEVICE = "intel:gpu"

# Model to load.  After exporting (see export step), this folder exists and runs
# on the iGPU.  If it's missing, we fall back to the slow .pt model on CPU.
# Using SMALL: phone-realistic choice (a phone can't run medium well). Decent
# detection, ~29 FPS on the iGPU. The real accuracy gain for the phone will come
# from FINE-TUNING this small model on glasses footage, not from a bigger model.
OPENVINO_MODEL = "yolo26s_openvino_model"
PT_MODEL = "yolo26s.pt"

ROTATE = "cw"               # glasses camera is sideways: "none"/"cw"/"ccw"/"180"
CONF = 0.25                 # min confidence (glasses footage is grainy)
IMGSZ = 640                 # detection size — must match the exported model

# --- Voice (offline TTS) ---
SPEAK = True                # speak the most prominent detection out loud
SPEAK_MIN_GAP = 2.0         # seconds: never speak more often than this
SPEAK_REPEAT_GAP = 5.0      # seconds: re-announce the SAME thing only this often
VOICE_RATE = 175            # words per minute (default ~200; slower = clearer)
SPEAK_MIN_BOX_FRAC = 0.05   # ignore boxes smaller than this fraction of the frame
                            # (too far / grainy junk -> don't bother announcing)

# ----------------------------------------------------------------------
# 2. LOAD MODEL  (prefer the fast OpenVINO/iGPU model, fall back to CPU)
# ----------------------------------------------------------------------

if os.path.isdir(OPENVINO_MODEL):
    print(f"Loading OpenVINO model on {DEVICE} ...")
    model = YOLO(OPENVINO_MODEL)
    predict_device = DEVICE
else:
    print("OpenVINO model not found -> using slow CPU model. (Run the export step!)")
    model = YOLO(PT_MODEL)
    predict_device = "cpu"
print("Model ready.\n")


def fix_rotation(frame):
    if ROTATE == "cw":
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if ROTATE == "ccw":
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if ROTATE == "180":
        return cv2.rotate(frame, cv2.ROTATE_180)
    return frame


def current_wifi():
    """The WiFi network the laptop is on right now (or None)."""
    try:
        out = subprocess.check_output(
            ["netsh", "wlan", "show", "interfaces"], text=True, errors="ignore"
        )
        for line in out.splitlines():
            s = line.strip()
            if s.startswith("SSID") and "BSSID" not in s:
                return s.split(":", 1)[1].strip()
    except Exception:
        pass
    return None


# ----------------------------------------------------------------------
# 3. FRAME SOURCES  -- generators that yield raw BGR frames
# ----------------------------------------------------------------------

def webcam_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("ERROR: could not open the webcam (another app using it?)")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def glasses_frames():
    """Read the ESP32 MJPEG stream directly (browser-style) and yield frames."""
    wifi = current_wifi()
    if wifi and "ESP32" not in wifi:
        raise SystemExit(
            f"\n  STOP: laptop is on WiFi '{wifi}', NOT the glasses.\n"
            f"  Connect to ESP32-CAM (pw 12345678) and run again.\n"
            f"  (Also close any browser tab showing the stream — glasses allow 1 client.)\n"
        )

    print(f"Opening glasses stream: {GLASSES_URL}")
    first_connect = True
    while True:  # auto-reconnect if the stream drops
        try:
            r = requests.get(GLASSES_URL, stream=True, timeout=10)
        except Exception as e:
            if first_connect:
                raise SystemExit(
                    f"ERROR: could not connect to {GLASSES_URL}\n"
                    f"  - On the ESP32-CAM WiFi? Browser tab closed? ({e})"
                )
            print(f"Stream dropped, reconnecting... ({e})")
            continue
        first_connect = False

        buf = b""
        try:
            for chunk in r.iter_content(chunk_size=4096):
                if not chunk:
                    continue
                buf += chunk
                start = buf.find(b"\xff\xd8")
                end = buf.find(b"\xff\xd9", start + 2)
                while start != -1 and end != -1:
                    jpg = buf[start:end + 2]
                    buf = buf[end + 2:]
                    frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        yield frame
                    start = buf.find(b"\xff\xd8")
                    end = buf.find(b"\xff\xd9", start + 2)
                if start == -1 and len(buf) > 1_000_000:
                    buf = b""
        except Exception as e:
            print(f"Stream read error, reconnecting... ({e})")
            continue


# ----------------------------------------------------------------------
# 4. THREADED GRABBER  -- a background worker that keeps only the NEWEST frame
# ----------------------------------------------------------------------

class LatestFrame:
    """Runs the frame generator in a background thread and always exposes only
    the most recent frame, so YOLO never works on stale/buffered frames."""

    def __init__(self, generator):
        self._gen = generator
        self._frame = None
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        for frame in self._gen:
            if not self._running:
                break
            with self._lock:
                self._frame = frame          # overwrite -> old frames are dropped

    def read(self):
        with self._lock:
            return self._frame

    def stop(self):
        self._running = False


# ----------------------------------------------------------------------
# 5. VOICE  -- say the detection out loud WITHOUT freezing the video
# ----------------------------------------------------------------------

class Speaker:
    """Speaks text in a background thread so the video loop never blocks.
    A FRESH pyttsx3 engine is created per phrase -- this is deliberately the
    robust pattern: reusing one SAPI engine across many runAndWait() calls is a
    known pyttsx3 bug where it hangs after a few phrases.  While one phrase is
    playing, any new request is dropped (no backlog / lag)."""

    def __init__(self, rate=VOICE_RATE):
        self._rate = rate
        self._busy = False
        self._lock = threading.Lock()

    def say(self, text):
        with self._lock:
            if self._busy:
                return                       # still talking -> skip this one
            self._busy = True
        threading.Thread(target=self._run, args=(text,), daemon=True).start()

    def _run(self, text):
        try:
            import pythoncom                  # SAPI needs COM init in this thread
            pythoncom.CoInitialize()
        except Exception:
            pass
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", self._rate)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            print(f"TTS error: {e}")
        finally:
            try:
                import pythoncom
                pythoncom.CoUninitialize()
            except Exception:
                pass
            self._busy = False               # always reset, even if it errored


def announcement(result, frame_w, frame_h):
    """Pick the BIGGEST box (closest / most important) and turn it into a phrase
    like 'person on your left' / 'chair ahead'.  Boxes that are too small a part
    of the frame (far away / grainy noise) are ignored -- not worth announcing."""
    best, best_area = None, 0.0
    for b in result.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        area = (x2 - x1) * (y2 - y1)
        if area > best_area:
            best_area, best = area, (x1, x2, int(b.cls))
    if best is None:
        return None
    if best_area < SPEAK_MIN_BOX_FRAC * frame_w * frame_h:
        return None                          # too small / far -> stay quiet
    x1, x2, cls = best
    cx = (x1 + x2) / 2.0
    if cx < frame_w / 3:
        where = "on your left"
    elif cx > 2 * frame_w / 3:
        where = "on your right"
    else:
        where = "ahead"
    return f"{model.names[cls]} {where}"


# ----------------------------------------------------------------------
# 6. MAIN LOOP
# ----------------------------------------------------------------------

generator = glasses_frames() if SOURCE == "glasses" else webcam_frames()
grabber = LatestFrame(generator)

os.makedirs("outputs", exist_ok=True)
saved = 0
speaker = Speaker() if SPEAK else None
last_phrase, last_speak = None, 0.0
print("Running. Press 'q' to quit, 's' to save a frame.\n")

# wait for the first frame to arrive
while grabber.read() is None:
    time.sleep(0.05)

while True:
    frame = grabber.read()
    if frame is None:
        continue
    if SOURCE == "glasses":          # only the glasses cam is mounted sideways
        frame = fix_rotation(frame)

    result = model(frame, conf=CONF, imgsz=IMGSZ, device=predict_device, verbose=False)[0]

    labels = [model.names[int(b.cls)] for b in result.boxes]
    if labels:
        print("Detected:", ", ".join(labels))

    # --- speak the most prominent detection (throttled) ---
    if speaker is not None:
        phrase = announcement(result, frame.shape[1], frame.shape[0])
        now = time.time()
        if phrase and (now - last_speak) > SPEAK_MIN_GAP:
            is_new = phrase != last_phrase
            if is_new or (now - last_speak) > SPEAK_REPEAT_GAP:
                speaker.say(phrase)
                last_phrase, last_speak = phrase, now

    annotated = result.plot()
    cv2.imshow("BCG live detection  (q=quit, s=save)", annotated)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("s"):
        path = os.path.join("outputs", f"frame_{saved:03d}.jpg")
        cv2.imwrite(path, annotated)
        print(f"Saved {path}")
        saved += 1

grabber.stop()
cv2.destroyAllWindows()
print("Done.")
