# Plate Imaging System

A Raspberry Pi-based bacterial colony imaging and quantification platform built for the UConn MLS laboratory. Captures backlit agar plate images, counts colonies, extracts per-colony features, and flags anomalies using a two-layer statistical and machine learning pipeline.

---

## Hardware

| Component | Details |
|---|---|
| Computer | Raspberry Pi (CM4/CM5) |
| Camera | Raspberry Pi Camera Module 3 Wide NoIR (IMX708) |
| Backlight | LED panel with diffuser (enclosure) |
| Interface | Touchscreen or remote via VNC / SSH -X |

---

## Features

- **Live preview** with real-time brightness, contrast, and saturation controls
- **Colony detection** using background subtraction and watershed segmentation
- **Distance-invariant measurements** — all areas reported in mm² regardless of camera height
- **Haemolysis detection** — estimates halo brightness around each colony under backlight
- **Statistical anomaly detection** — flags unusual colonies by size, shape, and morphology from the first run
- **ML anomaly detection** — Random Forest + Isolation Forest classifier, improves as labelled data accumulates
- **Passive data logging** — every run automatically logs per-colony features to CSV for ML training
- **Annotated output images** — green = normal colony, red = anomaly flagged

---

## Project Structure

```
imager.py          — Main GUI application
quantify.py        — Colony detection, segmentation, feature extraction
anomaly.py         — Two-layer anomaly detection (statistical + ML)
data_logger.py     — Passive CSV feature logger for ML training
launch.sh          — Launcher script (handles venv activation)
imager.desktop     — Linux desktop shortcut
requirements.txt   — Python dependencies
captures/          — Saved plate images
results/           — Annotated output images
data/              — Colony feature CSV (training data)
models/            — Saved ML model (generated after training)
```

---

## Setup

**1. Clone and install dependencies**
```bash
git clone https://github.com/yourusername/imager.git ~/imager
cd ~/imager
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Enable the camera (OV9281)**

Add to `/boot/firmware/config.txt` under `[all]`:
```
dtoverlay=ov9281,cam0
```
Then reboot.

**3. Run**
```bash
source venv/bin/activate
python3 imager.py
```

**4. Optional — desktop shortcut**
```bash
chmod +x launch.sh
cp imager.desktop ~/.local/share/applications/
cp imager.desktop ~/Desktop/
```
Update the `Exec=` path in `imager.desktop` to match your install directory first.

---

## Running Without a Camera

The system detects missing or faulty hardware at startup and displays a **No Camera Connected** message. Capture is disabled but all other features — gallery, quantification on existing images, anomaly detection — remain available.

---

## Workflow

1. **Capture** — click Capture Image or select an existing image from the gallery
2. **Quantify** — click Quantify Selected Image to run colony detection
3. **Review** — annotated image appears in the preview; results panel shows count, area stats, anomaly count, haemolysis candidates, and plate coverage
4. Features are automatically saved to `data/colony_features.csv` on every run

---

## Training the ML Anomaly Model

The ML layer activates once enough labelled data exists. Label the `is_anomaly` column in `data/colony_features.csv` (1 = anomaly, 0 = normal), then:

```bash
python anomaly.py data/colony_features.csv
```

The model saves to `models/anomaly.pkl` and loads automatically on next launch. Check dataset readiness at any time:

```bash
python data_logger.py
```

---

## Dependencies

```
numpy
opencv-python-headless
Pillow
scipy
scikit-learn
picamera2       # Raspberry Pi only
```

---

## Notes

- Designed for 90–100mm standard petri dishes
- Backlit imaging only — plates must be translucent (agar, not opaque blood agar base)
- Optimal performance requires diffused, even backlight — a strong central hotspot reduces detection accuracy in the plate centre
- All colony areas are expressed in mm² and are distance-invariant once the plate circle is detected
