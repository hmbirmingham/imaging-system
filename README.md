# Plate Imaging System

A Raspberry Pi 5–based bacterial colony imaging and quantification platform built for the UConn MLS laboratory. Captures backlit agar plate images, counts colonies, extracts per-colony features, flags anomalies using a two-layer statistical + machine-learning pipeline, and lets a PhD validate results per organism and culture medium.

The user interface is a Flask web app (Apple-style SPA) served at `http://localhost:5000` and shown full-screen in Chromium kiosk mode on the Pi.

---

## Hardware

| Component | Details |
|---|---|
| Computer | Raspberry Pi 5 |
| Camera | Raspberry Pi camera (picamera2) — runs in demo mode if absent |
| Backlight | White LED panel, hardware-PWM driven (RP1 mmap) |
| Interface | Touchscreen kiosk, or remote browser to `:5000` |

---

## Features

- **Live preview** with real-time brightness, contrast, and saturation controls
- **Colony detection** using background subtraction and watershed segmentation
- **Distance-invariant measurements** — all areas reported in mm² regardless of camera height
- **Haemolysis detection** — estimates halo brightness around each colony under backlight
- **Statistical anomaly detection** — flags unusual colonies by size, shape, and morphology from the first run
- **ML anomaly detection** — Random Forest + Isolation Forest, improves as labelled data accumulates
- **Organism × plate-type profiles** — 39 organisms and 11 culture media, with appearance resolved per medium
- **In-app validation workflow** — click-to-correct colonies, label anomalies, edit biology, and sign off; writes ground-truth labels to the training CSV and validated biology back to the profile
- **Passive data logging** — every run logs per-colony features to CSV for ML training
- **Annotated output images** — green = normal colony, red = anomaly flagged

---

## Project status & roadmap

| Capability | Status |
|---|---|
| Profile system (39 organisms × 11 plate types) | ✅ Built |
| Pipeline integration (swarming suppression, size bands) | ✅ Built |
| In-app validation UI | ✅ Built |
| Settings + ML Diagnostics | ✅ Built |
| **Biology validation (PhD sign-off)** | 🚩 **Not started** — all 39 profiles `validated: false` |
| **Per-plate appearance** (hemolysis, lactose, size, colour) | 🚩 **Not started** — seeded `unknown`, awaiting validation |
| **Ground-truth labels** (`is_anomaly`) | 🚩 **Not started** — 0 labelled colonies |
| **ML anomaly model** | 🚩 **Not trained** — needs ≥ 50 labelled colonies |
| **Counting / placement accuracy** | 🚩 **No data** — needs validated plates |
| Hardware — Custom Power Control PCB (incl. white LED board) | 🚩 **Pending** |
| Hardware — enclosure | ⏳ Pending (June) |

### Pending validation (flagged)

Until Dr. Ryberg signs off in-app, **treat all biology as provisional**:

- 🚩 All 39 organism profiles are unvalidated defaults. Organism-level facts (Gram, swarming, incubation window) are safe textbook values; **per-plate appearance is `unknown`** and must be validated per medium.
- 🚩 Profile-aware size flagging stays **off** until a profile is validated (the statistical anomaly layer still runs).
- 🚩 The ML anomaly classifier is **inactive** until labelled data accumulates through validation.

---

## Organism & plate profiles

Profiles live as YAML and are split into two concerns:

```
profiles/
  organisms/        39 organism profiles + _generic.yaml (fallback)
  plate_types/      11 culture media (BAP, MAC, CHOC, XLD, …)
```

- **`instrument` block** — values *you* set and lock after enclosure calibration (plate geometry, exposure hints). Owned by the operator.
- **`biology` block** — values the PhD validates. Because the same organism reads differently on different media (E. coli is grey on BAP, a pink lactose-fermenter on MAC), appearance is stored **per plate type** under `biology.plates`, with a `default` block for any medium not described. Organism-level facts (Gram, swarming, incubation window) sit above the per-plate blocks.

Profiles ship **seeded with safe textbook facts only** (Gram, cell morphology, swarming, incubation, standard media) and `validated: false`. Per-plate appearance (hemolysis, lactose reaction, colony size, colour) starts `unknown` and is filled by validation — no clinical specifics are asserted until a human signs off.

The pipeline uses the active profile to **suppress spreading/touching flags for swarming organisms** (e.g. *Proteus mirabilis*) and, **once a profile is validated**, to flag colonies outside the expected size band for that medium.

Regenerate or extend the roster with the seeder (existing validated profiles are preserved):

```bash
python3 seed_profiles.py                 # (re)seed organisms + plate types
python3 seed_profiles.py --skip-validated # never overwrite signed-off profiles
python3 profiles.py                       # inspect the roster
```

New organisms and plate types can also be added in-app.

---

## Project structure

```
server.py          — Flask web backend (MJPEG stream, SSE, REST API)
imager.py          — Standalone capture/quantify entry point
quantify.py        — Colony detection, segmentation, feature extraction
anomaly.py         — Two-layer anomaly detection (statistical + ML)
profiles.py        — Organism × plate-type profile store + pipeline application
seed_profiles.py   — One-shot seeder for the MLS organism/plate roster
app_settings.py    — Persistent device settings (plate diameter, sensitivity, …)
ml_diagnostics.py  — Counting/placement accuracy, training history, model card
data_logger.py     — Per-colony feature + validation-label CSV logger
led_pwm.py / .c    — Hardware-PWM LED driver (RP1 mmap), auto-compiled wrapper
templates/, static/— Web UI (index.html, app.js, styles.css)
profiles/          — Organism and plate-type YAML profiles
launch.sh          — Kiosk launcher (handles venv + DISPLAY)
imager.desktop     — Linux desktop shortcut
captures/          — Saved plate images
results/           — Annotated output images
data/              — Colony feature CSV (training data)
models/            — Saved ML model (generated after training)
```

---

## Setup

**1. Clone and install dependencies**
```bash
git clone https://github.com/hmbirmingham/imaging-system.git ~/imager/system/imaging-system
cd ~/imager/system/imaging-system
python3 -m venv venv --system-site-packages    # picamera2 is system-installed
source venv/bin/activate
pip install -r requirements.txt
```

`--system-site-packages` lets the venv see the system `picamera2`. `pyyaml` is required for the profile system; if it is missing the app still runs, with profile features disabled.

**2. Run the web UI**
```bash
source venv/bin/activate
python3 server.py          # → http://localhost:5000
```

**3. Optional — desktop / kiosk shortcut**
```bash
chmod +x launch.sh
cp imager.desktop ~/.local/share/applications/
cp imager.desktop ~/Desktop/
```
Update the `Exec=` path in `imager.desktop` to match your install directory first.

---

## Running without hardware

The system detects a missing camera at startup and shows a **No Camera Connected** state (demo mode) — capture is disabled, but the library, quantification of existing images, anomaly detection, profiles, and validation all remain available. The LED is greyed out until the board is present.

---

## Workflow

1. **Select sample** — choose the organism and plate type in the Capture view (defaults to *Unknown*)
2. **Capture** — click Capture Image, or pick an existing image from the library
3. **Quantify** — runs colony detection; the active profile is applied (swarming suppression, expected size). The annotated image and a results panel (count, area stats, anomalies, haemolysis, coverage) appear
4. **Validate** — open a result in the library, click **Validate & Sign off**:
   - Click colonies on the plate to cycle **normal → anomaly → false-positive**; click empty agar to **add** a missed colony
   - Assign/confirm the organism and plate, and edit the per-plate biology fields
   - Enter a name and sign off
5. Validation writes per-colony `is_anomaly` ground-truth labels and the corrected count to `data/colony_features.csv`, and the validated biology + sign-off back to the organism profile YAML

Every run logs features automatically; validation is what turns those rows into labelled training data.

---

## Settings & ML Diagnostics

The sidebar has **Profiles** (browse all 39 organisms, edit per-plate biology, sign off) and **Settings**. Settings is tabbed:

- **General** — plate diameter (propagates into every profile's instrument block and drives mm² calibration), anomaly sensitivity (Z-threshold 1.5–3.5; standard deviations from the plate mean), default organism/plate, and auto-quantify. JPEG quality is fixed high and not exposed.
- **ML Diagnostics** — two accuracy tracks:
  - *Counting & placement accuracy* — derived purely from validated plates (no trained model needed): average count error vs. your validated count, % within ±1, and detection precision/recall (from false-positive and missed clicks), with an auto-vs-validated count chart.
  - *Anomaly classifier* — labelling progress, and once trained, recall/precision/F1 in plain language plus feature importances. **Retrain** is guarded by a minimum label count and regenerates the model card.
- **About** — live camera/LED/profile status.

The profile is always applied during analysis (swarming suppression, expected size); a "show raw detection" view is available on results for diagnostics.

## Training the ML anomaly model

The ML layer activates once enough labelled data exists. Labels come from the in-app validation workflow (the `is_anomaly` column), so no manual CSV editing is needed. Check readiness and train:

```bash
python3 data_logger.py                       # dataset summary / readiness
python3 anomaly.py data/colony_features.csv  # train; saves models/anomaly.pkl
```

The model loads automatically on next launch.

---

## Training-data columns

`data/colony_features.csv` adds these over the base per-colony features (backward compatible — older CSVs auto-migrate on first run):

| Column | Meaning |
|---|---|
| `plate_type` | Culture medium code (BAP, MAC, …); default `unknown` |
| `profile_id` | Organism profile id; default `unknown` |
| `is_anomaly` | Ground-truth label (1/0) — filled by validation |
| `validation_status` | `confirmed` / `false_positive` / `added` / blank |
| `validated_by` | Who signed off |

---

## Dependencies

```
numpy, opencv-python-headless, Pillow, scipy
scikit-learn, pandas
flask, pyyaml
picamera2        # Raspberry Pi only (system package)
```

---

## Notes

- Designed for 90–100 mm standard petri dishes
- Backlit imaging — best with diffused, even backlight; a central hotspot reduces detection accuracy in the plate centre
- All colony areas are expressed in mm² and are distance-invariant once the plate circle is detected
- Commits use no co-author tags
