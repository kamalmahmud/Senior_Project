# Thermal Infiltration Detection
### Early Detection of Peripheral IV Infiltration/Extravasation via Thermal Video Analysis


---

## Overview

Peripheral intravenous (IV) catheter infiltration and extravasation are among the most common and underdetected complications in clinical settings. By the time visual signs (swelling, redness) appear, tissue damage has already occurred — particularly in chemotherapy patients and elderly or paediatric patients with fragile veins.

This project develops a **mobile, non-contact, real-time early warning system** that detects infiltration events from short thermal video clips before visual symptoms emerge. A lightweight deep learning model (TSM-MobileNetV2) analyses temperature patterns around the IV site, running entirely on-device with no cloud dependency.

**Target metrics (from proposal):**

| Metric                               | Target                          |
|--------------------------------------|---------------------------------|
| AUC                                  | ≥ 0.85                          |
| False alarms per hour (FA/h)         | ≤ 0.30                          |
| Mean early detection time            | ≥ 2 minutes before visual signs |
| End-to-end inference latency         | ≤ 2 seconds                     |
| Model size (after INT8 quantization) | ≤ 10 MB                         |

---

## Clinical Background

During normal IV infusion, cold IV fluid flows inside the vein and appears as a **narrow linear cool streak** in thermal imaging. When infiltration occurs, fluid leaks into subcutaneous tissue and spreads in a **fan-shaped pattern** from the puncture site — a thermally distinct signature that appears before any visible swelling.

Three thermographic patterns are clinically relevant:

| Pattern          | Thermal signature                         | Label           |
|------------------|-------------------------------------------|-----------------|
| Normal infusion  | Narrow cool streak along vein path        | Negative        |
| Extravasation    | Fan-shaped cold spread from puncture site | **Positive**    |
| Vein bifurcation | V-shaped cold area at vein fork           | Negative (hard) |

The bifurcation pattern is the primary source of false positives — it resembles extravasation spatially but is a normal anatomical variant. Temporal analysis across frames is key to distinguishing the two.

> **References:** Matsui et al. (2017), Nakamura et al. (2014), Hirata et al. ACS Sensors (2023)  
> reported AUC = 0.94 using static thermography by expert clinicians — this project targets video-temporal automation of the same task.

---

## Project Structure

```
thermal_infiltration/
│
├── data/
│   ├── synthetic.py        # Synthetic clip generation (Phase 1)
│   ├── channels.py         # Three-channel builder (T, T_smooth, ΔT)
│   └── dataset.py          # PyTorch Dataset — synthetic and real modes
│
├── models/
│   ├── tsm.py              # Temporal Shift Module
│   └── mobilenet.py        # TSM-MobileNetV2 architecture
│
├── training/
│   ├── trainer.py          # Training loop, evaluation, early stopping
│   └── metrics.py          # AUC, AUCPR, sensitivity, specificity, FA/h
│
├── utils/
│   └── clip_cutter.py      # Video → overlapping clips (for real data)
│
├── config.py               # All constants and hyperparameters
├── train.py                # Entry point
└── requirements.txt
```

---

## Model Architecture

```
Input: (B, 3, T, 112, 112)
  │
  │  Three channels per frame:
  │    Ch 0 — T        : raw temperature
  │    Ch 1 — T_smooth : temporally smoothed (window=3)
  │    Ch 2 — ΔT       : difference from baseline frame 0
  │
  ▼
┌─────────────────────────────┐
│  Temporal Shift Module      │  shifts 1/8 of channels ±1 frame
│  (no extra parameters)      │  so spatial conv sees temporal context
└─────────────────────────────┘
  │
  ▼
┌─────────────────────────────┐
│  MobileNetV2 backbone       │  pretrained on ImageNet
│  (processes B×T frames)     │  early layers frozen in Phase 2
└─────────────────────────────┘
  │
  ▼
┌─────────────────────────────┐
│  Temporal average pooling   │  mean across T frames → one vector
│  + Linear head (→ 2 class)  │
└─────────────────────────────┘
  │
  ▼
Output: (B, 2) logits → softmax → infiltration probability
```

**Why TSM-MobileNetV2?**
- TSM adds temporal reasoning with zero extra parameters
- MobileNetV2 is optimised for mobile inference — INT8 quantized model targets ≤ 10 MB
- Video-temporal analysis distinguishes fan-spread from bifurcation; single-frame methods cannot

---

## Two-Phase Development Plan

### Phase 1 — Synthetic data (current)
No real patient data required. Synthetic clips simulate:
- Realistic forearm skin temperature gradients
- Two-component camera noise (fixed pattern + temporal)
- Normal infusion (vein streak) for all negative clips
- Growing fan-shaped cold region for positive clips
- Randomised fan angle, spread rate, onset frame, and maximum cooling per clip

Goal: validate the full pipeline end-to-end. AUC = 1.0 on synthetic data is expected and does **not** indicate clinical validity — patterns are mathematically consistent and the model memorises them easily.

### Phase 2 — Real clinical data (after ethics approval)
Ethics application submitted under supervision of Dr. Şengül Dolu Kübilay. Clinical data collection at Lefke Avrupa Üniversitesi Hemşirelik Yüksekokulu.

- Phase 1 backbone weights used as initialisation
- Early layers frozen; only last blocks + head finetuned
- Leave-One-Patient-Out cross-validation to prevent data leakage
- Target: AUC ≥ 0.85, FA/h ≤ 0.30 on held-out test set

---

## Installation

```bash
git clone https://github.com/your-username/thermal-infiltration.git
cd thermal-infiltration
pip install -r requirements.txt
```

**Requirements:**
- Python ≥ 3.9
- torch ≥ 2.0.0
- torchvision ≥ 0.15.0
- numpy ≥ 1.24.0
- opencv-python ≥ 4.7.0
- scipy ≥ 1.10.0
- scikit-learn ≥ 1.2.0

---

## Usage

### Training with default settings
```bash
python train.py
```

### Custom training run
```bash
python train.py \
  --epochs 100 \
  --patience 15 \
  --batch-size 8 \
  --n-train 800 \
  --n-val 200 \
  --pos-ratio 0.2 \
  --lr 1e-3 \
  --device cuda \
  --save-path checkpoints/run1.pth
```

### Arguments

| Argument         | Default        | Description                  |
|------------------|----------------|------------------------------|
| `--epochs`       | 50             | Maximum training epochs      |
| `--patience`     | 10             | Early stopping patience      |
| `--batch-size`   | 8              | Batch size                   |
| `--n-train`      | 800            | Synthetic training samples   |
| `--n-val`        | 200            | Synthetic validation samples |
| `--pos-ratio`    | 0.2            | Fraction of positive clips   |
| `--lr`           | 1e-3           | AdamW learning rate          |
| `--weight-decay` | 1e-4           | AdamW weight decay           |
| `--device`       | auto           | `cuda` or `cpu`              |
| `--save-path`    | best_model.pth | Where to save best weights   |

---

## Three-Channel Input Design

Each raw clip of shape `(T, H, W)` is converted to three channels before the model sees it:

```python
T_ch     = clip                          # raw temperature values
T_smooth = uniform_filter1d(clip,        # noise-reduced trend
               size=3, axis=0)
delta_T  = clip - clip[0]               # change from baseline
```

**Why three channels matter:**

- `T` alone cannot distinguish a cool bifurcation from early infiltration
- `T_smooth` removes frame-to-frame flicker so the model reacts to trends, not noise
- `ΔT` is the most discriminating channel — infiltration shows a growing localised negative region; bifurcation shows a stable pattern; normal infusion shows near-zero ΔT outside the vein

---

## Metrics

All proposal-required metrics are computed in `training/metrics.py`:

```python
from training.metrics import compute_metrics, find_best_threshold, print_metrics

# find the threshold that meets FA/h target
best_thr = find_best_threshold(labels, probs, target_fa_h=0.3)

# compute all metrics at that threshold
metrics = compute_metrics(labels, probs, threshold=best_thr)
print_metrics(metrics)
```

Output:
```
--------------------------------------
  AUC          : 0.9459   (target ≥ 0.85)
  AUCPR        : 0.8821
  Sensitivity  : 0.8750
  Specificity  : 0.9125
  FA/h         : 0.2800  (target ≤ 0.30)
  Threshold    : 0.6340
  TP/TN/FP/FN  : 35 / 146 / 14 / 5
--------------------------------------
```

> **Note on FA/h during Phase 1:** The validation set contains ~0.11 hours of monitoring. FA/h values on small synthetic sets will appear inflated. This metric becomes meaningful with ≥ 1800 negative clips (≈ 1 hour). Interpret FA/h on synthetic data with caution.

---

## ROI and Privacy Design

- Camera frame is cropped to the IV site region only (ROI)
- No patient face, room, or identifying information is stored
- Only ROI-cropped thermal frames are retained after recording
- All processing runs on-device — no data leaves the phone
- KVKK compliant: no personally identifiable data stored

---

## Roadmap

- [x] Clip cutter — overlapping frame extraction from video
- [x] Three-channel builder — T, T_smooth, ΔT
- [x] Synthetic data generator — fan pattern, vein streak, randomised parameters
- [x] PyTorch Dataset and DataLoader — `(B, 3, T, 112, 112)` tensors
- [x] TSM-MobileNetV2 — temporal shift + lightweight backbone
- [x] Training loop — AdamW, early stopping, LR scheduling
- [x] Metrics — AUC, AUCPR, sensitivity, specificity, FA/h
- [ ] Hard negative generation — bifurcation false-positive clips
- [ ] Synthetic diversity — randomised puncture location, vein angle per clip
- [ ] INT8 quantization — TensorFlow Lite / ONNX Runtime Mobile export
- [ ] ECC ROI stabilization — micro-motion correction between frames
- [ ] Ethics application — Lefke Avrupa Üniversitesi clinical data collection
- [ ] Phase 2 finetuning — real clinical data, LOPO cross-validation
- [ ] TÜBİTAK 1512 / TÜSEB application — clinical validation funding

---

## License

MIT License — model weights, code, and SOP documents are freely usable for research and commercial development without restriction.

---

**Key references this work builds on:**

```
[1] Matsui et al. Journal of Infusion Nursing, 2017. doi:10.1097/NAN.0000000000000250
[2] Nakamura et al. Infrared Physics & Technology, 2014. doi:10.1016/j.infrared.2014.09.029
[3] Hirata et al. ACS Sensors, 2023. doi:10.1021/acssensors.2c02602
[4] Lin et al. TSM: Temporal Shift Module. ICCV 2019.
[5] Sandler et al. MobileNetV2. CVPR 2018.
```