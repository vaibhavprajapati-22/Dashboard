# 🛰️ Trajectory Analysis Dashboard

A modern, interactive Streamlit dashboard for comparing **Ground Truth** trajectories against **Kalman Filter** estimates, with accurate coordinate conversion, rich visualisations, and real-time metrics.

---

## 📁 Project Structure

```
trajectory_dashboard/
├── app.py                    # Main Streamlit dashboard
├── config_loader.py          # YAML config loading + validation
├── data_loader.py            # CSV loading, alignment, lat/lon → metres
├── metrics.py                # MSE, RMSE, error computation
├── plotting.py               # All Plotly visualisation functions
├── config.yaml               # Path & KF variant mappings
├── generate_sample_data.py   # Generate synthetic test data
├── requirements.txt
└── data/                     # (created by generate_sample_data.py)
    ├── path1_gt.csv          ← Ground Truth (circle)
    ├── path1_kf_standard.csv
    ├── path1_kf_extended.csv
    ├── path1_kf_unscented.csv
    ├── path1_kf_adaptive.csv
    ├── path2_gt.csv          ← Ground Truth (spiral)
    ├── ...                   (16 KF files + 4 GT files total)
```

---

## 🚀 Quick Start

### 1. Clone / download this folder

```bash
cd trajectory_dashboard
```

### 2. Create & activate a virtual environment (recommended)

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate sample data

```bash
python generate_sample_data.py
```

This creates **20 CSV files** (4 paths × 1 GT + 4 KF variants each) in the `data/` folder.

### 5. Launch the dashboard

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📂 Bring Your Own Data

### CSV format

**Ground Truth** (`data/my_path_gt.csv`):
```
sequence_no,latitude,longitude,ax,ay,az,baro_alt
1,30.37530,69.34510,0.012,-0.005,9.810,1000.0
2,30.37531,69.34512,0.015,-0.003,9.812,1000.1
...
```

**KF Output** (`data/my_path_kf.csv`):
```
sequence_no,KF_lat,KF_long
1,30.37529,69.34511
2,30.37532,69.34513
...
```

### Update `config.yaml`

```yaml
reference:
  lat_ref: 30.3753     # your reference latitude
  lon_ref: 69.3451     # your reference longitude

paths:
  My Path:
    gt_file: data/my_path_gt.csv
    kf_variants:
      KF Standard: data/my_path_kf.csv
      KF Extended: data/my_path_kf_ext.csv
```

---

## 🎯 Features

| Feature | Details |
|---|---|
| **Overlay plot** | GT + KF on same axes, zoom/pan/hover |
| **Side-by-side** | GT left · KF right |
| **Error plot** | Per-sequence Euclidean error + rolling avg ±1σ band |
| **Error histogram** | Distribution of errors |
| **Metric cards** | MSE · RMSE · Mean · Max · Min · Std (live update) |
| **Dark / Light theme** | Toggle in sidebar |
| **Export PNG** | Camera icon in every plot toolbar |
| **Download CSV** | Processed stats with one click |
| **Caching** | `@st.cache_data` for fast re-renders |

---

## 📐 Coordinate Conversion

Uses a **local tangent-plane** (flat-Earth) approximation:

```
x = (lon - lon_ref) × cos(lat_ref_rad) × (π/180) × R
y = (lat - lat_ref)                     × (π/180) × R
```

where `R = 6 371 000 m` (Earth radius).  
Accurate to < 0.1 % error for trajectories up to ~50 km.

---

## 🛠️ Customisation Tips

- **Add a new KF variant**: add an entry under `kf_variants` in `config.yaml` — no code changes needed.
- **Change the reference point**: update `lat_ref` / `lon_ref` in `config.yaml`.
- **Different colour scheme**: edit the palette constants at the top of `plotting.py`.
- **More metrics**: add fields to `TrajectoryMetrics` in `metrics.py` and render cards in `app.py`.
