# FCSViz

An interactive browser-based viewer for Flow Cytometry Standard (FCS) files built with Streamlit and Plotly. Configure a grid of plots, each showing a different channel combination or file, with live zoom and hover.

## Features

- **Multi-panel grid** — choose a 2×2, 3×2, or 3×3 layout; configure each subplot independently
- **Three plot types** per subplot:
  - **Scatter** — raw events coloured by a user-selected colour
  - **Density** — events coloured by local KDE density (Viridis colorscale); fast even on large files via grid interpolation
  - **Histogram** — single-channel event count with configurable bin count
- **FCS file upload** — drag-and-drop one or more `.fcs` files; data persists while you navigate between pages
- **Metadata page** — instrument model, acquisition date, total events, and per-channel parameters (PMT voltage, gain, range) for every loaded file
- **Demo mode** — subplots without a file show synthetic two-population data so you can explore the UI before loading any files

## Installation

Requires Python 3.9+ and conda (or a plain venv).

```bash
# clone the repo
git clone <repo-url>
cd FCSViz

# create and activate environment
conda create -n fscviz python=3.9 -y
conda activate fscviz

# install dependencies
pip install -r requirements.txt
```

## Running

```bash
conda activate fscviz
streamlit run FCSviz.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

## Usage

1. **Upload files** — use the sidebar file uploader to load one or more `.fcs` files
2. **Click a plot title** to open the configure dialog for that panel
3. **Choose** a data source, plot type, channels, and (for scatter/histogram) a colour, then click **Apply**
4. **Navigate** to the **Metadata** page via the sidebar to inspect instrument and channel settings for loaded files
5. **Remove a file** by clicking the `×` button next to it in the sidebar

## Dependencies

| Package | Version |
|---|---|
| streamlit | ≥ 1.35 |
| plotly | ≥ 5.22 |
| numpy | ≥ 1.26 |
| pandas | ≥ 2.0 |
| fcsparser | ≥ 0.2 |
| scipy | ≥ 1.11 |

## Project structure

```
FCSViz/
├── FCSviz.py               # main page — grid viewer
├── pages/
│   └── 1_Metadata.py       # metadata page
├── requirements.txt
├── .streamlit/
│   └── config.toml         # forces light mode
└── test_fcs_loader.py      # standalone FCS parse/validation script
```
