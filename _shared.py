"""Shared constants, session-state initialisation, and upload sidebar.

Imported by every page so session state is always ready regardless of
which page the user lands on first.
"""
import os
import tempfile

import fcsparser
import numpy as np
import pandas as pd
import streamlit as st
from gates import GateTree

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
_DEMO_KEY = "__demo__"
_NO_FILE  = "— random data —"
N_CELLS   = 5_000

PRESET_COLORS = {
    "Blue":   "#1f77b4",
    "Red":    "#d62728",
    "Green":  "#2ca02c",
    "Orange": "#ff7f0e",
    "Purple": "#9467bd",
    "Teal":   "#17becf",
}

PLOT_TYPES = ["Scatter", "Density", "Histogram"]
TRANSFORMS = ["Linear", "Log", "Ln", "asinh", "biex"]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def apply_transform(arr: np.ndarray, transform: str, cofactor: float = 150) -> np.ndarray:
    if transform == "Log":
        return np.where(arr > 0, np.log10(arr), np.nan)
    if transform == "Ln":
        return np.where(arr > 0, np.log(arr), np.nan)
    if transform == "asinh":
        return np.arcsinh(arr / cofactor)
    if transform == "biex":
        return np.sign(arr) * np.log10(np.abs(arr) / cofactor + 1)
    return arr  # Linear


def _demo_data(rng: np.random.Generator, n: int = N_CELLS) -> tuple:
    """Two log-normal populations spanning ~3 decades, with some negatives."""
    n1, n2 = int(n * 0.6), int(n * 0.4)
    x = np.concatenate([
        rng.lognormal(mean=np.log(500),   sigma=0.6, size=n1),
        rng.lognormal(mean=np.log(30000), sigma=0.5, size=n2),
    ])
    y = np.concatenate([
        rng.lognormal(mean=np.log(400),   sigma=0.7, size=n1),
        rng.lognormal(mean=np.log(40000), sigma=0.4, size=n2),
    ])
    neg_mask = rng.random(n1) < 0.15
    x[:n1][neg_mask] -= rng.exponential(200, size=neg_mask.sum())
    y[:n1][neg_mask] -= rng.exponential(150, size=neg_mask.sum())
    return x, y


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def init_session_state() -> None:
    """Initialise all session-state keys. Idempotent — safe to call from any page."""
    if "fcs_data" not in st.session_state:
        st.session_state.fcs_data = {}

    if _DEMO_KEY not in st.session_state.get("fcs_data", {}):
        _drng = np.random.default_rng(seed=42)
        _dx, _dy = _demo_data(_drng)
        st.session_state.fcs_data[_DEMO_KEY] = {
            "data": pd.DataFrame({"X": _dx, "Y": _dy}),
            "channels": ["X", "Y"],
            "meta": {},
        }

    if "gate_trees" not in st.session_state:
        st.session_state.gate_trees = {}

    if _DEMO_KEY not in st.session_state.get("gate_trees", {}):
        st.session_state.gate_trees[_DEMO_KEY] = GateTree()

    if "subplot_config" not in st.session_state:
        st.session_state.subplot_config = {}

    if "dialog_coords" not in st.session_state:
        st.session_state.dialog_coords = (1, 1)

    if "active_gate_tool" not in st.session_state:
        st.session_state.active_gate_tool = None

    if "drawing_subplot" not in st.session_state:
        st.session_state.drawing_subplot = None

    if "pending_gate" not in st.session_state:
        st.session_state.pending_gate = None

    if "processed_selection" not in st.session_state:
        st.session_state.processed_selection = {}

    if "gate_vertices" not in st.session_state:
        st.session_state.gate_vertices = []

    if "subplot_gates" not in st.session_state:
        st.session_state.subplot_gates = {}

    if "gate_groups" not in st.session_state:
        st.session_state.gate_groups = {}

    if "hover_pt" not in st.session_state:
        st.session_state.hover_pt = {}

    if "file_labels" not in st.session_state:
        st.session_state.file_labels = {}


# ---------------------------------------------------------------------------
# Upload sidebar (shared across pages)
# ---------------------------------------------------------------------------
def render_upload_sidebar() -> None:
    """File upload widget + loaded-file list with remove buttons.

    Call inside ``with st.sidebar:`` on any page that needs file management.
    Gate tree and export controls are Plot-page-only and rendered separately.
    """
    st.header("Data")
    uploaded_files = st.file_uploader(
        "Upload FCS files",
        type=["fcs"],
        accept_multiple_files=True,
        help="Upload one or more .fcs files",
    )

    for uf in uploaded_files or []:
        if uf.name not in st.session_state.fcs_data:
            try:
                with tempfile.NamedTemporaryFile(suffix=".fcs", delete=False) as tmp:
                    tmp.write(uf.getvalue())
                    tmp_path = tmp.name
                try:
                    meta, data = fcsparser.parse(tmp_path, reformat_meta=True)
                finally:
                    os.unlink(tmp_path)
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                st.session_state.fcs_data[uf.name] = {
                    "data": data[numeric_cols],
                    "channels": numeric_cols,
                    "meta": meta,
                }
                st.session_state.gate_trees[uf.name] = GateTree()
                if uf.name not in st.session_state.file_labels:
                    _next_n = len(st.session_state.file_labels) + 1
                    st.session_state.file_labels[uf.name] = f"File {_next_n}"
            except Exception as e:
                st.error(f"Failed to parse {uf.name}: {e}")

    _real_files = [k for k in st.session_state.fcs_data if k != _DEMO_KEY]
    if _real_files:
        st.write("**Loaded files:**")
        for name in _real_files:
            info = st.session_state.fcs_data[name]
            n_events = len(info["data"])
            n_ch = len(info["channels"])
            col_label, col_btn = st.columns([4, 1])
            _lbl = st.session_state.file_labels.get(name, name)
            col_label.caption(
                f"**{_lbl}** — {os.path.basename(name)}  \n{n_events:,} events · {n_ch} ch"
            )
            if col_btn.button("×", key=f"_rm_{name}", help=f"Remove {name}"):
                _rm_tree = st.session_state.gate_trees.pop(name, None)
                if _rm_tree:
                    _rm_ids = set(_rm_tree._gates.keys())
                    for _sg in st.session_state.get("subplot_gates", {}).values():
                        _sg -= _rm_ids
                    for _gid in _rm_ids:
                        st.session_state.gate_groups.pop(_gid, None)
                del st.session_state.fcs_data[name]
                st.session_state.file_labels.pop(name, None)
                for cfg in st.session_state.subplot_config.values():
                    if cfg.get("file") == name:
                        cfg.update({"configured": False, "file": None, "x_ch": None, "y_ch": None})
                st.rerun()
