import json
import os
import tempfile

import fcsparser
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from gates import Gate, GateTree, RECTANGLE, POLYGON, QUADRANT, THRESHOLD_V, THRESHOLD_H

st.set_page_config(page_title="FCSViz", layout="wide")
st.title("FCSViz")

with st.expander("How to use", expanded=False):
    st.markdown(
        "- **Upload** one or more FCS files using the sidebar panel on the left\n"
        "- **Click a plot title** to open the configuration dialog for that panel\n"
        "- **Select** a data source, plot type (Scatter / Density / Histogram), channels, transform (Linear / Log / Ln / asinh / biex), and color, then click **Apply**\n"
        "- Click the title again at any time to reconfigure or switch channels"
    )

LAYOUTS = {
    "2 × 2": (2, 2),
    "2 × 3": (2, 3),
    "3 × 2": (3, 2),
    "3 × 3": (3, 3),
}

N_CELLS = 5_000
_NO_FILE = "— random data —"

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


def _inverse_transform(val: float, transform: str, cofactor: float = 150) -> float:
    """Invert apply_transform — convert display-space value back to raw data space."""
    if transform == "Log":
        return 10.0 ** val
    if transform == "Ln":
        return float(np.exp(val))
    if transform == "asinh":
        return float(np.sinh(val) * cofactor)
    if transform == "biex":
        return float(np.sign(val) * cofactor * (10.0 ** abs(val) - 1.0))
    return val  # Linear passthrough


def _axis_label(channel: str, transform: str) -> str:
    return channel if transform == "Linear" else f"{channel} ({transform})"


def _demo_data(rng: np.random.Generator, n: int = N_CELLS) -> tuple:
    """Two log-normal populations spanning ~3 decades, with some negatives from compensation."""
    n1, n2 = int(n * 0.6), int(n * 0.4)
    x = np.concatenate([
        rng.lognormal(mean=np.log(500),   sigma=0.6, size=n1),
        rng.lognormal(mean=np.log(30000), sigma=0.5, size=n2),
    ])
    y = np.concatenate([
        rng.lognormal(mean=np.log(400),   sigma=0.7, size=n1),
        rng.lognormal(mean=np.log(40000), sigma=0.4, size=n2),
    ])
    # Add realistic negatives (~15 % of dim population) to motivate asinh/biex over log
    neg_mask = rng.random(n1) < 0.15
    x[:n1][neg_mask] -= rng.exponential(200, size=neg_mask.sum())
    y[:n1][neg_mask] -= rng.exponential(150, size=neg_mask.sum())
    return x, y


# --- Gate drawing helpers ---

def _cancel_drawing():
    st.session_state.active_gate_tool = None
    st.session_state.drawing_subplot = None
    st.session_state.gate_vertices = []
    st.session_state.processed_selection = {}
    st.session_state.hover_pt = {}


def _is_drawable_subplot(r, c):
    cfg = st.session_state.subplot_config.get((r, c), {})
    if not cfg.get("configured"):
        return False
    if cfg.get("plot_type") not in ("Scatter", "Density"):
        return False
    if not (cfg.get("x_ch") and cfg.get("y_ch")):
        return False
    file = cfg.get("file")
    return file is None or file in st.session_state.fcs_data


def _polygon_to_svg_path(vertices):
    if len(vertices) < 3:
        return ""
    pts = " L ".join(f"{x},{y}" for x, y in vertices)
    return f"M {pts} Z"


def _hex_to_rgba(hex_color, alpha=0.12):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _fmt_n(n: int) -> str:
    """Compact event count: raw if <10k, else 3-sig-fig scientific (e.g. 6.30E4)."""
    if n < 10_000:
        return f"{n:,}"
    exp = int(np.floor(np.log10(max(n, 1))))
    m = n / 10 ** exp
    mantissa = f"{m:.2f}".rstrip("0").rstrip(".")
    return f"{mantissa}E{exp}"


def _selection_fingerprint(sel):
    return json.dumps(sel, sort_keys=True, default=str)


# --- Session state init ---
if "fcs_data" not in st.session_state:
    st.session_state.fcs_data = {}

if "gate_trees" not in st.session_state:
    # {filename: GateTree} — one tree per loaded file
    st.session_state.gate_trees = {}

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
    st.session_state.gate_vertices = []   # list of (x, y) tuples for WIP gate

if "subplot_gates" not in st.session_state:
    st.session_state.subplot_gates = {}     # {(r, c): set of gate_ids drawn on that subplot}

if "hover_pt" not in st.session_state:
    st.session_state.hover_pt = {}        # {(r, c): (hx, hy)} — current hover position per subplot

if "file_labels" not in st.session_state:
    st.session_state.file_labels = {}     # {filename: "File 1", ...}

st.divider()
col1, col2 = st.columns([5, 1], vertical_alignment="center")
with col2:
    st.markdown('<span id="_layout_sel"></span>', unsafe_allow_html=True)
    layout_choice = st.selectbox(
        "Grid layout",
        list(LAYOUTS.keys()),
        label_visibility="collapsed",
    )
with col1:
    st.markdown("<p style='text-align:right;margin:0'>Plot layout:</p>", unsafe_allow_html=True)

st.markdown(
    "<style>"
    "div:has(#_layout_sel) [data-testid='stSelectbox'] [data-baseweb='select'] div{"
    "text-align:center!important;justify-content:center!important;}"
    "[data-baseweb='menu'] li{"
    "text-align:center!important;justify-content:center!important;}"
    "[data-baseweb='menu'] li div{"
    "text-align:center!important;justify-content:center!important;}"
    "[data-baseweb='popover'] li{"
    "text-align:center!important;justify-content:center!important;}"
    "</style>",
    unsafe_allow_html=True,
)

# --- Gate toolbar ---
_has_fcs = bool(st.session_state.fcs_data)
_TOOLS = [
    ("rectangle",   "▭", "Rectangle gate"),
    ("polygon",     "⬠", "Polygon gate (lasso)"),
    ("quadrant",    "⊞", "Quadrant gates (4 regions)"),
    ("threshold_v", "|",  "Vertical threshold"),
    ("threshold_h", "—",  "Horizontal threshold"),
]
_tool_cols = st.columns([1, 1, 1, 1, 1, 4])
for (_tid, _icon, _tip), _col in zip(_TOOLS, _tool_cols):
    with _col:
        _is_active = st.session_state.active_gate_tool == _tid
        if st.button(
            _icon, key=f"_tool_{_tid}", help=_tip,
            disabled=not _has_fcs,
            type="primary" if _is_active else "secondary",
            use_container_width=True,
        ):
            if _is_active:
                _cancel_drawing()
            else:
                _cancel_drawing()
                st.session_state.active_gate_tool = _tid
            st.rerun()

# Drawing instruction shown inline in the toolbar's right column
with _tool_cols[-1]:
    _active_tool = st.session_state.active_gate_tool
    _drawing_rc  = st.session_state.drawing_subplot
    _verts       = st.session_state.gate_vertices
    if _active_tool and not _drawing_rc:
        st.caption(f"**{_active_tool}** selected — click a subplot title to start drawing.")
    elif _active_tool and _drawing_rc:
        if _active_tool == "rectangle":
            st.caption("✏️ Drag on the plot to draw the gate rectangle.")
        elif _active_tool == "polygon":
            _n = len(_verts)
            if _n < 3:
                st.caption(f"✏️ Click to add vertices ({max(0, 3 - _n)} more needed).")
            else:
                if st.button(f"✓ Finish gate ({_n} pts)", type="primary", use_container_width=True):
                    _cfg_d = st.session_state.subplot_config.get(_drawing_rc, {})
                    _x_tr = _cfg_d.get("x_transform", "Linear"); _x_cof = _cfg_d.get("x_cofactor", 150)
                    _y_tr = _cfg_d.get("y_transform", "Linear"); _y_cof = _cfg_d.get("y_cofactor", 150)
                    _raw_verts = [
                        (_inverse_transform(v[0], _x_tr, _x_cof),
                         _inverse_transform(v[1], _y_tr, _y_cof))
                        for v in _verts
                    ]
                    _g = Gate(
                        name="Gate", gate_type=POLYGON,
                        x_channel=_cfg_d.get("x_ch") or "X",
                        y_channel=_cfg_d.get("y_ch") or "Y",
                        params={"vertices": _raw_verts},
                    )
                    st.session_state.pending_gate = {"gate_obj": _g, "file": _cfg_d.get("file"), "subplot_rc": _drawing_rc, "parent_id": _cfg_d.get("gate_id") or GateTree.ROOT_ID}
                    st.session_state.gate_vertices = []
                    st.session_state.drawing_subplot = None
                    st.rerun()
        elif _active_tool in ("quadrant", "threshold_v", "threshold_h"):
            st.caption("✏️ Click anywhere on the plot to place the gate.")

# --- Sidebar ---
with st.sidebar:
    st.header("Data")
    uploaded_files = st.file_uploader(
        "Upload FCS files",
        type=["fcs"],
        accept_multiple_files=True,
        help="Upload one or more .fcs files",
    )

    # Parse newly uploaded files (skip already-loaded ones)
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

    # Show loaded files with explicit remove buttons
    if st.session_state.fcs_data:
        st.write("**Loaded files:**")
        for name in list(st.session_state.fcs_data):
            info = st.session_state.fcs_data[name]
            n_events = len(info["data"])
            n_ch = len(info["channels"])
            col_label, col_btn = st.columns([4, 1])
            _lbl = st.session_state.file_labels.get(name, name)
            col_label.caption(f"**{_lbl}** — {os.path.basename(name)}  \n{n_events:,} events · {n_ch} ch")
            if col_btn.button("×", key=f"_rm_{name}", help=f"Remove {name}"):
                _rm_tree = st.session_state.gate_trees.pop(name, None)
                if _rm_tree:
                    _rm_ids = set(_rm_tree._gates.keys())
                    for _sg in st.session_state.get("subplot_gates", {}).values():
                        _sg -= _rm_ids
                del st.session_state.fcs_data[name]
                st.session_state.file_labels.pop(name, None)
                for cfg in st.session_state.subplot_config.values():
                    if cfg.get("file") == name:
                        cfg.update({"configured": False, "file": None, "x_ch": None, "y_ch": None})
                st.rerun()

    # Gate tree panel — shown when any loaded file has gates
    _any_gates = any(len(t) > 0 for t in st.session_state.gate_trees.values())
    if st.session_state.fcs_data and _any_gates:
        st.divider()
        st.write("**Gates**")
        st.markdown("""<style>
section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]{margin-bottom:-0.7rem}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p{margin:0;line-height:1.3}
section[data-testid="stSidebar"] button[kind="secondary"]{
    padding:0.05rem 0.3rem!important;min-height:unset!important;
    height:1.4rem!important;line-height:1!important}
</style>""", unsafe_allow_html=True)
        for _fname, _tree in st.session_state.gate_trees.items():
            if _fname not in st.session_state.fcs_data or len(_tree) == 0:
                continue
            _flat = _tree.flat_list()
            st.caption(st.session_state.file_labels.get(_fname, os.path.basename(_fname)))
            for _depth, _gate in _flat:
                _prefix = ("&nbsp;&nbsp;" * (_depth - 1) + "└─&nbsp;") if _depth > 0 else ""
                _rcols = st.columns([9, 1])
                with _rcols[0]:
                    st.markdown(
                        f'<span style="color:{_gate.color};font-weight:600;font-size:0.85rem">'
                        f'{_prefix}{_gate.name}</span>',
                        unsafe_allow_html=True,
                    )
                with _rcols[1]:
                    if st.button("×", key=f"_del_gate_{_gate.id}",
                                 help=f"Delete {_gate.name}"):
                        _tree.remove_gate(_gate.id)
                        for _cfg in st.session_state.subplot_config.values():
                            if _cfg.get("gate_id") == _gate.id:
                                _cfg["gate_id"] = None
                        for _sg in st.session_state.get("subplot_gates", {}).values():
                            _sg.discard(_gate.id)
                        st.rerun()

rows, cols = LAYOUTS[layout_choice]

# Reset seeds when layout changes
if st.session_state.get("last_layout") != layout_choice:
    st.session_state.seeds = {}
    st.session_state.last_layout = layout_choice

for r in range(1, rows + 1):
    for c in range(1, cols + 1):
        st.session_state.seeds.setdefault((r, c), (r - 1) * cols + (c - 1))
        st.session_state.subplot_config.setdefault(
            (r, c), {"configured": False, "file": None, "plot_type": "Scatter", "x_ch": None, "y_ch": None, "n_bins": 256, "color": "#1f77b4", "x_transform": "Linear", "y_transform": "Linear", "x_cofactor": 150, "y_cofactor": 150, "gate_id": None}
        )


_dr, _dc = st.session_state.dialog_coords


@st.dialog(f"Configure subplot ({_dr}, {_dc})")
def plot_dialog(r: int, c: int):
    cfg = st.session_state.subplot_config[(r, c)]
    fcs_data = st.session_state.fcs_data

    # --- File selector ---
    saved_file = cfg.get("file")
    _label_map = {st.session_state.file_labels.get(f, f): f for f in fcs_data.keys()}
    file_display_opts = [_NO_FILE] + list(_label_map.keys())
    saved_display = st.session_state.file_labels.get(saved_file, saved_file) if saved_file else _NO_FILE
    file_idx = file_display_opts.index(saved_display) if saved_display in file_display_opts else 0
    selected_display = st.selectbox("Data source", file_display_opts, index=file_idx)
    selected_file = None if selected_display == _NO_FILE else _label_map.get(selected_display)

    # --- Display population selector ---
    # Reset gate_id when the file changes; preserve it when the file is unchanged
    selected_gate_id = cfg.get("gate_id") if cfg.get("file") == selected_file else None

    if selected_file and selected_file in st.session_state.gate_trees:
        _pop_tree = st.session_state.gate_trees[selected_file]
        if len(_pop_tree) > 0:
            _flat = _pop_tree.flat_list()
            _pop_labels = ["All Events"] + [
                " > ".join([a.name for a in _pop_tree.get_ancestors(_g.id)] + [_g.name])
                for _, _g in _flat
            ]
            _pop_ids = [None] + [_g.id for _, _g in _flat]
            _saved_pop_idx = _pop_ids.index(selected_gate_id) if selected_gate_id in _pop_ids else 0
            _pop_choice = st.selectbox(
                "Display population",
                range(len(_pop_labels)),
                format_func=lambda i: _pop_labels[i],
                index=_saved_pop_idx,
            )
            selected_gate_id = _pop_ids[_pop_choice]

    # --- Plot type ---
    plot_type = st.radio(
        "Plot type",
        PLOT_TYPES,
        index=PLOT_TYPES.index(cfg.get("plot_type", "Scatter")),
        horizontal=True,
    )

    # --- Channel selectors + transforms ---
    x_ch = y_ch = None
    x_transform = cfg.get("x_transform", "Linear")
    y_transform = cfg.get("y_transform", "Linear")
    x_cofactor = cfg.get("x_cofactor", 150)
    y_cofactor = cfg.get("y_cofactor", 150)
    x_tr_idx = TRANSFORMS.index(x_transform) if x_transform in TRANSFORMS else 0
    y_tr_idx = TRANSFORMS.index(y_transform) if y_transform in TRANSFORMS else 0

    if selected_file:
        channels = fcs_data[selected_file]["channels"]
        file_unchanged = cfg.get("file") == selected_file
        saved_x = cfg.get("x_ch") if file_unchanged else None
        saved_y = cfg.get("y_ch") if file_unchanged else None
        x_idx = channels.index(saved_x) if saved_x in channels else 0
        y_idx = channels.index(saved_y) if saved_y in channels else min(1, len(channels) - 1)

        if plot_type in ("Scatter", "Density"):
            col_x, col_y = st.columns(2)
            with col_x:
                x_ch = st.selectbox("X axis", channels, index=x_idx)
                x_transform = st.selectbox("X transform", TRANSFORMS, index=x_tr_idx, key=f"_xtr_{r}_{c}")
                if x_transform in ("asinh", "biex"):
                    x_cofactor = st.number_input("X cofactor", min_value=1, max_value=100_000, value=x_cofactor, step=10, key=f"_xcof_{r}_{c}")
            with col_y:
                y_ch = st.selectbox("Y axis", channels, index=y_idx)
                y_transform = st.selectbox("Y transform", TRANSFORMS, index=y_tr_idx, key=f"_ytr_{r}_{c}")
                if y_transform in ("asinh", "biex"):
                    y_cofactor = st.number_input("Y cofactor", min_value=1, max_value=100_000, value=y_cofactor, step=10, key=f"_ycof_{r}_{c}")
        else:  # Histogram
            x_ch = st.selectbox("Channel", channels, index=x_idx)
            x_transform = st.selectbox("Transform", TRANSFORMS, index=x_tr_idx, key=f"_xtr_{r}_{c}")
            if x_transform in ("asinh", "biex"):
                x_cofactor = st.number_input("Cofactor", min_value=1, max_value=100_000, value=x_cofactor, step=10, key=f"_xcof_{r}_{c}")
            y_transform = "Linear"
    else:
        st.caption("No file selected — subplot will show random data.")
        if plot_type in ("Scatter", "Density"):
            col_x, col_y = st.columns(2)
            with col_x:
                x_transform = st.selectbox("X transform", TRANSFORMS, index=x_tr_idx, key=f"_xtr_{r}_{c}")
                if x_transform in ("asinh", "biex"):
                    x_cofactor = st.number_input("X cofactor", min_value=1, max_value=100_000, value=x_cofactor, step=10, key=f"_xcof_{r}_{c}")
            with col_y:
                y_transform = st.selectbox("Y transform", TRANSFORMS, index=y_tr_idx, key=f"_ytr_{r}_{c}")
                if y_transform in ("asinh", "biex"):
                    y_cofactor = st.number_input("Y cofactor", min_value=1, max_value=100_000, value=y_cofactor, step=10, key=f"_ycof_{r}_{c}")
        else:  # Histogram
            x_transform = st.selectbox("Transform", TRANSFORMS, index=x_tr_idx, key=f"_xtr_{r}_{c}")
            if x_transform in ("asinh", "biex"):
                x_cofactor = st.number_input("Cofactor", min_value=1, max_value=100_000, value=x_cofactor, step=10, key=f"_xcof_{r}_{c}")
            y_transform = "Linear"

    # --- Bin count (histogram only) ---
    n_bins = cfg.get("n_bins", 256)
    if plot_type == "Histogram":
        n_bins = st.number_input("Bins", min_value=10, max_value=1024, value=n_bins, step=10)

    st.divider()
    # --- Color ---
    if plot_type != "Density":
        saved_color = cfg.get("color", "#1f77b4")
        saved_preset = next((k for k, v in PRESET_COLORS.items() if v == saved_color), "Custom")
        preset_options = list(PRESET_COLORS.keys()) + ["Custom"]

        # Inject a unique marker span so CSS can scope to only this radio widget
        # (sibling selector targets the color radio but not the plot-type radio above it)
        _mid = f"cpick-{r}-{c}"
        st.markdown(f'<span id="{_mid}"></span>', unsafe_allow_html=True)

        _rules = []
        for _i, (_name, _hex) in enumerate(PRESET_COLORS.items()):
            _n = _i + 1
            _rules.append(
                # Hide the radio circle
                f"div:has(#{_mid})~[data-testid='stRadio'] label:nth-child({_n})>div:first-child"
                f"{{display:none!important;}}"
                # Color the label text
                f"div:has(#{_mid})~[data-testid='stRadio'] label:nth-child({_n}) p"
                f"{{color:{_hex}!important;font-weight:700!important;font-size:14px!important;}}"
            )
        # Underline the selected option instead of a box ring
        _rules.append(
            f"div:has(#{_mid})~[data-testid='stRadio'] label:has(input:checked) p"
            f"{{text-decoration:underline!important;text-underline-offset:3px!important;}}"
        )
        st.markdown(f"<style>{''.join(_rules)}</style>", unsafe_allow_html=True)

        st.write("**Color**")
        color_choice = st.radio(
            "Color",
            preset_options,
            index=preset_options.index(saved_preset),
            horizontal=True,
            key=f"_crad_{r}_{c}",
            label_visibility="collapsed",
        )
        if color_choice == "Custom":
            final_color = st.color_picker(
                "Color",
                value=saved_color if saved_preset == "Custom" else "#1f77b4",
                label_visibility="collapsed",
            )
        else:
            final_color = PRESET_COLORS[color_choice]
    else:
        final_color = cfg.get("color", "#1f77b4")

    st.divider()
    btn1, btn2 = st.columns(2)
    with btn1:
        if st.button("Apply", use_container_width=True, type="primary"):
            # For demo (no file) scatter/density plots, assign synthetic channel names
            # so the subplot is eligible for gate drawing without needing a real file.
            if not selected_file and plot_type in ("Scatter", "Density"):
                eff_x_ch = "X"
                eff_y_ch = "Y"
            else:
                eff_x_ch = x_ch
                eff_y_ch = y_ch if plot_type in ("Scatter", "Density") else None
            st.session_state.subplot_config[(r, c)] = {
                "configured": True,
                "file": selected_file,
                "plot_type": plot_type,
                "x_ch": eff_x_ch,
                "y_ch": eff_y_ch,
                "n_bins": n_bins,
                "color": final_color,
                "x_transform": x_transform,
                "y_transform": y_transform,
                "x_cofactor": x_cofactor,
                "y_cofactor": y_cofactor,
                "gate_id": selected_gate_id,
            }
            if not selected_file:
                st.session_state.seeds[(r, c)] = np.random.randint(0, 100_000)
            st.rerun()
    with btn2:
        if st.button("Cancel", use_container_width=True):
            st.rerun()


def _blank_fig() -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        height=260,
        margin=dict(l=40, r=10, t=10, b=40),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f0f4f8",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def make_plot_fig(r: int, c: int) -> go.Figure:
    cfg = st.session_state.subplot_config.get((r, c), {})

    if not cfg.get("configured", False):
        return _blank_fig()

    fcs_data = st.session_state.fcs_data
    file = cfg.get("file")
    plot_type = cfg.get("plot_type", "Scatter")
    x_ch = cfg.get("x_ch")
    y_ch = cfg.get("y_ch")
    n_bins = cfg.get("n_bins", 256)
    color = cfg.get("color", "#1f77b4")
    x_transform = cfg.get("x_transform", "Linear")
    y_transform = cfg.get("y_transform", "Linear")

    _gate_id = cfg.get("gate_id")
    if file and file in fcs_data:
        _df_all = fcs_data[file]["data"]
        if _gate_id and file in st.session_state.gate_trees:
            _gtree = st.session_state.gate_trees[file]
            if _gate_id in _gtree._gates:
                _mask = _gtree.get_mask(_gate_id, _df_all)
                _file_df = _df_all[_mask].reset_index(drop=True)
            else:
                _file_df = _df_all
        else:
            _file_df = _df_all
    else:
        _file_df = None

    has_real_data = file and file in fcs_data and x_ch
    has_scatter_data = has_real_data and plot_type == "Scatter" and y_ch

    seed = st.session_state.seeds[(r, c)]
    rng = np.random.default_rng(seed=seed)

    if plot_type == "Histogram":
        if has_real_data:
            x = apply_transform(_file_df[x_ch].values, x_transform)
            x_label = _axis_label(x_ch, x_transform)
        else:
            x_raw, _ = _demo_data(rng)
            x = apply_transform(x_raw, x_transform)
            x_label = _axis_label("X", x_transform)

        trace = go.Histogram(
            x=x,
            nbinsx=n_bins,
            marker_color=color,
            opacity=0.8,
            hovertemplate=f"{x_label}: %{{x:.2f}}<br>Count: %{{y}}<extra></extra>",
        )
        fig = go.Figure(trace)
        fig.update_layout(
            height=260,
            margin=dict(l=40, r=10, t=10, b=40),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f0f4f8",
            xaxis_title=x_label if has_real_data else None,
            yaxis_title="Count",
            bargap=0.02,
        )

    elif plot_type == "Density":
        has_density_data = has_real_data and y_ch and file in fcs_data

        if has_density_data:
            df = _file_df
            x = apply_transform(df[x_ch].values, x_transform)
            y = apply_transform(df[y_ch].values, y_transform)
            x_label, y_label = _axis_label(x_ch, x_transform), _axis_label(y_ch, y_transform)
        else:
            x_raw, y_raw = _demo_data(rng)
            x = apply_transform(x_raw, x_transform)
            y = apply_transform(y_raw, y_transform)
            x_label, y_label = _axis_label("X", x_transform), _axis_label("Y", y_transform)

        # Drop NaNs before KDE (produced by Log/Ln on non-positive values)
        valid = np.isfinite(x) & np.isfinite(y)
        x_v, y_v = x[valid], y[valid]

        try:
            from scipy.stats import gaussian_kde
            from scipy.interpolate import RegularGridInterpolator
            n_pts = len(x_v)
            if n_pts > 5000:
                idx = np.random.default_rng(seed=42).choice(n_pts, size=5000, replace=False)
                kde = gaussian_kde(np.vstack([x_v[idx], y_v[idx]]))
            else:
                kde = gaussian_kde(np.vstack([x_v, y_v]))
            g = 150
            xi = np.linspace(x_v.min(), x_v.max(), g)
            yi = np.linspace(y_v.min(), y_v.max(), g)
            xi_g, yi_g = np.meshgrid(xi, yi)
            z = kde(np.vstack([xi_g.ravel(), yi_g.ravel()])).reshape(g, g)
            interp = RegularGridInterpolator(
                (xi, yi), z.T, method="linear", bounds_error=False, fill_value=0
            )
            density = np.where(valid, interp(np.column_stack([
                np.where(valid, x, x_v[0]),
                np.where(valid, y, y_v[0]),
            ])), 0)
        except Exception:
            density = np.zeros(len(x))

        trace = go.Scattergl(
            x=x, y=y,
            mode="markers",
            marker=dict(
                size=2, opacity=0.5,
                color=density, colorscale="Jet", showscale=True,
                colorbar=dict(thickness=12, len=0.75, title="Density"),
            ),
            showlegend=False,
            hovertemplate=f"{x_label}: %{{x:.2f}}<br>{y_label}: %{{y:.2f}}<extra></extra>",
        )
        fig = go.Figure(trace)
        fig.update_layout(
            height=260,
            margin=dict(l=40, r=10, t=10, b=40),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f0f4f8",
            xaxis_title=x_label if has_density_data else None,
            yaxis_title=y_label if has_density_data else None,
        )

    else:  # Scatter
        if has_scatter_data:
            df = _file_df
            x = apply_transform(df[x_ch].values, x_transform)
            y = apply_transform(df[y_ch].values, y_transform)
            x_label, y_label = _axis_label(x_ch, x_transform), _axis_label(y_ch, y_transform)
        else:
            x_raw, y_raw = _demo_data(rng)
            x = apply_transform(x_raw, x_transform)
            y = apply_transform(y_raw, y_transform)
            x_label, y_label = _axis_label("X", x_transform), _axis_label("Y", y_transform)

        trace = go.Scattergl(
            x=x, y=y,
            mode="markers",
            marker=dict(size=2, opacity=0.4, color=color),
            showlegend=False,
            hovertemplate=f"{x_label}: %{{x:.2f}}<br>{y_label}: %{{y:.2f}}<extra></extra>",
        )
        fig = go.Figure(trace)
        fig.update_layout(
            height=260,
            margin=dict(l=40, r=10, t=10, b=40),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f0f4f8",
            xaxis_title=x_label if has_scatter_data else None,
            yaxis_title=y_label if has_scatter_data else None,
        )

    # Gate overlays — gates stored in raw data space; re-apply subplot transform for display
    if plot_type in ("Scatter", "Density"):
        _gtrees = st.session_state.get("gate_trees", {})
        if file and file in _gtrees:
            _x_cof = cfg.get("x_cofactor", 150)
            _y_cof = cfg.get("y_cofactor", 150)
            def _tx(v):  # raw → plot x
                return float(apply_transform(np.array([v]), x_transform, _x_cof)[0])
            def _ty(v):  # raw → plot y
                return float(apply_transform(np.array([v]), y_transform, _y_cof)[0])

            _subplot_gate_ids = st.session_state.get("subplot_gates", {}).get((r, c), set())
            for _dep, _g in _gtrees[file].flat_list():
                # Only draw gates that were drawn on this specific subplot
                if _g.id not in _subplot_gate_ids:
                    continue
                # Only draw if this subplot is showing the gate's channels
                _x_ok = (_g.x_channel == x_ch)
                _y_ok = (_g.y_channel == y_ch)
                if _g.gate_type in (RECTANGLE, POLYGON, QUADRANT) and not (_x_ok and _y_ok):
                    continue
                if _g.gate_type == THRESHOLD_V and not _x_ok:
                    continue
                if _g.gate_type == THRESHOLD_H and not _y_ok:
                    continue

                _c = _g.color; _fill = _hex_to_rgba(_c, 0.10); _p = _g.params
                if _g.gate_type == RECTANGLE:
                    fig.add_shape(type="rect",
                        x0=_tx(_p["x_min"]), x1=_tx(_p["x_max"]),
                        y0=_ty(_p["y_min"]), y1=_ty(_p["y_max"]),
                        line=dict(color=_c, width=2), fillcolor=_fill)
                elif _g.gate_type == POLYGON:
                    _tv = [(_tx(vx), _ty(vy)) for vx, vy in _p["vertices"]]
                    fig.add_shape(type="path",
                        path=_polygon_to_svg_path(_tv),
                        line=dict(color=_c, width=2), fillcolor=_fill)
                elif _g.gate_type == QUADRANT and _p.get("quadrant") == "Q1":
                    fig.add_vline(x=_tx(_p["x0"]), line_color=_c, line_dash="dash", line_width=1.5)
                    fig.add_hline(y=_ty(_p["y0"]), line_color=_c, line_dash="dash", line_width=1.5)
                elif _g.gate_type == THRESHOLD_V:
                    fig.add_vline(x=_tx(_p["x0"]), line_color=_c, line_dash="dash", line_width=1.5)
                elif _g.gate_type == THRESHOLD_H:
                    fig.add_hline(y=_ty(_p["y0"]), line_color=_c, line_dash="dash", line_width=1.5)

    return fig


# --- Gate drawing ---

# Modebar buttons to remove in drawing mode so zoom/pan don't intercept drags.
_DRAW_MODEBAR_REMOVE = ["zoom", "pan", "zoomin", "zoomout", "autoscale", "resetscale"]


def _extract_sel_list(event, attr):
    """Return a selection attribute as a list, handling dict and dataclass event forms."""
    if event is None:
        return []
    sel = getattr(event, "selection", None)
    if sel is None:
        return []
    val = sel.get(attr, []) if isinstance(sel, dict) else (getattr(sel, attr, None) or [])
    return val if isinstance(val, list) else [val]


def _xy_from_point(pt):
    """Extract (x, y) scalars from a selected point (dict or dataclass)."""
    if isinstance(pt, dict):
        return pt.get("x"), pt.get("y")
    return getattr(pt, "x", None), getattr(pt, "y", None)


def _extract_hover_list(event):
    """Return hovered points from on_hover event."""
    if event is None:
        return []
    hover = getattr(event, "hover", None)
    if hover is None:
        return []
    pts = hover.get("points", []) if isinstance(hover, dict) else getattr(hover, "points", None)
    return list(pts) if pts else []


def _render_drawing_chart(r, c, tool):
    """Render a subplot in drawing mode. Rectangle uses drag-box; others use click."""
    cfg = st.session_state.subplot_config[(r, c)]
    fig = make_plot_fig(r, c)

    # Invisible SVG grid — go.Scatter (SVG) keeps DOM elements at opacity=0 so
    # pointer events (click-selection, hover) still fire anywhere on the plot.
    if fig.data:
        _first = fig.data[0]
        _xs = np.array(_first.x, dtype=float)
        _ys = np.array(_first.y, dtype=float)
        _ok = np.isfinite(_xs) & np.isfinite(_ys)
        if _ok.any():
            _xr, _yr = _xs[_ok], _ys[_ok]
            _xpad = (_xr.max() - _xr.min()) * 0.05 or 1.0
            _ypad = (_yr.max() - _yr.min()) * 0.05 or 1.0
            _gx = np.linspace(_xr.min() - _xpad, _xr.max() + _xpad, 50)
            _gy = np.linspace(_yr.min() - _ypad, _yr.max() + _ypad, 50)
            _gx_m, _gy_m = np.meshgrid(_gx, _gy)
            fig.add_trace(go.Scatter(
                x=_gx_m.ravel(), y=_gy_m.ravel(),
                mode="markers",
                marker=dict(size=14, opacity=0),
                showlegend=False,
                hoverinfo="none",
            ))

    if tool == "rectangle":
        # Drag-to-draw: set dragmode='select' so Plotly renders the selection box
        # live on the client during the drag — that IS the shape preview.
        # On mouse-up the box coordinates come back via event.selection.box.
        fig.update_layout(
            dragmode="select",
            modebar={"remove": _DRAW_MODEBAR_REMOVE},
        )
        event = st.plotly_chart(
            fig, use_container_width=True, key=f"chart_{r}_{c}_draw",
            on_select="rerun", selection_mode="box",
        )
        box = _extract_sel_list(event, "box")
        if box:
            bx = box[0]
            xr = bx.get("x") if isinstance(bx, dict) else getattr(bx, "x", None)
            yr = bx.get("y") if isinstance(bx, dict) else getattr(bx, "y", None)
            if xr and yr and len(xr) >= 2 and len(yr) >= 2:
                x_min, x_max = float(min(xr)), float(max(xr))
                y_min, y_max = float(min(yr)), float(max(yr))
                if x_max > x_min and y_max > y_min:   # ignore accidental zero-area clicks
                    _x_tr = cfg.get("x_transform", "Linear"); _x_cof = cfg.get("x_cofactor", 150)
                    _y_tr = cfg.get("y_transform", "Linear"); _y_cof = cfg.get("y_cofactor", 150)
                    gate = Gate(
                        name="Gate", gate_type=RECTANGLE,
                        x_channel=cfg.get("x_ch") or "X",
                        y_channel=cfg.get("y_ch") or "Y",
                        params={
                            "x_min": _inverse_transform(x_min, _x_tr, _x_cof),
                            "x_max": _inverse_transform(x_max, _x_tr, _x_cof),
                            "y_min": _inverse_transform(y_min, _y_tr, _y_cof),
                            "y_max": _inverse_transform(y_max, _y_tr, _y_cof),
                        },
                    )
                    st.session_state.pending_gate = {"gate_obj": gate, "file": cfg.get("file"), "subplot_rc": (r, c), "parent_id": cfg.get("gate_id") or GateTree.ROOT_ID}
                    st.session_state.drawing_subplot = None
                    st.rerun()

    else:
        # Click-based for polygon / quadrant / threshold
        verts = list(st.session_state.gate_vertices)

        # Confirmed polygon vertex preview
        if verts and tool == "polygon":
            vx = [v[0] for v in verts]
            vy = [v[1] for v in verts]
            close_x = vx + [vx[0]] if len(verts) >= 2 else vx
            close_y = vy + [vy[0]] if len(verts) >= 2 else vy
            fig.add_trace(go.Scatter(
                x=close_x, y=close_y, mode="lines+markers",
                line=dict(color="#e74c3c", dash="dot", width=2),
                marker=dict(color="#e74c3c", size=8),
                showlegend=False, hoverinfo="skip",
            ))

        # Key encodes len(verts) — forces fresh Plotly instance after each vertex
        # so the next click is always a fresh selection (no deselect-first needed)
        draw_key = f"chart_{r}_{c}_draw_{len(verts)}"
        fig.update_layout(
            clickmode="event+select",
            modebar={"remove": _DRAW_MODEBAR_REMOVE},
        )
        event = st.plotly_chart(
            fig, use_container_width=True, key=draw_key,
            on_select="rerun", selection_mode="points",
        )

        pts = _extract_sel_list(event, "points")
        if pts:
            x, y = _xy_from_point(pts[-1])
            if x is not None and y is not None:
                fp = _selection_fingerprint(f"{x},{y}")
                if st.session_state.processed_selection.get("last") != fp:
                    st.session_state.processed_selection["last"] = fp

                    if tool == "polygon":
                        new_verts = verts + [(float(x), float(y))]
                        st.session_state.gate_vertices = new_verts
                        st.session_state.processed_selection = {}
                        st.rerun()

                    else:
                        # Quadrant / threshold — one click, direct tree mutation
                        file = cfg.get("file")
                        x_ch = cfg.get("x_ch") or "X"
                        y_ch = cfg.get("y_ch") or "Y"
                        _x_tr = cfg.get("x_transform", "Linear"); _x_cof = cfg.get("x_cofactor", 150)
                        _y_tr = cfg.get("y_transform", "Linear"); _y_cof = cfg.get("y_cofactor", 150)
                        _rx = _inverse_transform(float(x), _x_tr, _x_cof)
                        _ry = _inverse_transform(float(y), _y_tr, _y_cof)
                        if file and file in st.session_state.gate_trees:
                            tree = st.session_state.gate_trees[file]
                            _parent_id = cfg.get("gate_id") or GateTree.ROOT_ID
                            if tool == "quadrant":
                                _new_ids = tree.add_quadrant_gates(_parent_id, x_ch, y_ch, _rx, _ry)
                            elif tool == "threshold_v":
                                _new_ids = tree.add_threshold_pair(_parent_id, "v", x_ch, _rx)
                            elif tool == "threshold_h":
                                _new_ids = tree.add_threshold_pair(_parent_id, "h", y_ch, _ry)
                            else:
                                _new_ids = []
                            st.session_state.subplot_gates.setdefault((r, c), set()).update(_new_ids)
                        _cancel_drawing()
                        st.rerun()



# --- Export helpers ---

def _render_subplot_png(r: int, c: int, cell_w: int = 700, cell_h: int = 520):
    """Return PNG bytes for one subplot, or None if kaleido is unavailable."""
    try:
        import plotly.io as pio
        fig = make_plot_fig(r, c)
        fig.update_layout(width=cell_w, height=cell_h, margin=dict(l=100, r=50, t=40, b=80))
        return pio.to_image(fig, format="png", width=cell_w, height=cell_h, scale=1)
    except Exception:
        return None


def _build_grid_image(rows: int, cols: int, cell_w: int = 700, cell_h: int = 520):
    """Return a PIL Image of the full subplot grid."""
    from PIL import Image
    import io

    grid_img = Image.new("RGB", (cell_w * cols, cell_h * rows), color=(255, 255, 255))
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            png_bytes = _render_subplot_png(r, c, cell_w, cell_h)
            if png_bytes:
                cell_img = Image.open(io.BytesIO(png_bytes))
                grid_img.paste(cell_img, ((c - 1) * cell_w, (r - 1) * cell_h))
    return grid_img


def _export_png(rows: int, cols: int) -> bytes:
    from PIL import Image
    import io

    img = _build_grid_image(rows, cols)
    buf = io.BytesIO()
    img.save(buf, format="PNG", dpi=(150, 150))
    return buf.getvalue()


def _export_pdf(rows: int, cols: int) -> bytes:
    from PIL import Image
    import io
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate, Image as RLImage, Spacer, Table, TableStyle,
        Paragraph, PageBreak,
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER

    buf = io.BytesIO()
    page_w, page_h = landscape(A4)
    doc = SimpleDocTemplate(
        buf,
        pagesize=landscape(A4),
        leftMargin=15 * mm, rightMargin=15 * mm,
        topMargin=15 * mm, bottomMargin=15 * mm,
    )

    title_style = ParagraphStyle(
        "title", fontName="Helvetica-Bold", alignment=TA_CENTER,
        textColor=colors.HexColor("#1a5276"), fontSize=16,
        leading=20, spaceBefore=0, spaceAfter=4,
    )
    heading_style = ParagraphStyle(
        "heading", fontName="Helvetica-Bold",
        textColor=colors.HexColor("#1a5276"), fontSize=12,
        leading=16, spaceBefore=8, spaceAfter=4,
    )

    story = []

    # Page 1: plot grid — fills entire page (no title)
    grid_img = _build_grid_image(rows, cols, cell_w=700, cell_h=520)
    img_buf = io.BytesIO()
    grid_img.save(img_buf, format="PNG")
    img_buf.seek(0)

    avail_w = page_w - 30 * mm
    avail_h = page_h - 30 * mm - 16  # reportlab internal frame is ~12pt shorter than (page_h - margins)
    scale = min(avail_w / grid_img.width, avail_h / grid_img.height)
    rl_img = RLImage(img_buf, width=grid_img.width * scale, height=grid_img.height * scale)
    story.append(rl_img)

    # Metadata — one page per file
    fcs_data = st.session_state.get("fcs_data", {})
    if fcs_data:
        story.append(PageBreak())
        story.append(Paragraph("FCS File Metadata", title_style))
        story.append(Spacer(1, 4 * mm))

        def _get(meta, key, fallback="—"):
            val = meta.get(key)
            return str(val).strip() if val is not None and str(val).strip() else fallback

        _COL_LABELS = {
            "$PnN": "Name", "$PnS": "Label / Stain", "$PnV": "Voltage",
            "$PnG": "Gain", "$PnR": "Range", "$PnB": "Bits",
            "$PnE": "Amplification", "$PnT": "Detector",
        }

        table_style = TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a5276")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#eaf3fb"), colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#aaaaaa")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ])

        file_list = list(fcs_data.items())
        for file_idx, (filename, info) in enumerate(file_list):
            meta = info.get("meta", {})
            n_events = len(info["data"])
            n_channels = len(info["channels"])

            story.append(Paragraph(filename, heading_style))

            # Summary table
            raw_summary = {
                "Instrument ($CYT)": _get(meta, "$CYT"),
                "Date ($DATE)": _get(meta, "$DATE"),
                "Total events": _get(meta, "$TOT", str(n_events)),
                "Parameters": _get(meta, "$PAR", str(n_channels)),
                "Sample ($SRC)": _get(meta, "$SRC"),
            }
            filtered = {k: v for k, v in raw_summary.items() if v != "—"}
            if filtered:
                sum_data = [["Field", "Value"]] + [[k, v] for k, v in filtered.items()]
                sum_table = Table(sum_data, colWidths=[60 * mm, 80 * mm])
                sum_table.setStyle(table_style)
                story.append(sum_table)
                story.append(Spacer(1, 3 * mm))

            # Channel parameters table — keep only columns with known labels
            import pandas as pd
            _known = set(_COL_LABELS.values())
            channels_df = meta.get("_channels_")
            if channels_df is not None and isinstance(channels_df, pd.DataFrame) and not channels_df.empty:
                df = channels_df.copy().rename(columns={k: v for k, v in _COL_LABELS.items() if k in channels_df.columns})
                df = df[[c for c in df.columns if c in _known]]
                df = df.fillna("—")
                df = df.loc[:, (df != "—").any(axis=0)]
            else:
                try:
                    n_par = int(str(meta.get("$PAR", n_channels)).strip())
                except (ValueError, TypeError):
                    n_par = n_channels
                rows_data = [{"#": i, **{v: _get(meta, f"$P{i}{k[2:]}") for k, v in _COL_LABELS.items()}}
                              for i in range(1, n_par + 1)]
                df = pd.DataFrame(rows_data).set_index("#") if rows_data else pd.DataFrame()
                if not df.empty:
                    df = df.loc[:, (df != "—").any(axis=0)]

            if not df.empty:
                ch_headers = ["#"] + list(df.columns)
                avail_col_w = (page_w - 30 * mm) / len(ch_headers)
                ch_data = [ch_headers] + [[str(i + 1)] + [str(v) for v in row] for i, row in enumerate(df.values)]
                ch_table = Table(ch_data, colWidths=[avail_col_w] * len(ch_headers))
                ch_table.setStyle(table_style)
                story.append(ch_table)

            if file_idx < len(file_list) - 1:
                story.append(PageBreak())

    doc.build(story)
    buf.seek(0)
    return buf.read()


def subplot_label(r: int, c: int) -> str:
    cfg = st.session_state.subplot_config.get((r, c), {})

    if not cfg.get("configured", False):
        return "Click to configure this plot"

    file = cfg.get("file")
    if file:
        _lbl = st.session_state.file_labels.get(file, os.path.basename(file))
        _gate_id = cfg.get("gate_id")
        if _gate_id and file in st.session_state.gate_trees and _gate_id in st.session_state.gate_trees[file]._gates:
            _gtree = st.session_state.gate_trees[file]
            _gate = _gtree.get_gate(_gate_id)
            _path = " > ".join([a.name for a in _gtree.get_ancestors(_gate_id)] + [_gate.name])
            return f"{_lbl} > {_path}"
        return _lbl

    return "Demo (random data)"


def _set_dialog_coords(r: int, c: int):
    st.session_state.dialog_coords = (r, c)


# --- Grid ---
_active_tool = st.session_state.active_gate_tool
_drawing_rc = st.session_state.drawing_subplot
clicked_subplot = None

for r in range(1, rows + 1):
    grid_cols = st.columns(cols)
    for c_idx, col_widget in enumerate(grid_cols):
        c = c_idx + 1
        with col_widget:
            is_drawing = _drawing_rc == (r, c)
            can_draw = bool(_active_tool) and _is_drawable_subplot(r, c)

            if is_drawing:
                # Cancel button replaces title while drawing
                if st.button(
                    f"✕ Cancel | {subplot_label(r, c)}",
                    key=f"title_{r}_{c}",
                    use_container_width=True,
                    type="primary",
                ):
                    _cancel_drawing()
                    st.rerun()
            elif can_draw:
                # Tool active: clicking enters drawing mode
                if st.button(
                    f"[{_active_tool}] {subplot_label(r, c)}",
                    key=f"title_{r}_{c}",
                    use_container_width=True,
                    type="secondary",
                ):
                    st.session_state.drawing_subplot = (r, c)
                    st.rerun()
            else:
                # Normal: open configure dialog
                if st.button(
                    subplot_label(r, c),
                    key=f"title_{r}_{c}",
                    use_container_width=True,
                    type="tertiary",
                    on_click=_set_dialog_coords,
                    args=(r, c),
                ):
                    clicked_subplot = (r, c)

            if is_drawing:
                _render_drawing_chart(r, c, _active_tool)
            else:
                st.plotly_chart(
                    make_plot_fig(r, c),
                    use_container_width=True,
                    key=f"chart_{r}_{c}",
                )

if clicked_subplot:
    plot_dialog(*clicked_subplot)

@st.dialog("Name this gate")
def gate_name_dialog():
    pg = st.session_state.pending_gate
    gate = pg["gate_obj"]
    file = pg["file"]

    name = st.text_input("Gate name", value="Gate")

    saved_preset = "Red"
    preset_options = list(PRESET_COLORS.keys()) + ["Custom"]
    color_choice = st.radio(
        "Color", preset_options,
        index=preset_options.index(saved_preset),
        horizontal=True,
        key="_gname_color_radio",
    )
    if color_choice == "Custom":
        chosen_color = st.color_picker("Color", value="#e74c3c", label_visibility="collapsed")
    else:
        chosen_color = PRESET_COLORS[color_choice]

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Save gate", type="primary", use_container_width=True):
            gate.name = name.strip() or "Gate"
            gate.color = chosen_color
            if file and file in st.session_state.gate_trees:
                _parent_id = pg.get("parent_id") or GateTree.ROOT_ID
                _gid = st.session_state.gate_trees[file].add_gate(gate, _parent_id)
                _src_rc = pg.get("subplot_rc")
                if _src_rc:
                    st.session_state.subplot_gates.setdefault(_src_rc, set()).add(_gid)
            st.session_state.pending_gate = None
            st.rerun()
    with col2:
        if st.button("Cancel", use_container_width=True):
            st.session_state.pending_gate = None
            st.rerun()


if st.session_state.pending_gate:
    gate_name_dialog()

# --- Gate statistics table ---
def _build_gate_table_rows():
    rows = []
    for fname, tree in st.session_state.gate_trees.items():
        if fname not in st.session_state.fcs_data or len(tree) == 0:
            continue
        lbl = st.session_state.file_labels.get(fname, os.path.basename(fname))
        fdata = st.session_state.fcs_data[fname]["data"]
        total = len(fdata)
        for _, gate in tree.flat_list():
            ancestors = tree.get_ancestors(gate.id)
            path = [lbl] + [a.name for a in ancestors] + [gate.name]
            n = tree.event_count(gate.id, fdata)
            pct_p = tree.percent_of_parent(gate.id, fdata)
            pct_t = n / total * 100 if total > 0 else 0.0
            rows.append({
                "path": path,
                "gate": gate.name,
                "n": n,
                "pct_parent": round(pct_p, 1),
                "pct_total": round(pct_t, 1),
            })
    return rows

_gate_rows = _build_gate_table_rows()
if _gate_rows:
    st.divider()
    st.subheader("Gate statistics")
    try:
        import pandas as pd
        from st_aggrid import AgGrid, GridOptionsBuilder, JsCode

        _df_gates = pd.DataFrame(_gate_rows)
        _gb = GridOptionsBuilder.from_dataframe(_df_gates.drop(columns=["path"]))
        _gb.configure_column("gate", hide=True)
        _gb.configure_column("n", header_name="Count", type=["numericColumn"],
                              valueFormatter=JsCode("function(p){return p.value==null?'':p.value.toLocaleString();}"))
        _gb.configure_column("pct_parent", header_name="% of parent", type=["numericColumn"],
                              valueFormatter=JsCode("function(p){return p.value==null?'':p.value.toFixed(1)+'%';}"))
        _gb.configure_column("pct_total", header_name="% of total", type=["numericColumn"],
                              valueFormatter=JsCode("function(p){return p.value==null?'':p.value.toFixed(1)+'%';}"))
        _go = _gb.build()
        _go["treeData"] = True
        _go["animateRows"] = True
        _go["getDataPath"] = JsCode("function(data){return data.path;}")
        _go["autoGroupColumnDef"] = {
            "headerName": "Gate",
            "minWidth": 200,
            "cellRendererParams": {"suppressCount": True},
        }
        AgGrid(_df_gates, gridOptions=_go, height=300,
               fit_columns_on_grid_load=True, allow_unsafe_jscode=True)
    except Exception as _e:
        st.caption(f"Gate stats table unavailable: {_e}")

# Export lives in a second sidebar block so _export_png/_export_pdf are already defined
with st.sidebar:
    st.divider()
    st.write("**Export**")
    _ex1, _ex2 = st.columns(2)
    with _ex1:
        if st.button("PNG", type="primary", use_container_width=True):
            _er, _ec = LAYOUTS[layout_choice]
            with st.spinner("Rendering…"):
                try:
                    st.session_state._export_data = _export_png(_er, _ec)
                    st.session_state._export_fmt = "png"
                    st.session_state._export_mime = "image/png"
                    st.session_state._export_ready = True
                except Exception as _ex:
                    st.error(f"PNG failed: {_ex}")
    with _ex2:
        if st.button("PDF", type="primary", use_container_width=True):
            _er, _ec = LAYOUTS[layout_choice]
            with st.spinner("Rendering…"):
                try:
                    st.session_state._export_data = _export_pdf(_er, _ec)
                    st.session_state._export_fmt = "pdf"
                    st.session_state._export_mime = "application/pdf"
                    st.session_state._export_ready = True
                except Exception as _ex:
                    st.error(f"PDF failed: {_ex}")
    if st.session_state.get("_export_ready"):
        st.download_button(
            label=f"Download {st.session_state._export_fmt.upper()}",
            data=st.session_state._export_data,
            file_name=f"fscviz_export.{st.session_state._export_fmt}",
            mime=st.session_state._export_mime,
            use_container_width=True,
        )

