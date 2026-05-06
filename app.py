import os
import tempfile

import fcsparser
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="FCS Viewer", layout="wide")
st.title("FCS Viewer")

with st.expander("How to use", expanded=False):
    st.markdown(
        "- **Upload** one or more FCS files using the sidebar panel on the left\n"
        "- **Click a plot title** to open the configuration dialog for that panel\n"
        "- **Select** a data source, plot type (Scatter / Histogram), channels, and color, then click **Apply**\n"
        "- Click the title again at any time to reconfigure or switch channels"
    )

LAYOUTS = {
    "2 × 2": (2, 2),
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

# --- Session state init ---
if "fcs_data" not in st.session_state:
    # {filename: {"data": DataFrame, "channels": [str, ...]}}
    st.session_state.fcs_data = {}

if "subplot_config" not in st.session_state:
    # {(r, c): {"file": str|None, "plot_type": str, "x_ch": str|None, "y_ch": str|None, "n_bins": int}}
    st.session_state.subplot_config = {}

layout_choice = st.selectbox(
    "Grid layout",
    list(LAYOUTS.keys()),
    label_visibility="collapsed",
)

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
                }
            except Exception as e:
                st.error(f"Failed to parse {uf.name}: {e}")

    # Remove files that are no longer in the uploader
    if uploaded_files is not None:
        current_names = {uf.name for uf in uploaded_files}
        for name in list(st.session_state.fcs_data):
            if name not in current_names:
                del st.session_state.fcs_data[name]
                for cfg in st.session_state.subplot_config.values():
                    if cfg.get("file") == name:
                        cfg.update({"configured": False, "file": None, "x_ch": None, "y_ch": None})

    # Show summary of loaded files
    if st.session_state.fcs_data:
        st.write("**Loaded files:**")
        for name, info in st.session_state.fcs_data.items():
            n_events = len(info["data"])
            n_ch = len(info["channels"])
            st.caption(f"{name}  \n{n_events:,} events · {n_ch} channels")

rows, cols = LAYOUTS[layout_choice]

# Reset seeds when layout changes
if st.session_state.get("last_layout") != layout_choice:
    st.session_state.seeds = {}
    st.session_state.last_layout = layout_choice

for r in range(1, rows + 1):
    for c in range(1, cols + 1):
        st.session_state.seeds.setdefault((r, c), (r - 1) * cols + (c - 1))
        st.session_state.subplot_config.setdefault(
            (r, c), {"configured": False, "file": None, "plot_type": "Scatter", "x_ch": None, "y_ch": None, "n_bins": 256, "color": "#1f77b4"}
        )


@st.dialog("Configure Plot")
def plot_dialog(r: int, c: int):
    cfg = st.session_state.subplot_config[(r, c)]
    fcs_data = st.session_state.fcs_data

    st.subheader(f"Subplot ({r}, {c})")
    st.divider()

    # --- File selector ---
    file_options = [_NO_FILE] + list(fcs_data.keys())
    saved_file = cfg.get("file")
    file_idx = file_options.index(saved_file) if saved_file in file_options else 0
    selected_label = st.selectbox("Data source", file_options, index=file_idx)
    selected_file = None if selected_label == _NO_FILE else selected_label

    # --- Plot type ---
    plot_type = st.radio(
        "Plot type",
        ["Scatter", "Histogram"],
        index=0 if cfg.get("plot_type", "Scatter") == "Scatter" else 1,
        horizontal=True,
    )

    # --- Channel selectors ---
    x_ch = y_ch = None
    if selected_file:
        channels = fcs_data[selected_file]["channels"]
        file_unchanged = cfg.get("file") == selected_file
        saved_x = cfg.get("x_ch") if file_unchanged else None
        saved_y = cfg.get("y_ch") if file_unchanged else None
        x_idx = channels.index(saved_x) if saved_x in channels else 0
        y_idx = channels.index(saved_y) if saved_y in channels else min(1, len(channels) - 1)

        if plot_type == "Scatter":
            col_x, col_y = st.columns(2)
            with col_x:
                x_ch = st.selectbox("X axis", channels, index=x_idx)
            with col_y:
                y_ch = st.selectbox("Y axis", channels, index=y_idx)
        else:
            x_ch = st.selectbox("Channel", channels, index=x_idx)
    else:
        st.caption("No file selected — subplot will show random data.")

    # --- Bin count (histogram only) ---
    n_bins = cfg.get("n_bins", 256)
    if plot_type == "Histogram":
        n_bins = st.number_input("Bins", min_value=10, max_value=1024, value=n_bins, step=10)

    # --- Color ---
    saved_color = cfg.get("color", "#1f77b4")
    saved_preset = next((k for k, v in PRESET_COLORS.items() if v == saved_color), "Custom")
    preset_options = list(PRESET_COLORS.keys()) + ["Custom"]

    # Inject a unique marker span so CSS can scope to only this radio widget
    # (sibling selector targets the color radio but not the plot-type radio above it)
    _mid = f"cpick-{r}-{c}"
    st.markdown(f'<span id="{_mid}"></span>', unsafe_allow_html=True)

    _rules = []
    for _i, (_, _hex) in enumerate(PRESET_COLORS.items()):
        _n = _i + 1
        _rules.append(
            f"div:has(#{_mid})~[data-testid='stRadio'] label:nth-child({_n})"
            f"{{background:{_hex}!important;border-radius:5px!important;"
            f"min-width:32px!important;height:26px!important;padding:0 6px!important;}}"
            f"div:has(#{_mid})~[data-testid='stRadio'] label:nth-child({_n}) p"
            f"{{display:none!important;}}"
            f"div:has(#{_mid})~[data-testid='stRadio'] label:nth-child({_n})>div:first-child"
            f"{{display:none!important;}}"
        )
    # Selection ring via native CSS checked state — no Python rerun needed
    _rules.append(
        f"div:has(#{_mid})~[data-testid='stRadio'] label:has(input:checked)"
        f"{{box-shadow:0 0 0 3px #222!important;}}"
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

    st.divider()
    btn1, btn2 = st.columns(2)
    with btn1:
        if st.button("Apply", use_container_width=True, type="primary"):
            st.session_state.subplot_config[(r, c)] = {
                "configured": True,
                "file": selected_file,
                "plot_type": plot_type,
                "x_ch": x_ch,
                "y_ch": y_ch if plot_type == "Scatter" else None,
                "n_bins": n_bins,
                "color": final_color,
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

    has_real_data = file and file in fcs_data and x_ch
    has_scatter_data = has_real_data and plot_type == "Scatter" and y_ch

    seed = st.session_state.seeds[(r, c)]
    rng = np.random.default_rng(seed=seed)

    if plot_type == "Histogram":
        if has_real_data:
            x = fcs_data[file]["data"][x_ch].values
            x_label = x_ch
        else:
            n1, n2 = int(N_CELLS * 0.6), int(N_CELLS * 0.4)
            x = np.concatenate([rng.normal(3.5, 0.8, n1), rng.normal(6.0, 0.6, n2)])
            x_label = "X"

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

    else:  # Scatter
        if has_scatter_data:
            df = fcs_data[file]["data"]
            x = df[x_ch].values
            y = df[y_ch].values
            x_label, y_label = x_ch, y_ch
        else:
            n1, n2 = int(N_CELLS * 0.6), int(N_CELLS * 0.4)
            x = np.concatenate([rng.normal(3.5, 0.8, n1), rng.normal(6.0, 0.6, n2)])
            y = np.concatenate([rng.normal(4.0, 0.7, n1), rng.normal(7.0, 0.5, n2)])
            x_label, y_label = "X", "Y"

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

    return fig


def subplot_label(r: int, c: int) -> str:
    cfg = st.session_state.subplot_config.get((r, c), {})

    if not cfg.get("configured", False):
        return "Click to configure this plot"

    file = cfg.get("file")
    if file:
        stem = os.path.splitext(file)[0]
        truncated = stem[:20] + "..." if len(stem) > 20 else stem
        n_events = len(st.session_state.fcs_data[file]["data"])
        return f"{truncated} (n = {n_events:,})"

    return "Demo (random data)"


# --- Grid ---
clicked_subplot = None

for r in range(1, rows + 1):
    grid_cols = st.columns(cols)
    for c_idx, col_widget in enumerate(grid_cols):
        c = c_idx + 1
        with col_widget:
            if st.button(subplot_label(r, c), key=f"title_{r}_{c}", use_container_width=True):
                clicked_subplot = (r, c)
            st.plotly_chart(
                make_plot_fig(r, c),
                use_container_width=True,
                key=f"chart_{r}_{c}",
            )

if clicked_subplot:
    plot_dialog(*clicked_subplot)
