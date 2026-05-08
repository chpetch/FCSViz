import os
import tempfile

import fcsparser
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="FSCViz", layout="wide")
st.title("FSCViz")

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


# --- Session state init ---
if "fcs_data" not in st.session_state:
    # {filename: {"data": DataFrame, "channels": [str, ...]}}
    st.session_state.fcs_data = {}

if "subplot_config" not in st.session_state:
    # {(r, c): {"file": str|None, "plot_type": str, "x_ch": str|None, "y_ch": str|None, "n_bins": int}}
    st.session_state.subplot_config = {}

if "dialog_coords" not in st.session_state:
    st.session_state.dialog_coords = (1, 1)

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
            col_label.caption(f"{name}  \n{n_events:,} events · {n_ch} ch")
            if col_btn.button("×", key=f"_rm_{name}", help=f"Remove {name}"):
                del st.session_state.fcs_data[name]
                for cfg in st.session_state.subplot_config.values():
                    if cfg.get("file") == name:
                        cfg.update({"configured": False, "file": None, "x_ch": None, "y_ch": None})
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
            (r, c), {"configured": False, "file": None, "plot_type": "Scatter", "x_ch": None, "y_ch": None, "n_bins": 256, "color": "#1f77b4", "x_transform": "Linear", "y_transform": "Linear", "x_cofactor": 150, "y_cofactor": 150}
        )


_dr, _dc = st.session_state.dialog_coords


@st.dialog(f"Configure subplot ({_dr}, {_dc})")
def plot_dialog(r: int, c: int):
    cfg = st.session_state.subplot_config[(r, c)]
    fcs_data = st.session_state.fcs_data

    # --- File selector ---
    file_options = [_NO_FILE] + list(fcs_data.keys())
    saved_file = cfg.get("file")
    file_idx = file_options.index(saved_file) if saved_file in file_options else 0
    selected_label = st.selectbox("Data source", file_options, index=file_idx)
    selected_file = None if selected_label == _NO_FILE else selected_label

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
            st.session_state.subplot_config[(r, c)] = {
                "configured": True,
                "file": selected_file,
                "plot_type": plot_type,
                "x_ch": x_ch,
                "y_ch": y_ch if plot_type in ("Scatter", "Density") else None,
                "n_bins": n_bins,
                "color": final_color,
                "x_transform": x_transform,
                "y_transform": y_transform,
                "x_cofactor": x_cofactor,
                "y_cofactor": y_cofactor,
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

    has_real_data = file and file in fcs_data and x_ch
    has_scatter_data = has_real_data and plot_type == "Scatter" and y_ch

    seed = st.session_state.seeds[(r, c)]
    rng = np.random.default_rng(seed=seed)

    if plot_type == "Histogram":
        if has_real_data:
            x = apply_transform(fcs_data[file]["data"][x_ch].values, x_transform)
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
            df = fcs_data[file]["data"]
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
            df = fcs_data[file]["data"]
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

    return fig


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
        stem = os.path.splitext(file)[0]
        truncated = stem[:25] + "..." if len(stem) > 25 else stem
        n_events = len(st.session_state.fcs_data[file]["data"])
        return f"{truncated} (n = {n_events:,})"

    return "Demo (random data)"


def _set_dialog_coords(r: int, c: int):
    st.session_state.dialog_coords = (r, c)


# --- Grid ---
clicked_subplot = None

for r in range(1, rows + 1):
    grid_cols = st.columns(cols)
    for c_idx, col_widget in enumerate(grid_cols):
        c = c_idx + 1
        with col_widget:
            if st.button(
                subplot_label(r, c),
                key=f"title_{r}_{c}",
                use_container_width=True,
                type="tertiary",
                on_click=_set_dialog_coords,
                args=(r, c),
            ):
                clicked_subplot = (r, c)
            st.plotly_chart(
                make_plot_fig(r, c),
                use_container_width=True,
                key=f"chart_{r}_{c}",
            )

if clicked_subplot:
    plot_dialog(*clicked_subplot)

# --- Export ---
st.divider()
st.subheader("Export")

exp_col1, exp_col2, exp_col3 = st.columns([2, 2, 2])

with exp_col1:
    if st.button("Export PNG", type="primary", use_container_width=True):
        with st.spinner("Rendering subplots…"):
            try:
                st.session_state._export_data = _export_png(rows, cols)
                st.session_state._export_fmt = "png"
                st.session_state._export_mime = "image/png"
                st.session_state._export_ready = True
            except Exception as e:
                st.error(f"PNG export failed: {e}")

with exp_col2:
    if st.button("Export PDF", type="primary", use_container_width=True):
        with st.spinner("Rendering subplots…"):
            try:
                st.session_state._export_data = _export_pdf(rows, cols)
                st.session_state._export_fmt = "pdf"
                st.session_state._export_mime = "application/pdf"
                st.session_state._export_ready = True
            except Exception as e:
                st.error(f"PDF export failed: {e}")

if st.session_state.get("_export_ready"):
    with exp_col3:
        st.download_button(
            label=f"Download {st.session_state._export_fmt.upper()}",
            data=st.session_state._export_data,
            file_name=f"fscviz_export.{st.session_state._export_fmt}",
            mime=st.session_state._export_mime,
        )
