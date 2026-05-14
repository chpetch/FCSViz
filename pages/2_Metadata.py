from _shared import init_session_state, render_upload_sidebar

import pandas as pd
import streamlit as st

st.set_page_config(page_title="FCSViz – Metadata", layout="wide")
init_session_state()

with st.sidebar:
    render_upload_sidebar()

st.title("FCS File Metadata")

fcs_data = st.session_state.get("fcs_data", {})

if not fcs_data:
    st.info("No FCS files loaded. Upload files using the sidebar.")
    st.stop()


def _get(meta: dict, key: str, fallback: str = "—") -> str:
    val = meta.get(key)
    return str(val).strip() if val is not None and str(val).strip() else fallback


_COL_LABELS = {
    "$PnN": "Name",
    "$PnS": "Label / Stain",
    "$PnV": "Voltage",
    "$PnG": "Gain",
    "$PnR": "Range",
    "$PnB": "Bits",
    "$PnE": "Amplification",
    "$PnT": "Detector",
    "$PnD": "Display",
}


def _channel_table(meta: dict, n_channels: int) -> pd.DataFrame:
    """Build per-channel parameter table from meta dict."""
    # fcsparser reformat_meta=True stores a _channels_ DataFrame — use it when available
    channels_df = meta.get("_channels_")
    if channels_df is not None and isinstance(channels_df, pd.DataFrame) and not channels_df.empty:
        df = channels_df.copy()
        df = df.rename(columns={k: v for k, v in _COL_LABELS.items() if k in df.columns})
        df = df.fillna("—")
        df = df.loc[:, (df != "—").any(axis=0)]
        df.index = range(1, len(df) + 1)
        df.index.name = "#"
        return df

    # Fallback: build from individual $PnX keys
    try:
        n_par = int(str(meta.get("$PAR", n_channels)).strip())
    except (ValueError, TypeError):
        n_par = n_channels

    rows = [
        {
            "#":                i,
            "Name":             _get(meta, f"$P{i}N"),
            "Label / Stain":    _get(meta, f"$P{i}S"),
            "Voltage":          _get(meta, f"$P{i}V"),
            "Gain":             _get(meta, f"$P{i}G"),
            "Range":            _get(meta, f"$P{i}R"),
            "Bits":             _get(meta, f"$P{i}B"),
            "Amplification":    _get(meta, f"$P{i}E"),
            "Detector":         _get(meta, f"$P{i}T"),
        }
        for i in range(1, n_par + 1)
    ]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index("#")
    df = df.loc[:, (df != "—").any(axis=0)]
    return df


for filename, info in fcs_data.items():
    meta = info.get("meta", {})
    n_events = len(info["data"])
    n_channels = len(info["channels"])

    with st.expander(filename, expanded=True):
        st.subheader("Summary")
        summary = {k: v for k, v in {
            "Instrument ($CYT)": _get(meta, "$CYT"),
            "Date ($DATE)":      _get(meta, "$DATE"),
            "Total events":      _get(meta, "$TOT", str(n_events)),
            "Parameters":        _get(meta, "$PAR", str(n_channels)),
            "Sample ($SRC)":     _get(meta, "$SRC"),
            "Tube name":         _get(meta, "TUBE NAME"),
        }.items() if v != "—"}
        st.table({"Field": list(summary.keys()), "Value": list(summary.values())})

        st.subheader("Channel Parameters")
        df = _channel_table(meta, n_channels)
        if not df.empty:
            st.dataframe(df, use_container_width=True)
        else:
            st.caption("No per-channel metadata found.")
