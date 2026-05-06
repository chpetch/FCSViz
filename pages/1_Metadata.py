import pandas as pd
import streamlit as st

st.set_page_config(page_title="FSCViz – Metadata", layout="wide")
st.title("FCS File Metadata")

fcs_data = st.session_state.get("fcs_data", {})

if not fcs_data:
    st.info("No FCS files loaded. Upload files on the main page.")
    st.stop()


def _get(meta: dict, key: str, fallback: str = "—") -> str:
    val = meta.get(key)
    return str(val).strip() if val is not None and str(val).strip() else fallback


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
        try:
            n_par = int(str(meta.get("$PAR", n_channels)).strip())
        except (ValueError, TypeError):
            n_par = n_channels

        rows = [
            {
                "#":       i,
                "Name":    _get(meta, f"$P{i}N"),
                "Label":   _get(meta, f"$P{i}S"),
                "Voltage": _get(meta, f"$P{i}V"),
                "Gain":    _get(meta, f"$P{i}G"),
                "Range":   _get(meta, f"$P{i}R"),
            }
            for i in range(1, n_par + 1)
        ]

        if rows:
            df = pd.DataFrame(rows).set_index("#")
            df = df.loc[:, (df != "—").any(axis=0)]
            st.dataframe(df, use_container_width=True)
        else:
            st.caption("No per-channel metadata found.")
