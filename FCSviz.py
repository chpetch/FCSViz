from _shared import init_session_state, render_upload_sidebar

import streamlit as st

st.set_page_config(page_title="FCSViz", layout="wide")
init_session_state()

st.image("pics/FCSViz Banner.svg", use_container_width=True)

with st.sidebar:
    render_upload_sidebar()

# ---------------------------------------------------------------------------
# Landing page content
# ---------------------------------------------------------------------------
st.markdown(
    """
## What is FCSViz?

**FCSViz** is a browser-based viewer for Flow Cytometry Standard (FCS) files.
It lets you explore multi-channel cytometry data interactively — without installing
any desktop software — through a flexible grid of scatter, density, and histogram plots.

---

### Getting started

1. **Upload** one or more `.fcs` files using the sidebar on the left.
2. Go to the **Plot** page and click any panel title to configure it.
3. Choose a data source, plot type, channels, and transform, then click **Apply**.
4. Use the **Gate tools** toolbar to draw gates on your plots.
5. Export your grid as PNG or PDF from the sidebar.

---

### Supported file format

FCS 2.0, 3.0, and 3.1 files (`.fcs`). Parsed with
[fcsparser](https://github.com/eyurtsev/fcsparser).
All numeric channels are available for plotting.
"""
)
