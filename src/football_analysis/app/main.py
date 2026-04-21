"""Streamlit entrypoint. Run with `make app` or `streamlit run src/football_analysis/app/main.py`."""

import streamlit as st

from football_analysis import __version__
from football_analysis.app.data import list_ingested_matches
from football_analysis.logging import configure_logging

configure_logging()

st.set_page_config(page_title="football-analysis", layout="wide")

st.title("football-analysis")
st.caption(f"v{__version__}")

matches = list_ingested_matches()
st.metric("Ingested matches", len(matches))
if matches.empty:
    st.info("No matches ingested yet. Run `fa-data fetch` + `fa-data ingest` first.")
    st.caption("Then open **Match Overview** from the sidebar.")
else:
    st.success("Open **Match Overview** from the sidebar to explore an ingested match.")
    st.dataframe(matches[["label", "match_id", "ingested_at"]], hide_index=True)
