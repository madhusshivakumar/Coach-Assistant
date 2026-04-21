"""Match Overview page — Phase 1 primary dashboard.

Pick an ingested match, see shot map, per-team pass networks, team-metric table,
top xT contributors, and the xT surface learned from this match.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import streamlit as st

from football_analysis.analytics.pipeline.runner import run as run_pipeline
from football_analysis.app.data import list_ingested_matches, load_match_events, team_names_for_match
from football_analysis.viz.static.heatmap import plot_player_heatmap
from football_analysis.viz.static.pass_network import plot_pass_network
from football_analysis.viz.static.shot_map import plot_shot_map
from football_analysis.viz.static.xt_surface import plot_xt_surface

st.set_page_config(page_title="Match Overview", layout="wide")
st.title("Match Overview")

matches = list_ingested_matches()
if matches.empty:
    st.warning("No matches ingested yet. Run `fa-data fetch` + `fa-data ingest` first.")
    st.stop()

labels = matches["label"].tolist()
match_ids = matches["match_id"].tolist()
choice = st.selectbox("Match", options=range(len(labels)), format_func=lambda i: labels[i])
selected = match_ids[choice]
st.caption(f"`{selected}`")

events = load_match_events(selected)
names = team_names_for_match(selected)
analytics = run_pipeline(events)
enriched = analytics.events
team_ids = sorted(enriched["team_id"].dropna().unique())


# ---------- Team summary table ----------
summary_rows = []
for tid in team_ids:
    summary_rows.append(
        {
            "team": names.get(tid, tid),
            "passes": int(((enriched["team_id"] == tid) & (enriched["action_type"] == "pass")).sum()),
            "shots": int(((enriched["team_id"] == tid) & (enriched["action_type"] == "shot")).sum()),
            "goals": int(
                (
                    (enriched["team_id"] == tid)
                    & (enriched["action_type"] == "shot")
                    & (enriched["result"] == "success")
                ).sum()
            ),
            "PPDA (lower = press harder)": round(analytics.ppda.get(tid, 0.0), 2),
            "Field tilt (final-third share)": f"{analytics.field_tilt.get(tid, 0.0):.1%}",
            "Total xT gained": round(
                float(enriched.loc[enriched["team_id"] == tid, "xt_delta"].sum(skipna=True)), 3
            ),
        }
    )

st.subheader("Team summary")
st.dataframe(summary_rows, hide_index=True, use_container_width=True)


# ---------- Shot map ----------
st.subheader("Shot map")
fig = plot_shot_map(enriched, title=f"Shots — {labels[choice]}")
st.pyplot(fig, use_container_width=True)


# ---------- Pass networks side-by-side ----------
st.subheader("Pass networks")
cols = st.columns(len(team_ids))
for col, tid in zip(cols, team_ids, strict=True):
    with col:
        st.caption(names.get(tid, tid))
        fig = plot_pass_network(enriched, team_id=tid, min_passes_edge=5)
        st.pyplot(fig, use_container_width=True)


# ---------- xT surface ----------
st.subheader("xT surface (learned from this match)")
fig = plot_xt_surface(analytics.xt_grid)
st.pyplot(fig, use_container_width=True)


# ---------- Top xT contributors ----------
st.subheader("Top xT contributors")
pos_moves = enriched[enriched["xt_delta"].notna() & (enriched["xt_delta"] > 0)]
if pos_moves.empty:
    st.info("No xT-positive moves in this match.")
else:
    ranked = (
        pos_moves.groupby("player_id")["xt_delta"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "xt_total", "count": "actions"})
        .sort_values("xt_total", ascending=False)
        .head(10)
        .reset_index()
    )
    ranked["xt_total"] = ranked["xt_total"].round(3)
    st.dataframe(ranked, hide_index=True, use_container_width=True)

    # Heatmap for top player
    top_player = str(ranked.iloc[0]["player_id"])
    st.subheader(f"Top contributor heatmap — player {top_player}")
    fig = plot_player_heatmap(enriched, player_id=top_player)
    st.pyplot(fig, use_container_width=True)
