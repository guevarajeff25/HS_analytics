import streamlit as st
import pandas as pd
from utils import (
    load_dataset,
    compute_hitting_metrics,
    compute_pitching_metrics,
    compute_fielding_metrics,
    top_table,
    lineup_suggestions,
)

st.set_page_config(page_title="HS Baseball Analytics", layout="wide")

st.title("HS Baseball Analytics — Starter App")

with st.sidebar:
    st.header("Mode")
    mode = st.radio("Choose mode", ["Demo MLB (built-in)", "Upload GameChanger CSVs"], index=0)

    st.divider()
    st.header("Navigation")
    page = st.radio("Go to", ["Team Overview", "Player Cards", "Lineup Builder", "Exports"], index=0)

    st.divider()
    st.caption("v1: Upload batting/pitching/fielding CSVs (season totals).")

# ---- Load data ----
if mode == "Demo MLB (built-in)":
    bat, pit, fld = load_dataset(mode="demo")
else:
    st.info("Upload your GameChanger exports (season totals). If one file is missing, the app will still run with what you provide.")
    bat_file = st.file_uploader("Batting CSV", type=["csv"], key="bat")
    pit_file = st.file_uploader("Pitching CSV", type=["csv"], key="pit")
    fld_file = st.file_uploader("Fielding CSV", type=["csv"], key="fld")
    bat, pit, fld = load_dataset(mode="upload", batting_file=bat_file, pitching_file=pit_file, fielding_file=fld_file)

# Compute metrics
bat_m = compute_hitting_metrics(bat) if bat is not None else None
pit_m = compute_pitching_metrics(pit) if pit is not None else None
fld_m = compute_fielding_metrics(fld) if fld is not None else None

# ---- Pages ----
if page == "Team Overview":
    st.subheader("Team Overview")

    cols = st.columns(4)
    if bat_m is not None and len(bat_m):
        team_ops = (bat_m["OPS"].mean())
        team_obp = (bat_m["OBP"].mean())
        team_slg = (bat_m["SLG"].mean())
        team_k = (bat_m["K%"].mean())
        cols[0].metric("Team OPS (avg)", f"{team_ops:.3f}")
        cols[1].metric("Team OBP (avg)", f"{team_obp:.3f}")
        cols[2].metric("Team SLG (avg)", f"{team_slg:.3f}")
        cols[3].metric("Team K% (avg)", f"{team_k:.1%}")
    else:
        st.warning("No batting file loaded.")

    cols2 = st.columns(3)
    if pit_m is not None and len(pit_m):
        team_whip = pit_m["WHIP"].replace([float("inf")], pd.NA).dropna().mean()
        team_kbb = pit_m["K/BB"].replace([float("inf")], pd.NA).dropna().mean()
        team_bb_inn = pit_m["BB/INN"].replace([float("inf")], pd.NA).dropna().mean()
        cols2[0].metric("Staff WHIP (avg)", f"{team_whip:.2f}" if pd.notna(team_whip) else "—")
        cols2[1].metric("Staff K/BB (avg)", f"{team_kbb:.2f}" if pd.notna(team_kbb) else "—")
        cols2[2].metric("Staff BB/INN (avg)", f"{team_bb_inn:.2f}" if pd.notna(team_bb_inn) else "—")
    else:
        cols2[0].metric("Staff WHIP (avg)", "—")
        cols2[1].metric("Staff K/BB (avg)", "—")
        cols2[2].metric("Staff BB/INN (avg)", "—")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.markdown("### Batting leaders")
        if bat_m is not None and len(bat_m):
            tabs = st.tabs(["OPS", "OBP", "SLG", "BB%", "K%", "XBH"])
            metrics = ["OPS", "OBP", "SLG", "BB%", "K%", "XBH"]
            for t, m in zip(tabs, metrics):
                with t:
                    st.dataframe(top_table(bat_m, m, n=10), use_container_width=True)
        else:
            st.info("Load batting to see leaders.")

    with right:
        st.markdown("### Pitching leaders")
        if pit_m is not None and len(pit_m):
            tabs = st.tabs(["WHIP (low)", "K/BB", "K/BF", "BB/INN (low)", "ERA (low)"])
            with tabs[0]:
                st.dataframe(top_table(pit_m, "WHIP", n=10, ascending=True), use_container_width=True)
            with tabs[1]:
                st.dataframe(top_table(pit_m, "K/BB", n=10), use_container_width=True)
            with tabs[2]:
                st.dataframe(top_table(pit_m, "K/BF", n=10), use_container_width=True)
            with tabs[3]:
                st.dataframe(top_table(pit_m, "BB/INN", n=10, ascending=True), use_container_width=True)
            with tabs[4]:
                st.dataframe(top_table(pit_m, "ERA", n=10, ascending=True), use_container_width=True)
        else:
            st.info("Load pitching to see leaders.")

    st.divider()
    st.markdown("### Defense snapshot")
    if fld_m is not None and len(fld_m):
        st.dataframe(top_table(fld_m, "FPCT", n=12), use_container_width=True)
    else:
        st.info("Load fielding to see defense.")

elif page == "Player Cards":
    st.subheader("Player Cards")
    players = set()
    if bat_m is not None and "PLAYER" in bat_m.columns:
        players |= set(bat_m["PLAYER"].dropna().astype(str))
    if pit_m is not None and "PLAYER" in pit_m.columns:
        players |= set(pit_m["PLAYER"].dropna().astype(str))
    if fld_m is not None and "PLAYER" in fld_m.columns:
        players |= set(fld_m["PLAYER"].dropna().astype(str))

    if not players:
        st.warning("No players found. Load at least one CSV.")
    else:
        player = st.selectbox("Select a player", sorted(players))
        c1, c2, c3 = st.columns(3)

        if bat_m is not None and player in set(bat_m["PLAYER"].astype(str)):
            pbat = bat_m[bat_m["PLAYER"].astype(str) == player].iloc[0]
            with c1:
                st.markdown("#### Hitting")
                st.write({
                    "PA": int(pbat.get("PA", 0)),
                    "AVG": round(float(pbat.get("AVG", 0)), 3),
                    "OBP": round(float(pbat.get("OBP", 0)), 3),
                    "SLG": round(float(pbat.get("SLG", 0)), 3),
                    "OPS": round(float(pbat.get("OPS", 0)), 3),
                    "BB%": f"{float(pbat.get('BB%', 0)):.1%}",
                    "K%": f"{float(pbat.get('K%', 0)):.1%}",
                    "XBH": int(pbat.get("XBH", 0)),
                    "SB": int(pbat.get("SB", 0)),
                })
        else:
            with c1:
                st.markdown("#### Hitting")
                st.caption("No batting data for this player.")

        if pit_m is not None and player in set(pit_m["PLAYER"].astype(str)):
            ppit = pit_m[pit_m["PLAYER"].astype(str) == player].iloc[0]
            with c2:
                st.markdown("#### Pitching")
                st.write({
                    "IP": float(ppit.get("IP", 0)),
                    "ERA": float(ppit.get("ERA", 0)),
                    "WHIP": float(ppit.get("WHIP", 0)),
                    "K/BB": float(ppit.get("K/BB", 0)),
                    "BB/INN": float(ppit.get("BB/INN", 0)),
                    "K/BF": float(ppit.get("K/BF", 0)),
                    "HR": int(ppit.get("HR", 0)),
                })
        else:
            with c2:
                st.markdown("#### Pitching")
                st.caption("No pitching data for this player.")

        if fld_m is not None and player in set(fld_m["PLAYER"].astype(str)):
            pfld = fld_m[fld_m["PLAYER"].astype(str) == player].iloc[0]
            with c3:
                st.markdown("#### Fielding")
                st.write({
                    "TC": int(pfld.get("TC", 0)),
                    "PO": int(pfld.get("PO", 0)),
                    "A": int(pfld.get("A", 0)),
                    "E": int(pfld.get("E", 0)),
                    "FPCT": round(float(pfld.get("FPCT", 0)), 3),
                    "DP": int(pfld.get("DP", 0)),
                })
        else:
            with c3:
                st.markdown("#### Fielding")
                st.caption("No fielding data for this player.")

        st.divider()
        st.markdown("### Development notes (rule-based, v1)")
        notes = []
        if bat_m is not None and player in set(bat_m["PLAYER"].astype(str)):
            k = float(pbat.get("K%", 0))
            bb = float(pbat.get("BB%", 0))
            ops = float(pbat.get("OPS", 0))
            if k > 0.22:
                notes.append("Cut swing-and-miss: prioritize 2-strike approach + zone discipline.")
            if bb < 0.06:
                notes.append("On-base upside: improve walk rate by hunting one zone early in counts.")
            if ops < bat_m["OPS"].median():
                notes.append("Find impact: focus on hard contact (line drives) and driving mistakes.")
            if not notes:
                notes.append("Maintain: keep approach stable and look for marginal gains (BB% up, K% down).")
        if pit_m is not None and player in set(pit_m["PLAYER"].astype(str)):
            whip = float(ppit.get("WHIP", 0))
            kbb = float(ppit.get("K/BB", 0))
            if whip > 1.6:
                notes.append("Traffic reduction: tighten free passes and attack early in count.")
            if kbb < 2.0:
                notes.append("Strikeout-to-walk improvement: emphasize first-pitch strikes + put-away pitch execution.")
        st.write("\n".join([f"- {n}" for n in notes]))

elif page == "Lineup Builder":
    st.subheader("Lineup Builder (transparent logic)")
    if bat_m is None or not len(bat_m):
        st.warning("Load batting to generate lineup suggestions.")
    else:
        min_pa = st.slider("Minimum PA to include", 0, int(bat_m["PA"].max() if "PA" in bat_m.columns else 40), 10)
        lineup = lineup_suggestions(bat_m, min_pa=min_pa)
        st.dataframe(lineup, use_container_width=True)

        st.caption("Logic: top-of-order = OBP + low K%; middle = SLG/OPS; bottom = contact/speed/turnover.")

elif page == "Exports":
    st.subheader("Exports")
    st.caption("Download computed tables for your weekly staff report / recruiting inserts.")

    if bat_m is not None and len(bat_m):
        st.download_button("Download batting (with computed metrics) CSV",
                           bat_m.to_csv(index=False).encode("utf-8"),
                           file_name="batting_with_metrics.csv",
                           mime="text/csv")
    else:
        st.info("No batting table to export.")

    if pit_m is not None and len(pit_m):
        st.download_button("Download pitching (with computed metrics) CSV",
                           pit_m.to_csv(index=False).encode("utf-8"),
                           file_name="pitching_with_metrics.csv",
                           mime="text/csv")
    else:
        st.info("No pitching table to export.")

    if fld_m is not None and len(fld_m):
        st.download_button("Download fielding (with computed metrics) CSV",
                           fld_m.to_csv(index=False).encode("utf-8"),
                           file_name="fielding_with_metrics.csv",
                           mime="text/csv")
    else:
        st.info("No fielding table to export.")
