import os
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

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="HS Baseball Recruiting Dashboard", layout="wide")

# ============================================================
# HERITAGE THEME (update hex if you want exact logo-matched)
# ============================================================
HERITAGE_NAVY = "#0B1F3B"
HERITAGE_BLUE = "#5FA8D3"
HERITAGE_GOLD = "#D6B25E"

st.markdown(
    f"""
<style>
/* Layout */
.block-container {{ padding-top: 1.0rem; padding-bottom: 2rem; }}
h1, h2, h3 {{ letter-spacing: -0.02em; }}

/* Sidebar */
section[data-testid="stSidebar"] {{
  border-right: 3px solid {HERITAGE_NAVY};
}}
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {{
  color: {HERITAGE_NAVY};
}}

/* Metric cards */
[data-testid="stMetric"] {{
  background: white;
  border: 1px solid rgba(49,51,63,.10);
  border-left: 6px solid {HERITAGE_BLUE};
  border-radius: 14px;
  padding: 14px 14px 10px 14px;
}}
[data-testid="stMetricLabel"] p {{
  font-weight: 900;
  color: {HERITAGE_NAVY};
}}
[data-testid="stMetricValue"] div {{
  font-weight: 950;
}}

/* Section title helper */
.section-title {{
  font-size: 1.05rem;
  font-weight: 950;
  color: {HERITAGE_NAVY};
  margin: 0.35rem 0 0.25rem 0;
}}
.subtle {{
  color: rgba(49,51,63,.70);
  font-size: 0.9rem;
}}
.hr {{
  border-top: 1px solid rgba(49,51,63,.10);
  margin: 1rem 0;
}}

/* DataFrames */
[data-testid="stDataFrame"] {{ border-radius: 14px; overflow: hidden; }}

/* Buttons */
.stDownloadButton button, .stButton button {{
  border-radius: 12px !important;
  border: 1px solid rgba(49,51,63,.18) !important;
}}
.stDownloadButton button:hover, .stButton button:hover {{
  border-color: {HERITAGE_NAVY} !important;
}}

/* Pills */
.pill {{
  display:inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  background: rgba(95,168,211,.12);
  border: 1px solid rgba(95,168,211,.25);
  color: {HERITAGE_NAVY};
  font-weight: 850;
  font-size: 0.85rem;
  margin-right: 8px;
}}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# HELPERS (formatting + display)
# ============================================================
def fmt_no0(x, decimals: int) -> str:
    """
    Remove leading 0 for decimals:
      0.232 -> .232
      1.232 -> 1.232
     -0.450 -> -.450
    """
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return "—"
        v = float(x)
        s = f"{v:.{decimals}f}"
        if s.startswith("0."):
            s = s.replace("0.", ".", 1)
        elif s.startswith("-0."):
            s = s.replace("-0.", "-.", 1)
        return s
    except Exception:
        return "—"


def fmt_pct(x, decimals: int = 1) -> str:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return "—"
        return f"{float(x) * 100:.{decimals}f}%"
    except Exception:
        return "—"


def int_safe(x) -> int:
    try:
        return int(float(x))
    except Exception:
        return 0


def df_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """
    Force strings so Streamlit doesn’t reformat numeric columns.
    (Especially helpful for things like IP_DISPLAY.)
    """
    return df.astype(str)


def ensure_avg(bat_m: pd.DataFrame | None) -> pd.DataFrame | None:
    """Ensure AVG exists on the computed batting metrics table."""
    if bat_m is None or bat_m.empty:
        return bat_m
    if "AVG" in bat_m.columns:
        return bat_m
    if "H" in bat_m.columns and "AB" in bat_m.columns:
        h = pd.to_numeric(bat_m["H"], errors="coerce").fillna(0)
        ab = pd.to_numeric(bat_m["AB"], errors="coerce").replace(0, pd.NA)
        bat_m["AVG"] = (h / ab).fillna(0).round(3)
    else:
        bat_m["AVG"] = 0.000
    return bat_m


def get_players(bat_m, pit_m, fld_m) -> list[str]:
    players = set()
    if bat_m is not None and "PLAYER" in bat_m.columns:
        players |= set(bat_m["PLAYER"].dropna().astype(str))
    if pit_m is not None and "PLAYER" in pit_m.columns:
        players |= set(pit_m["PLAYER"].dropna().astype(str))
    if fld_m is not None and "PLAYER" in fld_m.columns:
        players |= set(fld_m["PLAYER"].dropna().astype(str))
    return sorted(players)


def player_row(df: pd.DataFrame | None, player: str):
    if df is None or df.empty or "PLAYER" not in df.columns:
        return None
    m = df[df["PLAYER"].astype(str) == player]
    if m.empty:
        return None
    return m.iloc[0]


def sum_col(df: pd.DataFrame | None, col: str) -> float:
    if df is None or df.empty or col not in df.columns:
        return 0.0
    return float(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())


def team_batting_totals(bat_df: pd.DataFrame | None) -> dict:
    """
    TRUE TEAM TOTALS:
      AVG = H/AB
      OBP = (H+BB+HBP)/(AB+BB+HBP+SF)  (SF if present)
      SLG = TB/AB (TB from TB or 1B/2B/3B/HR)
      OPS = OBP + SLG
      K% = SO/PA (PA if present else AB+BB+HBP+SF)
      BB% = BB/PA
    """
    AB = sum_col(bat_df, "AB")
    H = sum_col(bat_df, "H")
    BB = sum_col(bat_df, "BB")
    HBP = sum_col(bat_df, "HBP")
    SF = sum_col(bat_df, "SF")
    SO = sum_col(bat_df, "SO")
    PA = sum_col(bat_df, "PA")

    if bat_df is not None and not bat_df.empty and "TB" in bat_df.columns:
        TB = sum_col(bat_df, "TB")
    else:
        oneB = sum_col(bat_df, "1B")
        twoB = sum_col(bat_df, "2B")
        threeB = sum_col(bat_df, "3B")
        HR = sum_col(bat_df, "HR")
        TB = oneB + 2 * twoB + 3 * threeB + 4 * HR

    if PA <= 0:
        PA = AB + BB + HBP + SF

    AVG = (H / AB) if AB > 0 else 0.0
    OBP_den = (AB + BB + HBP + SF) if (AB + BB + HBP + SF) > 0 else 0.0
    OBP = ((H + BB + HBP) / OBP_den) if OBP_den > 0 else 0.0
    SLG = (TB / AB) if AB > 0 else 0.0
    OPS = OBP + SLG
    Kp = (SO / PA) if PA > 0 else 0.0
    BBp = (BB / PA) if PA > 0 else 0.0

    return {
        "AB": AB,
        "H": H,
        "BB": BB,
        "HBP": HBP,
        "SF": SF,
        "SO": SO,
        "PA": PA,
        "TB": TB,
        "AVG": AVG,
        "OBP": OBP,
        "SLG": SLG,
        "OPS": OPS,
        "K%": Kp,
        "BB%": BBp,
    }


def team_pitching_totals(pit_df: pd.DataFrame | None) -> dict:
    """
    TRUE STAFF TOTALS (outs-aware):
      IP_TRUE is computed if available; else fall back to numeric IP.
    """
    if pit_df is None or pit_df.empty:
        return {
            "IP": 0.0,
            "BF": 0.0,
            "H": 0.0,
            "BB": 0.0,
            "SO": 0.0,
            "HR": 0.0,
            "ER": 0.0,
            "ERA": 0.0,
            "WHIP": 0.0,
            "K/BB": 0.0,
            "BB/INN": 0.0,
            "K/BF": 0.0,
        }

    # Prefer outs-aware numeric innings from utils.compute_pitching_metrics
    if "IP_TRUE" in pit_df.columns:
        IP = float(pd.to_numeric(pit_df["IP_TRUE"], errors="coerce").fillna(0).sum())
    else:
        IP = float(pd.to_numeric(pit_df.get("IP", 0), errors="coerce").fillna(0).sum())

    BF = sum_col(pit_df, "BF")
    H = sum_col(pit_df, "H")
    BB = sum_col(pit_df, "BB")
    SO = sum_col(pit_df, "SO")
    HR = sum_col(pit_df, "HR")
    ER = sum_col(pit_df, "ER")

    ERA = (9.0 * ER / IP) if IP > 0 else 0.0
    WHIP = ((H + BB) / IP) if IP > 0 else 0.0
    KBB = (SO / BB) if BB > 0 else float("inf") if SO > 0 else 0.0
    BBINN = (BB / IP) if IP > 0 else 0.0
    KBF = (SO / BF) if BF > 0 else 0.0

    return {
        "IP": IP,
        "BF": BF,
        "H": H,
        "BB": BB,
        "SO": SO,
        "HR": HR,
        "ER": ER,
        "ERA": ERA,
        "WHIP": WHIP,
        "K/BB": KBB,
        "BB/INN": BBINN,
        "K/BF": KBF,
    }


# ============================================================
# OPTIONAL MEDIA (V2-ready, NO submission form yet)
# ============================================================
MEDIA_CSV = "player_media.csv"


def load_player_media(path: str = MEDIA_CSV) -> pd.DataFrame:
    if os.path.exists(path):
        mdf = pd.read_csv(path)
        if "PLAYER" in mdf.columns:
            mdf["PLAYER"] = mdf["PLAYER"].astype(str).str.strip()
        return mdf
    return pd.DataFrame(
        columns=[
            "PLAYER",
            "HEADSHOT_URL",
            "SPRAY_BATTING_URL",
            "SPRAY_PITCHING_URL",
            "VIDEO_URL_1",
            "VIDEO_URL_2",
        ]
    )


def media_row(media_df: pd.DataFrame, player: str) -> dict:
    if media_df is None or media_df.empty or "PLAYER" not in media_df.columns:
        return {}
    m = media_df[media_df["PLAYER"].astype(str) == str(player)]
    if m.empty:
        return {}
    row = m.iloc[0].to_dict()
    return {k: ("" if pd.isna(v) else str(v)) for k, v in row.items()}


def is_url(s: str) -> bool:
    return isinstance(s, str) and s.strip().lower().startswith(("http://", "https://"))


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("Mode")
    mode = st.radio("Choose mode", ["Demo Mode", "Upload GameChanger CSVs"], index=0)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    st.header("Navigation")
    page = st.radio(
        "Go to",
        ["Team Overview", "Recruiting Profile", "Player Profiles", "Lineup Builder", "Exports"],
        index=0,
    )

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    logo_file = st.file_uploader("Optional: upload Heritage logo (png/jpg)", type=["png", "jpg", "jpeg"])
    if logo_file is not None:
        st.image(logo_file, use_container_width=True)

    st.caption("Upload batting / pitching / fielding CSVs (season totals).")


# ============================================================
# LOAD DATA
# ============================================================
if mode == "Demo Mode":
    bat, pit, fld = load_dataset(mode="demo")
else:
    st.info("Upload your GameChanger exports (season totals). If one file is missing, the app will still run.")
    bat_file = st.file_uploader("Batting CSV", type=["csv"], key="bat")
    pit_file = st.file_uploader("Pitching CSV", type=["csv"], key="pit")
    fld_file = st.file_uploader("Fielding CSV", type=["csv"], key="fld")
    bat, pit, fld = load_dataset(
        mode="upload",
        batting_file=bat_file,
        pitching_file=pit_file,
        fielding_file=fld_file,
    )

bat_m = compute_hitting_metrics(bat) if bat is not None else None
bat_m = ensure_avg(bat_m)

pit_m = compute_pitching_metrics(pit) if pit is not None else None
fld_m = compute_fielding_metrics(fld) if fld is not None else None

players = get_players(bat_m, pit_m, fld_m)
media_df = load_player_media()

# ============================================================
# HEADER
# ============================================================
st.title("HS Baseball Recruiting Dashboard")
st.markdown(
    """
<span class="pill">D1-style profile view</span>
<span class="pill">Recruiting Profile</span>
<span class="pill">Player compare</span>
<span class="pill">One-pager export</span>
<span class="pill">True team totals</span>
""",
    unsafe_allow_html=True,
)
st.caption("Team totals are computed from summed stat components (not medians / not averaging player rate stats).")


# ============================================================
# TEAM OVERVIEW
# ============================================================
if page == "Team Overview":
    st.subheader("Team Overview (True Totals)")

    bat_tot = team_batting_totals(bat)
    pit_tot = team_pitching_totals(pit_m if pit_m is not None else pit)

    c = st.columns(5)
    c[0].metric("Team AVG", fmt_no0(bat_tot["AVG"], 3))
    c[1].metric("Team OPS", fmt_no0(bat_tot["OPS"], 3))
    c[2].metric("Team OBP", fmt_no0(bat_tot["OBP"], 3))
    c[3].metric("Team SLG", fmt_no0(bat_tot["SLG"], 3))
    c[4].metric("Team K%", fmt_pct(bat_tot["K%"], 1))

    st.caption(
        f"Batting totals used — AB: {int(bat_tot['AB'])} | H: {int(bat_tot['H'])} | "
        f"BB: {int(bat_tot['BB'])} | SO: {int(bat_tot['SO'])} | PA: {int(bat_tot['PA'])}"
    )

    c2 = st.columns(4)
    staff_whip = pit_tot["WHIP"] if pit_tot["IP"] > 0 else None
    staff_kbb = pit_tot["K/BB"] if pit_tot["BB"] > 0 else None
    staff_bbi = pit_tot["BB/INN"] if pit_tot["IP"] > 0 else None
    staff_era = pit_tot["ERA"] if pit_tot["IP"] > 0 else None

    c2[0].metric("Staff WHIP", fmt_no0(staff_whip, 2) if staff_whip is not None else "—")
    c2[1].metric("Staff K/BB", fmt_no0(staff_kbb, 2) if staff_kbb is not None else "—")
    c2[2].metric("Staff BB/INN", fmt_no0(staff_bbi, 2) if staff_bbi is not None else "—")
    c2[3].metric("Staff ERA", fmt_no0(staff_era, 2) if staff_era is not None else "—")

    st.caption(
        f"Pitching totals used — IP (numeric): {fmt_no0(pit_tot['IP'], 1)} | H: {int(pit_tot['H'])} | "
        f"BB: {int(pit_tot['BB'])} | SO: {int(pit_tot['SO'])} | ER: {int(pit_tot['ER'])}"
    )

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:
        st.markdown("### Batting leaders")
        if bat_m is None or bat_m.empty:
            st.info("Load batting to see leaders.")
        else:
            tabs = st.tabs(["OPS", "AVG", "OBP", "SLG", "BB%", "K%", "XBH"])
            metrics = ["OPS", "AVG", "OBP", "SLG", "BB%", "K%", "XBH"]

            # display formatting map
            decimals = {"OPS": 3, "AVG": 3, "OBP": 3, "SLG": 3, "BB%": 3, "K%": 3}

            def _format_bat(df: pd.DataFrame) -> pd.DataFrame:
                out = df.copy()
                for col, dec in decimals.items():
                    if col in out.columns:
                        out[col] = pd.to_numeric(out[col], errors="coerce").apply(lambda v: fmt_no0(v, dec))
                if "XBH" in out.columns:
                    out["XBH"] = pd.to_numeric(out["XBH"], errors="coerce").fillna(0).astype(int).astype(str)
                return out

            for t, m in zip(tabs, metrics):
                with t:
                    df = top_table(bat_m, m, n=12, ascending=False)
                    st.dataframe(df_for_display(_format_bat(df)), use_container_width=True)

    with right:
        st.markdown("### Pitching leaders (IP shown as baseball innings)")
        if pit_m is None or pit_m.empty:
            st.info("Load pitching to see leaders.")
        else:
            pit_show = pit_m.copy()

            # Filter to pitchers with true innings > 0 (outs-aware)
            if "IP_TRUE" in pit_show.columns:
                pit_show["IP_TRUE_NUM"] = pd.to_numeric(pit_show["IP_TRUE"], errors="coerce").fillna(0)
                pit_show = pit_show[pit_show["IP_TRUE_NUM"] > 0].copy()
            else:
                pit_show["IP_NUM"] = pd.to_numeric(pit_show.get("IP", 0), errors="coerce").fillna(0)
                pit_show = pit_show[pit_show["IP_NUM"] > 0].copy()

            if pit_show.empty:
                st.info("No pitchers with IP > 0 found in this dataset.")
            else:
                tabs = st.tabs(["WHIP (low)", "K/BB", "K/BF", "BB/INN (low)", "ERA (low)"])
                dec = {"WHIP": 2, "K/BB": 2, "K/BF": 3, "BB/INN": 2, "ERA": 2}

                def _format_pit(df: pd.DataFrame) -> pd.DataFrame:
                    out = df.copy()
                    for col, d in dec.items():
                        if col in out.columns:
                            out[col] = pd.to_numeric(out[col], errors="coerce").apply(lambda v: fmt_no0(v, d))
                    # Keep IP_DISPLAY as-is (already a string like "12.2"), but ensure it's a string
                    if "IP_DISPLAY" in out.columns:
                        out["IP_DISPLAY"] = out["IP_DISPLAY"].astype(str)
                    return out

                with tabs[0]:
                    df = top_table(pit_show, "WHIP", n=12, ascending=True)
                    st.dataframe(df_for_display(_format_pit(df)), use_container_width=True)
                with tabs[1]:
                    df = top_table(pit_show, "K/BB", n=12, ascending=False)
                    st.dataframe(df_for_display(_format_pit(df)), use_container_width=True)
                with tabs[2]:
                    df = top_table(pit_show, "K/BF", n=12, ascending=False)
                    st.dataframe(df_for_display(_format_pit(df)), use_container_width=True)
                with tabs[3]:
                    df = top_table(pit_show, "BB/INN", n=12, ascending=True)
                    st.dataframe(df_for_display(_format_pit(df)), use_container_width=True)
                with tabs[4]:
                    df = top_table(pit_show, "ERA", n=12, ascending=True)
                    st.dataframe(df_for_display(_format_pit(df)), use_container_width=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    st.markdown("### Defense snapshot")
    if fld_m is None or fld_m.empty:
        st.info("Load fielding to see defense.")
    else:
        def_table = top_table(fld_m, "FPCT", n=12, ascending=False).copy()
        if "FPCT" in def_table.columns:
            def_table["FPCT"] = pd.to_numeric(def_table["FPCT"], errors="coerce").apply(lambda v: fmt_no0(v, 3))
        st.dataframe(df_for_display(def_table), use_container_width=True)


# ============================================================
# RECRUITING PROFILE (V2-ready: headshot + spray screenshot + videos)
# ============================================================
elif page == "Recruiting Profile":
    st.subheader("Recruiting Profile (D1 Recruiting Card)")

    if not players:
        st.warning("No players found. Load at least one CSV.")
    else:
        player = st.selectbox("Select player", players, index=0)
        m = media_row(media_df, player)

        top = st.columns([1, 2, 3])

        with top[0]:
            headshot = m.get("HEADSHOT_URL", "")
            if is_url(headshot):
                st.image(headshot, use_container_width=True)
            else:
                st.caption("Headshot: add HEADSHOT_URL in player_media.csv")

        with top[1]:
            st.markdown(f"### {player}")
            st.caption("Recruiting-facing snapshot • Metrics • Spray • Video")

        with top[2]:
            pbat = player_row(bat_m, player)
            ppit = player_row(pit_m, player)

            a, b, c, d = st.columns(4)
            if pbat is not None:
                a.metric("AVG", fmt_no0(pbat.get("AVG", 0), 3))
                b.metric("OBP", fmt_no0(pbat.get("OBP", 0), 3))
                c.metric("SLG", fmt_no0(pbat.get("SLG", 0), 3))
                d.metric("OPS", fmt_no0(pbat.get("OPS", 0), 3))
            elif ppit is not None:
                a.metric("ERA", fmt_no0(ppit.get("ERA", 0), 2))
                b.metric("WHIP", fmt_no0(ppit.get("WHIP", 0), 2))
                c.metric("K/BB", fmt_no0(ppit.get("K/BB", 0), 2))
                # ✅ show baseball-style IP, not 12.7
                d.metric("IP", str(ppit.get("IP_DISPLAY", "0.0")))
            else:
                st.info("No metrics available for this player yet.")

        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

        left, right = st.columns(2)

        with left:
            st.markdown("### Spray Chart (GameChanger screenshot)")
            tab1, tab2 = st.tabs(["Batting", "Pitching"])

            with tab1:
                sb = m.get("SPRAY_BATTING_URL", "")
                if is_url(sb):
                    st.image(sb, use_container_width=True)
                else:
                    st.caption("Add SPRAY_BATTING_URL in player_media.csv")

            with tab2:
                sp = m.get("SPRAY_PITCHING_URL", "")
                if is_url(sp):
                    st.image(sp, use_container_width=True)
                else:
                    st.caption("Add SPRAY_PITCHING_URL in player_media.csv")

        with right:
            st.markdown("### Video (highlights)")
            v1 = m.get("VIDEO_URL_1", "")
            v2 = m.get("VIDEO_URL_2", "")

            if is_url(v1):
                st.video(v1)
            else:
                st.caption("Add VIDEO_URL_1 in player_media.csv")

            if is_url(v2):
                st.video(v2)

        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

        st.markdown("### Player Evaluation Notes")
        # NOTE: you said evaluation templates later — keeping this placeholder for now.
        st.caption("Notes are currently auto-generated from performance indicators (V2 templates later).")
        st.write("- Keep approach stable and look for marginal gains (BB% up, K% down).")


# ============================================================
# PLAYER PROFILES (minimal view for now; keeps compare)
# ============================================================
elif page == "Player Profiles":
    st.subheader("Player Profiles (D1 Recruiting View)")

    if not players:
        st.warning("No players found. Load at least one CSV.")
    else:
        top_row = st.columns([3, 2, 5])
        with top_row[0]:
            player_a = st.selectbox("Player A", players, index=0)
        with top_row[1]:
            compare = st.toggle("Compare Players", value=False)

        player_b = None
        if compare:
            with top_row[2]:
                other_players = [p for p in players if p != player_a] or players
                player_b = st.selectbox("Player B", other_players, index=0)

        def render_profile(player_name: str):
            st.markdown(f"## {player_name}")
            st.markdown(
                '<div class="subtle">Quick scan: tools snapshot.</div>',
                unsafe_allow_html=True,
            )

            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown('<div class="section-title">Hitting</div>', unsafe_allow_html=True)
                pbat = player_row(bat_m, player_name)
                if pbat is None:
                    st.caption("No hitting data.")
                else:
                    r1 = st.columns(2)
                    r1[0].metric("AVG", fmt_no0(pbat.get("AVG", 0), 3))
                    r1[1].metric("OPS", fmt_no0(pbat.get("OPS", 0), 3))

                    r2 = st.columns(2)
                    r2[0].metric("OBP", fmt_no0(pbat.get("OBP", 0), 3))
                    r2[1].metric("SLG", fmt_no0(pbat.get("SLG", 0), 3))

                    r3 = st.columns(2)
                    r3[0].metric("BB%", fmt_pct(pbat.get("BB%", 0), 1))
                    r3[1].metric("K%", fmt_pct(pbat.get("K%", 0), 1))

                    r4 = st.columns(2)
                    r4[0].metric("XBH", str(int_safe(pbat.get("XBH", 0))))
                    r4[1].metric("SB", str(int_safe(pbat.get("SB", 0))))
                    st.caption(f"PA: {int_safe(pbat.get('PA', 0))}")

            with c2:
                st.markdown('<div class="section-title">Pitching</div>', unsafe_allow_html=True)
                ppit = player_row(pit_m, player_name)
                if ppit is None:
                    st.caption("No pitching data.")
                else:
                    r1 = st.columns(2)
                    r1[0].metric("IP", str(ppit.get("IP_DISPLAY", "0.0")))
                    r1[1].metric("ERA", fmt_no0(ppit.get("ERA", 0), 2))

                    r2 = st.columns(2)
                    r2[0].metric("WHIP", fmt_no0(ppit.get("WHIP", 0), 2))
                    r2[1].metric("K/BB", fmt_no0(ppit.get("K/BB", 0), 2))

                    r3 = st.columns(2)
                    r3[0].metric("BB/INN", fmt_no0(ppit.get("BB/INN", 0), 2))
                    r3[1].metric("K/BF", fmt_no0(ppit.get("K/BF", 0), 3))

                    st.metric("HR", str(int_safe(ppit.get("HR", 0))))

            with c3:
                st.markdown('<div class="section-title">Defense</div>', unsafe_allow_html=True)
                pfld = player_row(fld_m, player_name)
                if pfld is None:
                    st.caption("No fielding data.")
                else:
                    st.metric("FPCT", fmt_no0(pfld.get("FPCT", 0), 3))
                    r1 = st.columns(2)
                    r1[0].metric("TC", str(int_safe(pfld.get("TC", 0))))
                    r1[1].metric("E", str(int_safe(pfld.get("E", 0))))

                    r2 = st.columns(2)
                    r2[0].metric("PO", str(int_safe(pfld.get("PO", 0))))
                    r2[1].metric("A", str(int_safe(pfld.get("A", 0))))
                    st.metric("DP", str(int_safe(pfld.get("DP", 0))))

        if compare and player_b:
            l, r = st.columns(2)
            with l:
                render_profile(player_a)
            with r:
                render_profile(player_b)
        else:
            render_profile(player_a)


# ============================================================
# LINEUP BUILDER
# ============================================================
elif page == "Lineup Builder":
    st.subheader("Lineup Builder (transparent logic)")

    if bat_m is None or bat_m.empty:
        st.warning("Load batting to generate lineup suggestions.")
    else:
        max_pa = int(pd.to_numeric(bat_m.get("PA", 40), errors="coerce").fillna(40).max())
        min_pa = st.slider("Minimum PA to include", 0, max_pa, min(10, max_pa))

        lineup = lineup_suggestions(bat_m, min_pa=min_pa)
        st.dataframe(lineup, use_container_width=True)
        st.caption("Logic: top-of-order = OBP + low K%; middle = SLG/OPS impact; bottom = contact/speed/turnover.")


# ============================================================
# EXPORTS (Recruiting One-Pager + Data)
# ============================================================
elif page == "Exports":
    st.subheader("Exports")
    st.caption("Computed data tables export (CSV). One-pager export can be added back in later.")

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.markdown("### Data Exports")

    if bat_m is not None and not bat_m.empty:
        st.download_button(
            "Download batting (with computed metrics) CSV",
            bat_m.to_csv(index=False).encode("utf-8"),
            file_name="batting_with_metrics.csv",
            mime="text/csv",
        )
    else:
        st.info("No batting table to export.")

    if pit_m is not None and not pit_m.empty:
        st.download_button(
            "Download pitching (with computed metrics) CSV",
            pit_m.to_csv(index=False).encode("utf-8"),
            file_name="pitching_with_metrics.csv",
            mime="text/csv",
        )
    else:
        st.info("No pitching table to export.")

    if fld_m is not None and not fld_m.empty:
        st.download_button(
            "Download fielding (with computed metrics) CSV",
            fld_m.to_csv(index=False).encode("utf-8"),
            file_name="fielding_with_metrics.csv",
            mime="text/csv",
        )
    else:
        st.info("No fielding table to export.")
