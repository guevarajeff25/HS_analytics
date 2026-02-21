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

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="HS Baseball Recruiting Dashboard", layout="wide")

# ----------------------------
# Heritage theme (edit these 3 hex values if needed)
# ----------------------------
HERITAGE_NAVY = "#0B1F3B"
HERITAGE_BLUE = "#5FA8D3"
HERITAGE_GOLD = "#D6B25E"  # optional accent

st.markdown(
    f"""
<style>
.block-container {{ padding-top: 1.0rem; }}
h1, h2, h3 {{ letter-spacing: -0.02em; }}

section[data-testid="stSidebar"] {{
  border-right: 3px solid {HERITAGE_NAVY};
}}
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {{
  color: {HERITAGE_NAVY};
}}

[data-testid="stMetric"] {{
  background: white;
  border: 1px solid rgba(49,51,63,.10);
  border-left: 6px solid {HERITAGE_BLUE};
  border-radius: 14px;
  padding: 14px 14px 10px 14px;
}}
[data-testid="stMetricLabel"] p {{
  font-weight: 800;
  color: {HERITAGE_NAVY};
}}
[data-testid="stMetricValue"] div {{
  font-weight: 900;
}}

.section-title {{
  font-size: 1.05rem;
  font-weight: 900;
  color: {HERITAGE_NAVY};
  margin: 0.35rem 0 0.25rem 0;
}}
.hr {{
  border-top: 1px solid rgba(49,51,63,.10);
  margin: 1rem 0;
}}

[data-testid="stDataFrame"] {{ border-radius: 14px; overflow: hidden; }}

.stDownloadButton button, .stButton button {{
  border-radius: 12px !important;
  border: 1px solid rgba(49,51,63,.18) !important;
}}
.stDownloadButton button:hover, .stButton button:hover {{
  border-color: {HERITAGE_NAVY} !important;
}}
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------
# Helpers
# ----------------------------
def fmt_no_leading_zero(x, decimals: int) -> str:
    """0.232 -> .232, -0.450 -> -.450"""
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


def col_sum(df, col) -> float:
    if df is None or col not in df.columns:
        return 0.0
    return float(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())


def safe_div(n: float, d: float) -> float:
    return (n / d) if d and d != 0 else 0.0


def ensure_avg(bat_m: pd.DataFrame) -> pd.DataFrame:
    """If AVG wasn't created in utils, compute it safely."""
    if bat_m is None or not len(bat_m):
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
    if df is None or "PLAYER" not in df.columns:
        return None
    m = df[df["PLAYER"].astype(str) == player]
    if len(m) == 0:
        return None
    return m.iloc[0]


def build_coach_notes(player: str, bat_m, pit_m) -> list[str]:
    notes = []

    pbat = player_row(bat_m, player)
    if pbat is not None:
        k = float(pbat.get("K%", 0) or 0)
        bb = float(pbat.get("BB%", 0) or 0)
        ops = float(pbat.get("OPS", 0) or 0)

        if k > 0.22:
            notes.append("Approach: reduce swing-and-miss — tighten 2-strike plan and zone control.")
        if bb < 0.06:
            notes.append("On-base: target a higher walk rate by hunting one zone early in counts.")

        try:
            med_ops = float(pd.to_numeric(bat_m["OPS"], errors="coerce").dropna().median())
            if ops < med_ops:
                notes.append("Impact: prioritize hard contact and driving mistakes (barrel the mistake).")
        except Exception:
            pass

    ppit = player_row(pit_m, player)
    if ppit is not None:
        whip = float(ppit.get("WHIP", 0) or 0)
        kbb = float(ppit.get("K/BB", 0) or 0)
        if whip > 1.6:
            notes.append("Pitching: reduce traffic — attack early and limit free passes.")
        if kbb < 2.0:
            notes.append("Pitching: improve K/BB — first-pitch strikes + better put-away execution.")

    if not notes:
        notes.append("Maintain: keep approach stable and look for marginal gains (BB% up, K% down).")

    return notes


def build_one_pager_markdown(player: str, bat_m, pit_m, fld_m) -> str:
    pbat = player_row(bat_m, player)
    ppit = player_row(pit_m, player)
    pfld = player_row(fld_m, player)
    notes = build_coach_notes(player, bat_m, pit_m)

    lines = []
    lines.append(f"# Recruiting One-Pager — {player}")
    lines.append("")
    lines.append("## Snapshot")

    lines.append("### Hitting")
    if pbat is None:
        lines.append("- No hitting data available.")
    else:
        lines.append(f"- PA: {int_safe(pbat.get('PA', 0))}")
        lines.append(f"- AVG: {fmt_no_leading_zero(pbat.get('AVG', 0), 3)}")
        lines.append(f"- OBP: {fmt_no_leading_zero(pbat.get('OBP', 0), 3)}")
        lines.append(f"- SLG: {fmt_no_leading_zero(pbat.get('SLG', 0), 3)}")
        lines.append(f"- OPS: {fmt_no_leading_zero(pbat.get('OPS', 0), 3)}")
        lines.append(f"- BB%: {fmt_pct(pbat.get('BB%', 0), 1)}")
        lines.append(f"- K%: {fmt_pct(pbat.get('K%', 0), 1)}")
        lines.append(f"- XBH: {int_safe(pbat.get('XBH', 0))}")
        lines.append(f"- SB: {int_safe(pbat.get('SB', 0))}")

    lines.append("")
    lines.append("### Pitching")
    if ppit is None:
        lines.append("- No pitching data available.")
    else:
        lines.append(f"- IP: {fmt_no_leading_zero(ppit.get('IP', 0), 1)}")
        lines.append(f"- ERA: {fmt_no_leading_zero(ppit.get('ERA', 0), 2)}")
        lines.append(f"- WHIP: {fmt_no_leading_zero(ppit.get('WHIP', 0), 2)}")
        lines.append(f"- K/BB: {fmt_no_leading_zero(ppit.get('K/BB', 0), 2)}")
        lines.append(f"- BB/INN: {fmt_no_leading_zero(ppit.get('BB/INN', 0), 2)}")
        lines.append(f"- K/BF: {fmt_no_leading_zero(ppit.get('K/BF', 0), 3)}")
        lines.append(f"- HR: {int_safe(ppit.get('HR', 0))}")

    lines.append("")
    lines.append("### Fielding")
    if pfld is None:
        lines.append("- No fielding data available.")
    else:
        lines.append(f"- FPCT: {fmt_no_leading_zero(pfld.get('FPCT', 0), 3)}")
        lines.append(f"- TC: {int_safe(pfld.get('TC', 0))}")
        lines.append(f"- PO: {int_safe(pfld.get('PO', 0))}")
        lines.append(f"- A: {int_safe(pfld.get('A', 0))}")
        lines.append(f"- E: {int_safe(pfld.get('E', 0))}")
        lines.append(f"- DP: {int_safe(pfld.get('DP', 0))}")

    lines.append("")
    lines.append("## Coach Notes")
    for n in notes:
        lines.append(f"- {n}")

    lines.append("")
    lines.append("_Generated by HS Baseball Recruiting Dashboard._")
    return "\n".join(lines)


# ----- Team totals computations -----
def compute_team_batting(bat_raw: pd.DataFrame | None, bat_m: pd.DataFrame | None):
    """TRUE team batting from summed counting stats."""
    src = bat_raw if bat_raw is not None and len(bat_raw) else bat_m
    if src is None or not len(src):
        return None

    AB = col_sum(src, "AB")
    H = col_sum(src, "H")
    BB = col_sum(src, "BB")
    HBP = col_sum(src, "HBP")
    SF = col_sum(src, "SF")
    SO = col_sum(src, "SO")
    PA = col_sum(src, "PA")
    if PA == 0:
        PA = AB + BB + HBP + SF

    team_avg = safe_div(H, AB)
    obp_den = AB + BB + HBP + SF
    team_obp = safe_div(H + BB + HBP, obp_den)

    TB = col_sum(src, "TB")
    if TB == 0:
        HR = col_sum(src, "HR")
        d2 = col_sum(src, "2B") if "2B" in src.columns else col_sum(src, "Doubles")
        d3 = col_sum(src, "3B") if "3B" in src.columns else col_sum(src, "Triples")
        singles = max(H - d2 - d3 - HR, 0)
        TB = singles + (2 * d2) + (3 * d3) + (4 * HR)

    team_slg = safe_div(TB, AB)
    team_ops = team_obp + team_slg
    team_kp = safe_div(SO, PA)

    return {"AVG": team_avg, "OBP": team_obp, "SLG": team_slg, "OPS": team_ops, "K%": team_kp, "AB": AB, "H": H, "PA": PA}


def compute_team_pitching(pit_raw: pd.DataFrame | None, pit_m: pd.DataFrame | None):
    """
    TRUE team pitching from summed counting stats.

    Uses:
      - WHIP = (BB + H) / IP
      - K/BB = K / BB
      - BB/INN = BB / IP
      - ERA = 9 * ER / IP  (falls back to R if ER missing)

    Notes:
      - Prefer pit_raw if it contains counting stats; otherwise fall back to pit_m.
      - If IP is stored as innings with .1/.2 (baseball outs format), we convert to true innings.
    """
    src = pit_raw if pit_raw is not None and len(pit_raw) else pit_m
    if src is None or not len(src):
        return None

    # Convert IP to true innings if it uses .1/.2 notation
    ip_series = pd.to_numeric(src.get("IP", 0), errors="coerce").fillna(0)

    # If values like 12.2 mean 12 and 2/3, convert:
    ip_int = ip_series.astype(int)
    ip_frac = (ip_series - ip_int).round(1)
    # treat .1 = 1 out, .2 = 2 outs
    ip_true = ip_int + (ip_frac.replace({0.1: 1 / 3, 0.2: 2 / 3, 0.0: 0.0}))
    IP = float(ip_true.sum())

    BB = col_sum(src, "BB")
    H = col_sum(src, "H")
    K = col_sum(src, "SO")
    if K == 0:
        K = col_sum(src, "K")  # some exports use K

    ER = col_sum(src, "ER")
    if ER == 0:
        ER = col_sum(src, "R")  # fallback if ER not present

    team_whip = safe_div(BB + H, IP)
    team_kbb = safe_div(K, BB)
    team_bb_inn = safe_div(BB, IP)
    team_era = safe_div(9.0 * ER, IP)

    return {"WHIP": team_whip, "K/BB": team_kbb, "BB/INN": team_bb_inn, "ERA": team_era, "IP": IP, "BB": BB, "H": H, "K": K, "ER": ER}


# ----------------------------
# Sidebar controls
# ----------------------------
with st.sidebar:
    st.header("Mode")
    mode = st.radio("Choose mode", ["Demo Mode", "Upload GameChanger CSVs"], index=0)

    st.divider()
    st.header("Navigation")
    page = st.radio("Go to", ["Team Overview", "Player Cards", "Lineup Builder", "Exports"], index=1)

    st.divider()
    logo_file = st.file_uploader("Optional: upload Heritage logo (png/jpg)", type=["png", "jpg", "jpeg"])
    if logo_file is not None:
        st.image(logo_file, use_container_width=True)

    st.caption("Upload batting / pitching / fielding CSVs (season totals).")

# ----------------------------
# Load data
# ----------------------------
if mode == "Demo Mode":
    bat, pit, fld = load_dataset(mode="demo")
else:
    st.info(
        "Upload your GameChanger exports (season totals). If one file is missing, the app will still run with what you provide."
    )
    bat_file = st.file_uploader("Batting CSV", type=["csv"], key="bat")
    pit_file = st.file_uploader("Pitching CSV", type=["csv"], key="pit")
    fld_file = st.file_uploader("Fielding CSV", type=["csv"], key="fld")
    bat, pit, fld = load_dataset(mode="upload", batting_file=bat_file, pitching_file=pit_file, fielding_file=fld_file)

bat_m = compute_hitting_metrics(bat) if bat is not None else None
bat_m = ensure_avg(bat_m) if bat_m is not None else None

pit_m = compute_pitching_metrics(pit) if pit is not None else None
fld_m = compute_fielding_metrics(fld) if fld is not None else None

players = get_players(bat_m, pit_m, fld_m)

# ----------------------------
# Header
# ----------------------------
st.title("HS Baseball Recruiting Dashboard")
st.caption("Team totals (true sums) + player cards + compare + exports (one-pager now, PDF later).")

# ============================================================
# TEAM OVERVIEW
# ============================================================
if page == "Team Overview":
    st.subheader("Team Overview")

    # Batting totals
    team_bat = compute_team_batting(bat, bat_m)
    cols = st.columns(5)
    if team_bat is not None:
        cols[0].metric("Team AVG", fmt_no_leading_zero(team_bat["AVG"], 3))
        cols[1].metric("Team OPS", fmt_no_leading_zero(team_bat["OPS"], 3))
        cols[2].metric("Team OBP", fmt_no_leading_zero(team_bat["OBP"], 3))
        cols[3].metric("Team SLG", fmt_no_leading_zero(team_bat["SLG"], 3))
        cols[4].metric("Team K%", fmt_pct(team_bat["K%"], 1))
        st.caption(f"Batting totals used — AB: {int(team_bat['AB'])} | H: {int(team_bat['H'])} | PA: {int(team_bat['PA'])}")
    else:
        cols[0].metric("Team AVG", "—")
        cols[1].metric("Team OPS", "—")
        cols[2].metric("Team OBP", "—")
        cols[3].metric("Team SLG", "—")
        cols[4].metric("Team K%", "—")

    # Pitching totals
    cols2 = st.columns(4)
    team_pit = compute_team_pitching(pit, pit_m)
    if team_pit is not None:
        cols2[0].metric("Staff WHIP", fmt_no_leading_zero(team_pit["WHIP"], 2))
        cols2[1].metric("Staff K/BB", fmt_no_leading_zero(team_pit["K/BB"], 2))
        cols2[2].metric("Staff BB/INN", fmt_no_leading_zero(team_pit["BB/INN"], 2))
        cols2[3].metric("Staff ERA", fmt_no_leading_zero(team_pit["ERA"], 2))
        st.caption(
            f"Pitching totals used — IP: {fmt_no_leading_zero(team_pit['IP'], 1)} | K: {int(team_pit['K'])} | BB: {int(team_pit['BB'])} | H: {int(team_pit['H'])} | ER: {int(team_pit['ER'])}"
        )
    else:
        cols2[0].metric("Staff WHIP", "—")
        cols2[1].metric("Staff K/BB", "—")
        cols2[2].metric("Staff BB/INN", "—")
        cols2[3].metric("Staff ERA", "—")

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:
        st.markdown("### Batting leaders")
        if bat_m is not None and len(bat_m):
            tabs = st.tabs(["OPS", "AVG", "OBP", "SLG", "BB%", "K%", "XBH"])
            metrics = ["OPS", "AVG", "OBP", "SLG", "BB%", "K%", "XBH"]
            for t, m in zip(tabs, metrics):
                with t:
                    st.dataframe(top_table(bat_m, m, n=10), use_container_width=True)
        else:
            st.info("Load batting to see leaders.")

    with right:
        st.markdown("### Pitching leaders")

        pit_show = pd.DataFrame()
        if pit_m is not None and len(pit_m):
            pit_show = pit_m.copy()
            pit_show["IP_NUM"] = pd.to_numeric(pit_show.get("IP", 0), errors="coerce").fillna(0)
            pit_show = pit_show[pit_show["IP_NUM"] > 0].copy()

        if pit_show.empty:
            st.info("No pitchers with IP > 0 found in this dataset.")
        else:
            tabs = st.tabs(["WHIP (low)", "K/BB", "K/BF", "BB/INN (low)", "ERA (low)"])
            with tabs[0]:
                st.dataframe(top_table(pit_show, "WHIP", n=10, ascending=True), use_container_width=True)
            with tabs[1]:
                st.dataframe(top_table(pit_show, "K/BB", n=10), use_container_width=True)
            with tabs[2]:
                st.dataframe(top_table(pit_show, "K/BF", n=10), use_container_width=True)
            with tabs[3]:
                st.dataframe(top_table(pit_show, "BB/INN", n=10, ascending=True), use_container_width=True)
            with tabs[4]:
                st.dataframe(top_table(pit_show, "ERA", n=10, ascending=True), use_container_width=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    st.markdown("### Defense snapshot")
    if fld_m is not None and len(fld_m):
        def_table = top_table(fld_m, "FPCT", n=12).copy()
        if "FPCT" in def_table.columns:
            def_table["FPCT"] = pd.to_numeric(def_table["FPCT"], errors="coerce").fillna(0)

        st.dataframe(
            def_table,
            use_container_width=True,
            column_config=(
                {"FPCT": st.column_config.NumberColumn("FPCT", format="%.3f")}
                if "FPCT" in def_table.columns
                else None
            ),
        )
    else:
        st.info("Load fielding to see defense.")

# ============================================================
# PLAYER CARDS + COMPARE
# ============================================================
elif page == "Player Cards":
    st.subheader("Player Cards")

    if not players:
        st.warning("No players found. Load at least one CSV.")
    else:
        top_row = st.columns([2, 2, 6])
        with top_row[0]:
            player_a = st.selectbox("Player A", players, index=0)
        with top_row[1]:
            compare = st.toggle("Compare Players", value=False)
        player_b = None
        if compare:
            with top_row[2]:
                other_players = [p for p in players if p != player_a] or players
                player_b = st.selectbox("Player B", other_players, index=0)

        def render_player_cards(player_name: str):
            st.markdown(f"### {player_name}")

            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown('<div class="section-title">Hitting</div>', unsafe_allow_html=True)
                pbat = player_row(bat_m, player_name)
                if pbat is None:
                    st.caption("No hitting data.")
                else:
                    r1 = st.columns(2)
                    r1[0].metric("AVG", fmt_no_leading_zero(pbat.get("AVG", 0), 3))
                    r1[1].metric("OPS", fmt_no_leading_zero(pbat.get("OPS", 0), 3))

                    r2 = st.columns(2)
                    r2[0].metric("OBP", fmt_no_leading_zero(pbat.get("OBP", 0), 3))
                    r2[1].metric("SLG", fmt_no_leading_zero(pbat.get("SLG", 0), 3))

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
                    r1[0].metric("IP", fmt_no_leading_zero(ppit.get("IP", 0), 1))
                    r1[1].metric("ERA", fmt_no_leading_zero(ppit.get("ERA", 0), 2))

                    r2 = st.columns(2)
                    r2[0].metric("WHIP", fmt_no_leading_zero(ppit.get("WHIP", 0), 2))
                    r2[1].metric("K/BB", fmt_no_leading_zero(ppit.get("K/BB", 0), 2))

                    r3 = st.columns(2)
                    r3[0].metric("BB/INN", fmt_no_leading_zero(ppit.get("BB/INN", 0), 2))
                    r3[1].metric("K/BF", fmt_no_leading_zero(ppit.get("K/BF", 0), 3))

                    st.metric("HR", str(int_safe(ppit.get("HR", 0))))

            with c3:
                st.markdown('<div class="section-title">Fielding</div>', unsafe_allow_html=True)
                pfld = player_row(fld_m, player_name)
                if pfld is None:
                    st.caption("No fielding data.")
                else:
                    st.metric("FPCT", fmt_no_leading_zero(pfld.get("FPCT", 0), 3))
                    r1 = st.columns(2)
                    r1[0].metric("TC", str(int_safe(pfld.get("TC", 0))))
                    r1[1].metric("E", str(int_safe(pfld.get("E", 0))))

                    r2 = st.columns(2)
                    r2[0].metric("PO", str(int_safe(pfld.get("PO", 0))))
                    r2[1].metric("A", str(int_safe(pfld.get("A", 0))))

                    st.metric("DP", str(int_safe(pfld.get("DP", 0))))

            st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
            st.markdown("### Coach Notes")
            notes = build_coach_notes(player_name, bat_m, pit_m)
            st.write("\n".join([f"- {n}" for n in notes]))

        if compare and player_b:
            left_col, right_col = st.columns(2)
            with left_col:
                render_player_cards(player_a)
            with right_col:
                render_player_cards(player_b)
        else:
            render_player_cards(player_a)

# ============================================================
# LINEUP BUILDER
# ============================================================
elif page == "Lineup Builder":
    st.subheader("Lineup Builder (transparent logic)")

    if bat_m is None or not len(bat_m):
        st.warning("Load batting to generate lineup suggestions.")
    else:
        max_pa = int(pd.to_numeric(bat_m.get("PA", 40), errors="coerce").fillna(40).max())
        min_pa = st.slider("Minimum PA to include", 0, max_pa, min(10, max_pa))

        lineup = lineup_suggestions(bat_m, min_pa=min_pa)
        st.dataframe(lineup, use_container_width=True)
        st.caption("Logic: top-of-order = OBP + low K%; middle = SLG/OPS; bottom = contact/speed/turnover.")

# ============================================================
# EXPORTS
# ============================================================
elif page == "Exports":
    st.subheader("Exports")
    st.caption("Download computed tables + recruiting one-pager (Markdown now, PDF later).")

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.markdown("### Recruiting One-Pager (Markdown)")

    if not players:
        st.info("Load at least one CSV to export a one-pager.")
    else:
        one_pager_player = st.selectbox("Select player for one-pager", players, index=0)
        md = build_one_pager_markdown(one_pager_player, bat_m, pit_m, fld_m)

        with st.expander("Preview one-pager"):
            st.markdown(md)

        st.download_button(
            "Download Recruiting One-Pager (.md)",
            md.encode("utf-8"),
            file_name=f"{one_pager_player.replace(' ', '_')}_recruiting_one_pager.md",
            mime="text/markdown",
        )

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.markdown("### Data Exports")

    if bat_m is not None and len(bat_m):
        st.download_button(
            "Download batting (with computed metrics) CSV",
            bat_m.to_csv(index=False).encode("utf-8"),
            file_name="batting_with_metrics.csv",
            mime="text/csv",
        )
    else:
        st.info("No batting table to export.")

    if pit_m is not None and len(pit_m):
        st.download_button(
            "Download pitching (with computed metrics) CSV",
            pit_m.to_csv(index=False).encode("utf-8"),
            file_name="pitching_with_metrics.csv",
            mime="text/csv",
        )
    else:
        st.info("No pitching table to export.")

    if fld_m is not None and len(fld_m):
        st.download_button(
            "Download fielding (with computed metrics) CSV",
            fld_m.to_csv(index=False).encode("utf-8"),
            file_name="fielding_with_metrics.csv",
            mime="text/csv",
        )
    else:
        st.info("No fielding table to export.")
