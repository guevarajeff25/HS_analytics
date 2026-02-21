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
# HELPERS (formatting, totals, profiles)
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


def num_safe(series_or_val, default=0.0) -> float:
    try:
        if series_or_val is None:
            return float(default)
        v = float(series_or_val)
        if pd.isna(v):
            return float(default)
        return v
    except Exception:
        return float(default)


def ensure_avg(bat_m: pd.DataFrame | None) -> pd.DataFrame | None:
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


def ip_to_outs(ip_val: float) -> int:
    """
    Convert baseball IP formats safely.
    - If fractional part looks like .1/.2 => treat as 1/3 and 2/3 innings.
    - Otherwise treat as true decimal innings.
    """
    try:
        ip = float(ip_val)
    except Exception:
        return 0

    whole = int(ip)
    frac = round(ip - whole, 3)

    # Common baseball notation
    if abs(frac - 0.1) < 0.001:
        return whole * 3 + 1
    if abs(frac - 0.2) < 0.001:
        return whole * 3 + 2

    # Otherwise: decimal innings
    return int(round(ip * 3))


def outs_to_ip(outs: int) -> float:
    if outs <= 0:
        return 0.0
    return outs / 3.0


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

    # TB
    if bat_df is not None and not bat_df.empty and "TB" in bat_df.columns:
        TB = sum_col(bat_df, "TB")
    else:
        oneB = sum_col(bat_df, "1B")
        twoB = sum_col(bat_df, "2B")
        threeB = sum_col(bat_df, "3B")
        HR = sum_col(bat_df, "HR")
        TB = oneB + 2 * twoB + 3 * threeB + 4 * HR

    # If PA missing, estimate
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
        "AB": AB, "H": H, "BB": BB, "HBP": HBP, "SF": SF, "SO": SO, "PA": PA, "TB": TB,
        "AVG": AVG, "OBP": OBP, "SLG": SLG, "OPS": OPS, "K%": Kp, "BB%": BBp
    }


def team_pitching_totals(pit_df: pd.DataFrame | None) -> dict:
    """
    TRUE STAFF TOTALS (sum components then compute rates):
      IP = sum IP (via outs-safe conversion)
      ERA = 9*ER/IP
      WHIP = (H+BB)/IP
      K/BB = SO/BB
      BB/INN = BB/IP
      K/BF = SO/BF
    """
    if pit_df is None or pit_df.empty:
        return {
            "IP": 0.0, "BF": 0.0, "H": 0.0, "BB": 0.0, "SO": 0.0, "HR": 0.0, "ER": 0.0,
            "ERA": 0.0, "WHIP": 0.0, "K/BB": 0.0, "BB/INN": 0.0, "K/BF": 0.0
        }

    # Sum innings via outs conversion
    if "IP" in pit_df.columns:
        outs = int(pd.to_numeric(pit_df["IP"], errors="coerce").fillna(0).apply(ip_to_outs).sum())
        IP = outs_to_ip(outs)
    else:
        IP = 0.0

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
        "IP": IP, "BF": BF, "H": H, "BB": BB, "SO": SO, "HR": HR, "ER": ER,
        "ERA": ERA, "WHIP": WHIP, "K/BB": KBB, "BB/INN": BBINN, "K/BF": KBF
    }


def grade_20_80(pct: float) -> int:
    """Convert 0..1 percentile to a scouting grade on 20-80 (rounded to nearest 5)."""
    try:
        p = max(0.0, min(1.0, float(pct)))
        g = 20 + 60 * p
        return int(round(g / 5.0) * 5)
    except Exception:
        return 50


def percentile(series: pd.Series, value: float, higher_is_better: bool = True) -> float:
    """
    Percentile of a value relative to a series. Returns 0..1.
    If higher_is_better=False, invert.
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return 0.5
    v = float(value)
    pct = (s.rank(pct=True).loc[s.index]).mean()  # not used directly
    # Use empirical CDF:
    p = (s <= v).mean()
    if higher_is_better:
        return float(p)
    return float(1.0 - p)


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
    lines.append(f"# D1 Recruiting One-Pager — {player}")
    lines.append("")
    lines.append("## Tools Snapshot")

    lines.append("### Hitting")
    if pbat is None:
        lines.append("- No hitting data available.")
    else:
        lines.append(f"- PA: {int_safe(pbat.get('PA', 0))}")
        lines.append(f"- AVG: {fmt_no0(pbat.get('AVG', 0), 3)}")
        lines.append(f"- OBP: {fmt_no0(pbat.get('OBP', 0), 3)}")
        lines.append(f"- SLG: {fmt_no0(pbat.get('SLG', 0), 3)}")
        lines.append(f"- OPS: {fmt_no0(pbat.get('OPS', 0), 3)}")
        lines.append(f"- BB%: {fmt_pct(pbat.get('BB%', 0), 1)}")
        lines.append(f"- K%: {fmt_pct(pbat.get('K%', 0), 1)}")
        lines.append(f"- XBH: {int_safe(pbat.get('XBH', 0))}")
        lines.append(f"- SB: {int_safe(pbat.get('SB', 0))}")

    lines.append("")
    lines.append("### Pitching")
    if ppit is None:
        lines.append("- No pitching data available.")
    else:
        lines.append(f"- IP: {fmt_no0(ppit.get('IP', 0), 1)}")
        lines.append(f"- ERA: {fmt_no0(ppit.get('ERA', 0), 2)}")
        lines.append(f"- WHIP: {fmt_no0(ppit.get('WHIP', 0), 2)}")
        lines.append(f"- K/BB: {fmt_no0(ppit.get('K/BB', 0), 2)}")
        lines.append(f"- BB/INN: {fmt_no0(ppit.get('BB/INN', 0), 2)}")
        lines.append(f"- K/BF: {fmt_no0(ppit.get('K/BF', 0), 3)}")
        lines.append(f"- HR: {int_safe(ppit.get('HR', 0))}")

    lines.append("")
    lines.append("### Defense")
    if pfld is None:
        lines.append("- No fielding data available.")
    else:
        lines.append(f"- FPCT: {fmt_no0(pfld.get('FPCT', 0), 3)}")
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
    lines.append("_Generated by HS Baseball Recruiting Dashboard (D1-style)._")
    return "\n".join(lines)


def format_leaderboard(df: pd.DataFrame, decimals_map: dict) -> pd.DataFrame:
    """
    Convert key numeric columns to formatted strings without leading zeros.
    Keeps other columns as-is.
    """
    out = df.copy()
    for col, dec in decimals_map.items():
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").apply(lambda v: fmt_no0(v, dec))
    return out


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("Mode")
    mode = st.radio("Choose mode", ["Demo Mode", "Upload GameChanger CSVs"], index=0)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    st.header("Navigation")
    page = st.radio("Go to", ["Team Overview", "Player Profiles", "Lineup Builder", "Exports"], index=0)

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
    bat, pit, fld = load_dataset(mode="upload", batting_file=bat_file, pitching_file=pit_file, fielding_file=fld_file)

bat_m = compute_hitting_metrics(bat) if bat is not None else None
bat_m = ensure_avg(bat_m)

pit_m = compute_pitching_metrics(pit) if pit is not None else None
fld_m = compute_fielding_metrics(fld) if fld is not None else None

players = get_players(bat_m, pit_m, fld_m)

# ============================================================
# HEADER
# ============================================================
st.title("HS Baseball Recruiting Dashboard")
st.markdown(
    f"""
<span class="pill">D1-style profile view</span>
<span class="pill">Player compare</span>
<span class="pill">One-pager export</span>
<span class="pill">True team totals</span>
""",
    unsafe_allow_html=True,
)
st.caption("Team totals are computed from summed stat components (not medians / not averaging player rate stats).")


# ============================================================
# TEAM OVERVIEW (TRUE TOTALS)
# ============================================================
if page == "Team Overview":
    st.subheader("Team Overview (True Totals)")

    bat_tot = team_batting_totals(bat)
    pit_tot = team_pitching_totals(pit)

    c = st.columns(5)
    c[0].metric("Team AVG", fmt_no0(bat_tot["AVG"], 3))
    c[1].metric("Team OPS", fmt_no0(bat_tot["OPS"], 3))
    c[2].metric("Team OBP", fmt_no0(bat_tot["OBP"], 3))
    c[3].metric("Team SLG", fmt_no0(bat_tot["SLG"], 3))
    c[4].metric("Team K%", fmt_pct(bat_tot["K%"], 1))

    st.caption(
        f"Batting totals used — AB: {int(bat_tot['AB'])} | H: {int(bat_tot['H'])} | BB: {int(bat_tot['BB'])} | SO: {int(bat_tot['SO'])} | PA: {int(bat_tot['PA'])}"
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
        f"Pitching totals used — IP: {fmt_no0(pit_tot['IP'], 1)} | H: {int(pit_tot['H'])} | BB: {int(pit_tot['BB'])} | SO: {int(pit_tot['SO'])} | ER: {int(pit_tot['ER'])}"
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
            decimals = {"OPS": 3, "AVG": 3, "OBP": 3, "SLG": 3, "BB%": 3, "K%": 3}
            for t, m in zip(tabs, metrics):
                with t:
                    df = top_table(bat_m, m, n=12, ascending=False)
                    df_show = format_leaderboard(df, decimals_map=decimals)
                    st.dataframe(df_show, use_container_width=True)

    with right:
        st.markdown("### Pitching leaders")
        if pit_m is None or pit_m.empty:
            st.info("Load pitching to see leaders.")
        else:
            pit_show = pit_m.copy()
            pit_show["IP_NUM"] = pd.to_numeric(pit_show.get("IP", 0), errors="coerce").fillna(0)
            pit_show = pit_show[pit_show["IP_NUM"] > 0].copy()

            if pit_show.empty:
                st.info("No pitchers with IP > 0 found in this dataset.")
            else:
                tabs = st.tabs(["WHIP (low)", "K/BB", "K/BF", "BB/INN (low)", "ERA (low)"])
                dec = {"WHIP": 2, "K/BB": 2, "K/BF": 3, "BB/INN": 2, "ERA": 2}

                with tabs[0]:
                    df = top_table(pit_show, "WHIP", n=12, ascending=True)
                    st.dataframe(format_leaderboard(df, dec), use_container_width=True)
                with tabs[1]:
                    df = top_table(pit_show, "K/BB", n=12, ascending=False)
                    st.dataframe(format_leaderboard(df, dec), use_container_width=True)
                with tabs[2]:
                    df = top_table(pit_show, "K/BF", n=12, ascending=False)
                    st.dataframe(format_leaderboard(df, dec), use_container_width=True)
                with tabs[3]:
                    df = top_table(pit_show, "BB/INN", n=12, ascending=True)
                    st.dataframe(format_leaderboard(df, dec), use_container_width=True)
                with tabs[4]:
                    df = top_table(pit_show, "ERA", n=12, ascending=True)
                    st.dataframe(format_leaderboard(df, dec), use_container_width=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    st.markdown("### Defense snapshot")
    if fld_m is None or fld_m.empty:
        st.info("Load fielding to see defense.")
    else:
        def_table = top_table(fld_m, "FPCT", n=12, ascending=False).copy()
        # show FPCT without leading zero
        if "FPCT" in def_table.columns:
            def_table["FPCT"] = pd.to_numeric(def_table["FPCT"], errors="coerce").apply(lambda v: fmt_no0(v, 3))
        st.dataframe(def_table, use_container_width=True)


# ============================================================
# PLAYER PROFILES (D1 style) + COMPARE
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

        def recruiting_grades(player_name: str):
            """
            Compute a small set of percentiles + 20-80 grades for a D1-ish quick scan.
            """
            grades = []

            # Hitting (higher better except K%)
            pbat = player_row(bat_m, player_name)
            if pbat is not None and bat_m is not None and not bat_m.empty:
                for label, col, hib, dec in [
                    ("OPS", "OPS", True, 3),
                    ("OBP", "OBP", True, 3),
                    ("SLG", "SLG", True, 3),
                    ("K%", "K%", False, 3),
                    ("BB%", "BB%", True, 3),
                ]:
                    if col in bat_m.columns:
                        val = float(pbat.get(col, 0) or 0)
                        pct = percentile(bat_m[col], val, higher_is_better=hib)
                        grades.append((label, fmt_no0(val, dec) if " %" not in label else fmt_pct(val), pct, grade_20_80(pct)))

            # Pitching (lower better for ERA/WHIP/BBINN)
            ppit = player_row(pit_m, player_name)
            if ppit is not None and pit_m is not None and not pit_m.empty:
                for label, col, hib, dec in [
                    ("ERA", "ERA", False, 2),
                    ("WHIP", "WHIP", False, 2),
                    ("K/BB", "K/BB", True, 2),
                    ("BB/INN", "BB/INN", False, 2),
                    ("K/BF", "K/BF", True, 3),
                ]:
                    if col in pit_m.columns:
                        val = float(ppit.get(col, 0) or 0)
                        pct = percentile(pit_m[col], val, higher_is_better=hib)
                        grades.append((label, fmt_no0(val, dec), pct, grade_20_80(pct)))

            # Defense (higher better)
            pfld = player_row(fld_m, player_name)
            if pfld is not None and fld_m is not None and not fld_m.empty:
                if "FPCT" in fld_m.columns:
                    val = float(pfld.get("FPCT", 0) or 0)
                    pct = percentile(fld_m["FPCT"], val, higher_is_better=True)
                    grades.append(("FPCT", fmt_no0(val, 3), pct, grade_20_80(pct)))

            return grades

        def render_profile(player_name: str):
            st.markdown(f"## {player_name}")
            st.markdown('<div class="subtle">Quick scan: tools + percentiles + grades (20–80 scale).</div>', unsafe_allow_html=True)

            # ---- TOP: Tools Snapshot ----
            c1, c2, c3 = st.columns(3)

            # Hitting card
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

            # Pitching card
            with c2:
                st.markdown('<div class="section-title">Pitching</div>', unsafe_allow_html=True)
                ppit = player_row(pit_m, player_name)
                if ppit is None:
                    st.caption("No pitching data.")
                else:
                    r1 = st.columns(2)
                    r1[0].metric("IP", fmt_no0(ppit.get("IP", 0), 1))
                    r1[1].metric("ERA", fmt_no0(ppit.get("ERA", 0), 2))

                    r2 = st.columns(2)
                    r2[0].metric("WHIP", fmt_no0(ppit.get("WHIP", 0), 2))
                    r2[1].metric("K/BB", fmt_no0(ppit.get("K/BB", 0), 2))

                    r3 = st.columns(2)
                    r3[0].metric("BB/INN", fmt_no0(ppit.get("BB/INN", 0), 2))
                    r3[1].metric("K/BF", fmt_no0(ppit.get("K/BF", 0), 3))

                    st.metric("HR", str(int_safe(ppit.get("HR", 0))))

            # Defense card
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

            # ---- Middle: Recruiting Grades ----
            st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
            st.markdown("### Recruiting Grades (20–80) + Percentiles")

            grades = recruiting_grades(player_name)
            if not grades:
                st.info("Not enough data to compute grades.")
            else:
                for label, val_str, pct, grade in grades[:12]:
                    row = st.columns([2, 2, 5, 2])
                    row[0].markdown(f"**{label}**")
                    row[1].markdown(val_str)
                    row[2].progress(max(0.0, min(1.0, float(pct))))
                    row[3].markdown(f"**{grade}**")

            # ---- Bottom: Coach Notes ----
            st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
            st.markdown("### Coach Notes (actionable)")
            notes = build_coach_notes(player_name, bat_m, pit_m)
            st.write("\n".join([f"- {n}" for n in notes]))

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
    st.caption("Recruiting one-pager (Markdown now; PDF later) + computed data tables.")

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.markdown("### Recruiting One-Pager (Markdown export)")

    if not players:
        st.info("Load at least one CSV to export a one-pager.")
    else:
        one_pager_player = st.selectbox("Select player", players, index=0)
        md = build_one_pager_markdown(one_pager_player, bat_m, pit_m, fld_m)

        with st.expander("Preview one-pager"):
            st.markdown(md)

        st.download_button(
            "Download One-Pager (.md)",
            md.encode("utf-8"),
            file_name=f"{one_pager_player.replace(' ', '_')}_recruiting_one_pager.md",
            mime="text/markdown",
        )

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
