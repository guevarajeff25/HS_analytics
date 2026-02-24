# app.py
import os
from io import BytesIO

import pandas as pd
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

from utils import (
    compute_fielding_metrics,
    compute_hitting_metrics,
    compute_pitching_metrics,
    lineup_suggestions,
    load_dataset,
    top_table,
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
    (Especially helpful for things like IP.)
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
      OBP = (H+BB+HBP)/(AB+BB+HBP+SF)
      SLG = TB/AB
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
      Prefer IP_TRUE if present (from utils.compute_pitching_metrics).
      Otherwise fall back to numeric IP sum (less accurate if .1/.2 present).
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
def render_scouting_grades_compact(grades: list[tuple[str, str, float, int]], max_rows: int = 6):
    """
    grades rows are: (label, val_str, pct_0to1, grade_20to80)
    Compact UI version for Recruiting Profile.
    """
    if not grades:
        st.caption("Scouting grades: not enough data yet.")
        return

    # Small header
    st.markdown("#### Scouting Grades (20–80)")
    st.caption("Quick snapshot (top signals)")

    # Render compact rows
    show = grades[:max_rows]
    for label, val_str, pct, grade in show:
        row = st.columns([2.4, 1.4, 3.2, 1.0])
        row[0].markdown(f"**{label}**")
        row[1].markdown(f"<span class='subtle'>{val_str}</span>", unsafe_allow_html=True)
        row[2].progress(max(0.0, min(1.0, float(pct))))
        row[3].markdown(f"**{grade}**")

# ============================================================
# SCOUTING GRADE HELPERS (20–80) + PERCENTILES
# Put these at module-level (NOT nested inside a page or render func)
# ============================================================
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
    p = float((s <= v).mean())
    return p if higher_is_better else float(1.0 - p)


def recruiting_grades(player_name: str, bat_m: pd.DataFrame | None, pit_m: pd.DataFrame | None, fld_m: pd.DataFrame | None):
    """
    Compute a small set of percentiles + 20-80 grades for a D1-ish quick scan.
    Returns list[(label, val_str, pct_0to1, grade_20to80)].
    """
    grades: list[tuple[str, str, float, int]] = []

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
                val_str = fmt_pct(val, 1) if label in {"K%", "BB%"} else fmt_no0(val, dec)
                grades.append((label, val_str, pct, grade_20_80(pct)))

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
    if pfld is not None and fld_m is not None and not fld_m.empty and "FPCT" in fld_m.columns:
        val = float(pfld.get("FPCT", 0) or 0)
        pct = percentile(fld_m["FPCT"], val, higher_is_better=True)
        grades.append(("FPCT", fmt_no0(val, 3), pct, grade_20_80(pct)))

    return grades


# ============================================================
# PLAYER MEDIA + BIO (FULL SCHEMA, backward compatible)
# ============================================================
MEDIA_CSV = "player_media.csv"

# Accept both schemas (old/full + new/simple) by mapping aliases -> canonical names.
COLUMN_ALIASES = {
    # headshot
    "PHOTO_URL": "HEADSHOT_URL",
    "HEADSHOT": "HEADSHOT_URL",
    "HEADSHOT": "HEADSHOT_URL",

    # spray charts
    "SPRAY_CHART_URL": "SPRAY_BATTING_URL",        # older single-spray becomes batting by default
    "SPRAY_URL": "SPRAY_BATTING_URL",
    "SPRAY_HITTING_URL": "SPRAY_BATTING_URL",
    "SPRAY_PITCH_URL": "SPRAY_PITCHING_URL",

    # video
    "VIDEO_URL": "VIDEO_URL_1",
    "VIDEO1_URL": "VIDEO_URL_1",
    "VIDEO2_URL": "VIDEO_URL_2",
}

# Canonical columns the app will use everywhere
CANONICAL_MEDIA_COLS = [
    "PLAYER",
    "HEADSHOT_URL",
    "SPRAY_BATTING_URL",
    "SPRAY_PITCHING_URL",
    "VIDEO_URL_1",
    "VIDEO_URL_2",
]

# Full bio columns (optional, shown on Recruiting Profile)
BIO_COLS = [
    "NUMBER",
    "POSITION",
    "HEIGHT",
    "WEIGHT",
    "BATS",
    "THROWS",
    "GRAD_YEAR",
    "HOMETOWN",
    "PLAYER_EVAL_NOTES",
]


def load_player_media(path: str = MEDIA_CSV) -> pd.DataFrame:
    """
    Loads player_media.csv and normalizes columns so BOTH schemas work:
      - Full bio schema (your CSV): PHOTO_URL, SPRAY_CHART_URL, VIDEO_URL, etc.
      - Simple schema: HEADSHOT_URL, SPRAY_BATTING_URL, VIDEO_URL_1, etc.
    """
    if not os.path.exists(path):
        # Empty template with both canonical + bio columns
        return pd.DataFrame(columns=CANONICAL_MEDIA_COLS + BIO_COLS)

    mdf = pd.read_csv(path)

    # Normalize PLAYER
    if "PLAYER" in mdf.columns:
        mdf["PLAYER"] = mdf["PLAYER"].astype(str).str.strip()

    # Apply alias mapping (only if alias exists and canonical does not)
    for src, dst in COLUMN_ALIASES.items():
        if src in mdf.columns and dst not in mdf.columns:
            mdf[dst] = mdf[src]

    # Ensure canonical columns exist
    for c in CANONICAL_MEDIA_COLS:
        if c not in mdf.columns:
            mdf[c] = ""

    # Ensure bio cols exist (optional)
    for c in BIO_COLS:
        if c not in mdf.columns:
            mdf[c] = ""

    return mdf


def media_row(media_df: pd.DataFrame, player: str) -> dict:
    if media_df is None or media_df.empty or "PLAYER" not in media_df.columns:
        return {}
    m = media_df[media_df["PLAYER"].astype(str) == str(player).strip()]
    if m.empty:
        return {}
    row = m.iloc[0].to_dict()
    return {k: ("" if pd.isna(v) else str(v).strip()) for k, v in row.items()}


def is_url(s: str) -> bool:
    return isinstance(s, str) and s.strip().lower().startswith(("http://", "https://"))


def media_exists_local(path: str) -> bool:
    """
    Support local repo paths like:
      assets/headshots/jett.jpg
    """
    if not isinstance(path, str) or not path.strip():
        return False
    return os.path.exists(path)


def show_image_or_hint(label: str, value: str, hint: str):
    if is_url(value):
        st.image(value, use_container_width=True)
    elif media_exists_local(value):
        st.image(value, use_container_width=True)
    else:
        st.caption(f"{label}: {hint}")

# ============================================================
# ONE-PAGER PDF EXPORT (real PDF bytes)
# ============================================================
def _hex_to_rl_color(hex_str: str):
    hs = hex_str.lstrip("#")
    r = int(hs[0:2], 16) / 255.0
    g = int(hs[2:4], 16) / 255.0
    b = int(hs[4:6], 16) / 255.0
    return colors.Color(r, g, b)


def build_one_pager_pdf_bytes(player: str, pbat, ppit, pfld) -> bytes:
    """
    Build a real PDF in memory (ReportLab). Returns bytes suitable for st.download_button.
    """
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    w, h = letter

    navy = _hex_to_rl_color(HERITAGE_NAVY)
    blue = _hex_to_rl_color(HERITAGE_BLUE)
    gold = _hex_to_rl_color(HERITAGE_GOLD)

    margin = 0.7 * inch
    x0 = margin
    y = h - margin

    # Header bar
    c.setFillColor(blue)
    c.roundRect(x0, y - 0.55 * inch, w - 2 * margin, 0.55 * inch, 10, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(x0 + 0.25 * inch, y - 0.35 * inch, "HS Baseball Recruiting — One-Pager")

    c.setFont("Helvetica-Bold", 20)
    c.setFillColor(navy)
    y -= 0.8 * inch
    c.drawString(x0, y, player)

    c.setFont("Helvetica", 10)
    c.setFillColor(colors.black)
    y -= 0.18 * inch
    c.drawString(x0, y, "Generated from HS Baseball Recruiting Dashboard")

    def draw_card(title: str, rows: list[tuple[str, str]], left: float, top: float, width: float, height: float):
        c.setFillColor(colors.white)
        c.setStrokeColor(colors.lightgrey)
        c.roundRect(left, top - height, width, height, 10, fill=1, stroke=1)

        c.setFillColor(navy)
        c.setFont("Helvetica-Bold", 11)
        c.drawString(left + 0.18 * inch, top - 0.25 * inch, title)

        c.setStrokeColor(gold)
        c.setLineWidth(3)
        c.line(left + 0.18 * inch, top - 0.30 * inch, left + width - 0.18 * inch, top - 0.30 * inch)

        c.setFillColor(colors.black)
        c.setFont("Helvetica", 10)
        yy = top - 0.48 * inch
        for k, v in rows:
            if yy < (top - height + 0.22 * inch):
                break
            c.setFillColor(colors.HexColor("#333333"))
            c.drawString(left + 0.18 * inch, yy, str(k))
            c.setFillColor(colors.black)
            c.setFont("Helvetica-Bold", 10)
            c.drawRightString(left + width - 0.18 * inch, yy, str(v))
            c.setFont("Helvetica", 10)
            yy -= 0.20 * inch

    def v(row, key, default=None):
        if row is None:
            return default
        try:
            return row.get(key, default)
        except Exception:
            return default

    hit_rows = [
        ("AVG", fmt_no0(v(pbat, "AVG", None), 3) if pbat is not None else "—"),
        ("OBP", fmt_no0(v(pbat, "OBP", None), 3) if pbat is not None else "—"),
        ("SLG", fmt_no0(v(pbat, "SLG", None), 3) if pbat is not None else "—"),
        ("OPS", fmt_no0(v(pbat, "OPS", None), 3) if pbat is not None else "—"),
        ("BB%", fmt_pct(v(pbat, "BB%", 0), 1) if pbat is not None else "—"),
        ("K%", fmt_pct(v(pbat, "K%", 0), 1) if pbat is not None else "—"),
        ("PA", str(int_safe(v(pbat, "PA", 0))) if pbat is not None else "—"),
        ("XBH", str(int_safe(v(pbat, "XBH", 0))) if pbat is not None else "—"),
        ("SB", str(int_safe(v(pbat, "SB", 0))) if pbat is not None else "—"),
    ]

    pit_rows = [
        ("IP", str(v(ppit, "IP", "—")) if ppit is not None else "—"),
        ("ERA", fmt_no0(v(ppit, "ERA", None), 2) if ppit is not None else "—"),
        ("WHIP", fmt_no0(v(ppit, "WHIP", None), 2) if ppit is not None else "—"),
        ("K/BB", fmt_no0(v(ppit, "K/BB", None), 2) if ppit is not None else "—"),
        ("K/BF", fmt_no0(v(ppit, "K/BF", None), 3) if ppit is not None else "—"),
        ("BB/INN", fmt_no0(v(ppit, "BB/INN", None), 2) if ppit is not None else "—"),
        ("HR Allowed", str(int_safe(v(ppit, "HR", 0))) if ppit is not None else "—"),
    ]

    def_rows = [
        ("FPCT", fmt_no0(v(pfld, "FPCT", None), 3) if pfld is not None else "—"),
        ("TC", str(int_safe(v(pfld, "TC", 0))) if pfld is not None else "—"),
        ("E", str(int_safe(v(pfld, "E", 0))) if pfld is not None else "—"),
        ("PO", str(int_safe(v(pfld, "PO", 0))) if pfld is not None else "—"),
        ("A", str(int_safe(v(pfld, "A", 0))) if pfld is not None else "—"),
        ("DP", str(int_safe(v(pfld, "DP", 0))) if pfld is not None else "—"),
    ]

    y -= 0.35 * inch
    card_top = y
    card_h = 2.75 * inch
    gap = 0.25 * inch
    card_w = (w - 2 * margin - 2 * gap) / 3.0

    draw_card("Hitting", hit_rows, x0, card_top, card_w, card_h)
    draw_card("Pitching", pit_rows, x0 + card_w + gap, card_top, card_w, card_h)
    draw_card("Defense", def_rows, x0 + 2 * (card_w + gap), card_top, card_w, card_h)

    c.setFont("Helvetica-Oblique", 9)
    c.setFillColor(colors.HexColor("#555555"))
    c.drawString(x0, margin - 0.15 * inch, "Tip: This PDF is dashboard-generated; attach to recruiting emails.")
    c.showPage()
    c.save()

    return buf.getvalue()


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
        f"Pitching totals used — IP: {fmt_no0(pit_tot['IP'], 1)} | H: {int(pit_tot['H'])} | "
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
                    df = top_table(bat_m, m, n=25, ascending=False)
                    st.dataframe(df_for_display(_format_bat(df)), use_container_width=True, hide_index=True)

    with right:
        st.markdown("### Pitching leaders")
        if pit_m is None or pit_m.empty:
            st.info("Load pitching to see leaders.")
        else:
            pit_show = pit_m.copy()

            if "IP_TRUE" in pit_show.columns:
                ip_true = pd.to_numeric(pit_show["IP_TRUE"], errors="coerce").fillna(0)
                pit_show = pit_show[ip_true > 0].copy()
            else:
                ip_num = pd.to_numeric(pit_show.get("IP", 0), errors="coerce").fillna(0)
                pit_show = pit_show[ip_num > 0].copy()

            pit_show = pit_show.drop(columns=["IP_DISPLAY", "IP_TRUE", "IP_TRUE_NUM", "IP_NUM"], errors="ignore")

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
                    if "IP" in out.columns:
                        out["IP"] = out["IP"].astype(str)
                    return out

                with tabs[0]:
                    df = top_table(pit_show, "WHIP", n=12, ascending=True)
                    st.dataframe(df_for_display(_format_pit(df)), use_container_width=True, hide_index=True)
                with tabs[1]:
                    df = top_table(pit_show, "K/BB", n=12, ascending=False)
                    st.dataframe(df_for_display(_format_pit(df)), use_container_width=True, hide_index=True)
                with tabs[2]:
                    df = top_table(pit_show, "K/BF", n=12, ascending=False)
                    st.dataframe(df_for_display(_format_pit(df)), use_container_width=True, hide_index=True)
                with tabs[3]:
                    df = top_table(pit_show, "BB/INN", n=12, ascending=True)
                    st.dataframe(df_for_display(_format_pit(df)), use_container_width=True, hide_index=True)
                with tabs[4]:
                    df = top_table(pit_show, "ERA", n=12, ascending=True)
                    st.dataframe(df_for_display(_format_pit(df)), use_container_width=True, hide_index=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    st.markdown("### Defense snapshot")
    if fld_m is None or fld_m.empty:
        st.info("Load fielding to see defense.")
    else:
        def_table = top_table(fld_m, "FPCT", n=25, ascending=False).copy()
        if "FPCT" in def_table.columns:
            def_table["FPCT"] = pd.to_numeric(def_table["FPCT"], errors="coerce").apply(lambda v: fmt_no0(v, 3))
        st.dataframe(df_for_display(def_table), use_container_width=True, hide_index=True)

# ============================================================
# RECRUITING PROFILE
# ============================================================
elif page == "Recruiting Profile":
    st.subheader("Recruiting Profile")

    if not players:
        st.warning("No players found. Load at least one CSV.")
    else:
        player = st.selectbox("Select player", players, index=0)
        m = media_row(media_df, player)

        # ---- Top layout ----
        top = st.columns([1.2, 1.6, 3.2])

        with top[0]:
            headshot = m.get("HEADSHOT_URL", "")
            show_image_or_hint(
                "Headshot",
                headshot,
                "Add PHOTO_URL or HEADSHOT_URL (URL or local path like assets/headshots/jett.jpg)",
            )

        with top[1]:
            st.markdown(f"### {player}")

            # Bio pills (only show if present)
            bio_lines = []
            if m.get("NUMBER"):
                bio_lines.append(f"**#{m['NUMBER']}**")
            if m.get("POSITION"):
                bio_lines.append(f"**Pos:** {m['POSITION']}")
            if m.get("BATS") or m.get("THROWS"):
                bio_lines.append(f"**B/T:** {m.get('BATS','')} / {m.get('THROWS','')}".strip(" /"))
            if m.get("HEIGHT") or m.get("WEIGHT"):
                bio_lines.append(f"**HT/WT:** {m.get('HEIGHT','')} / {m.get('WEIGHT','')}".strip(" /"))
            if m.get("GRAD_YEAR"):
                bio_lines.append(f"**Grad:** {m['GRAD_YEAR']}")
            if m.get("HOMETOWN"):
                bio_lines.append(f"**Hometown:** {m['HOMETOWN']}")

            if bio_lines:
                st.write(" • ".join(bio_lines))
            else:
                st.caption("Add bio fields in player_media.csv (POSITION, BATS/THROWS, HT/WT, etc.)")

        with top[2]:
            # Stats snapshot from computed metrics (already in your app)
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
                d.metric("IP", str(ppit.get("IP", "0.0")))
            else:
                st.info("No metrics available for this player yet.")

        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

        # ---- Media section ----
        left, right = st.columns(2)

        with left:
            st.markdown("### Spray Chart")
            tab1, tab2 = st.tabs(["Batting", "Pitching"])

            with tab1:
                sb = m.get("SPRAY_BATTING_URL", "")
                # If they only have SPRAY_CHART_URL, our alias mapping puts it into SPRAY_BATTING_URL
                show_image_or_hint(
                    "Batting spray",
                    sb,
                    "Add SPRAY_CHART_URL or SPRAY_BATTING_URL (URL or local path like assets/spray/jett_batting.png)",
                )

            with tab2:
                sp = m.get("SPRAY_PITCHING_URL", "")
                show_image_or_hint(
                    "Pitching spray",
                    sp,
                    "Add SPRAY_PITCHING_URL (optional)",
                )

        with right:
            # --- Compact grades above video ---
            st.markdown("### Evaluation")
            grades = recruiting_grades(player, bat_m, pit_m, fld_m)
            render_scouting_grades_compact(grades, max_rows=6)

            st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

            # --- Video ---
            st.markdown("### Video")
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
        notes = m.get("PLAYER_EVAL_NOTES", "").strip()
        if notes:
            st.write(notes)
        else:
            st.caption("Add PLAYER_EVAL_NOTES in player_media.csv (optional).")

# ============================================================
# PLAYER PROFILES
# ============================================================
elif page == "Player Profiles":
    st.subheader("Player Profiles")

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
            st.markdown('<div class="subtle">Quick scan: tools snapshot.</div>', unsafe_allow_html=True)

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
                    r1[0].metric("IP", str(ppit.get("IP", "0.0")))
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

            st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
            st.markdown("### Scouting Grades (20–80) + Percentiles")

            grades = recruiting_grades(player_name, bat_m, pit_m, fld_m)
            if not grades:
                st.info("Not enough data to compute scouting grades.")
            else:
                for label, val_str, pct, grade in grades[:12]:
                    row = st.columns([2, 2, 5, 2])
                    row[0].markdown(f"**{label}**")
                    row[1].markdown(val_str)
                    row[2].progress(max(0.0, min(1.0, float(pct))))
                    row[3].markdown(f"**{grade}**")

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
    st.subheader("Lineup Builder")

    if bat_m is None or bat_m.empty:
        st.warning("Load batting to generate lineup suggestions.")
    else:
        max_pa = int(pd.to_numeric(bat_m.get("PA", 40), errors="coerce").fillna(40).max())
        min_pa = st.slider("Minimum PA to include", 0, max_pa, min(10, max_pa))

        lineup = lineup_suggestions(bat_m, min_pa=min_pa)
        st.dataframe(lineup, use_container_width=True)
        st.caption("Logic: top-of-order = OBP + low K%; middle = SLG/OPS impact; bottom = contact/speed/turnover.")

# ============================================================
# EXPORTS (PDF one-pager + CSV exports)
# ============================================================
elif page == "Exports":
    st.subheader("Exports")
    st.caption("Download a real PDF one-pager, plus the computed tables as CSV.")

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    st.markdown("### One-Pager Export (PDF)")

    if not players:
        st.info("No players found. Load batting/pitching/fielding to generate a one-pager.")
    else:
        export_player = st.selectbox("Select player", players, index=0, key="onepager_player")

        pbat = player_row(bat_m, export_player)
        ppit = player_row(pit_m, export_player)
        pfld = player_row(fld_m, export_player)

        pdf_bytes = build_one_pager_pdf_bytes(export_player, pbat, ppit, pfld)

        st.download_button(
            "Download One-Pager (PDF)",
            data=pdf_bytes,
            file_name=f"{export_player.replace(' ', '_')}_one_pager.pdf",
            mime="application/pdf",
        )

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    st.markdown("### Data Exports (CSV)")

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
        pit_export = pit_m.copy()
        for internal in ["IP_TRUE", "IP_DISPLAY", "IP_TRUE_NUM", "IP_NUM"]:
            pit_export = pit_export.drop(columns=[internal], errors="ignore")

        st.download_button(
            "Download pitching (with computed metrics) CSV",
            pit_export.to_csv(index=False).encode("utf-8"),
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
