import os
import re
import pandas as pd

DEMO_DIR = os.path.join(os.path.dirname(__file__), "demo")


def read_gamechanger_csv(path_or_file) -> pd.DataFrame:
    """Read a GameChanger export CSV robustly.

    Some GC exports include a *section row* (e.g., 'Batting', 'Pitching') and/or a
    *glossary row* and may render real headers as the first non-empty row.
    This function detects that pattern and returns a clean DataFrame with proper headers.
    """
    # First attempt: normal read
    try:
        df = pd.read_csv(path_or_file)
    except Exception:
        # fallback for odd encodings
        df = pd.read_csv(path_or_file, encoding="utf-8-sig", engine="python")

    # If the file looks like it has real headers (few Unnamed), keep it.
    unnamed_ratio = sum(str(c).startswith("Unnamed") for c in df.columns) / max(len(df.columns), 1)
    if unnamed_ratio < 0.5 and any(c for c in df.columns if str(c).strip()):
        return df

    # Otherwise, parse as raw rows and detect the header row.
    raw = pd.read_csv(path_or_file, header=None, encoding="utf-8-sig", engine="python")

    def norm_cell(x):
        if pd.isna(x): 
            return ""
        return str(x).strip()

    # Find the row that looks like the header row.
    header_row = None
    for i in range(min(len(raw), 30)):
        row = [norm_cell(v) for v in raw.iloc[i].tolist()]
        joined = "|".join(row).lower()
        if ("number" in joined and "last" in joined and "first" in joined) or ("player" in joined and "gp" in joined):
            header_row = i
            break

    if header_row is None:
        # Give up and return the best-effort df we already read
        return df

    header = [norm_cell(v) for v in raw.iloc[header_row].tolist()]
    # Drop completely empty header cells
    header = [h if h != "" else f"COL_{idx}" for idx, h in enumerate(header)]

    data = raw.iloc[header_row+1:].copy()
    data.columns = header

    # Drop glossary/blank rows
    first_col = data.columns[0]
    data = data[~data[first_col].astype(str).str.contains(r"^\s*Glossary\s*$", case=False, na=False)]
    data = data[~data[first_col].astype(str).str.contains(r"^\s*$", na=False)]

    # If there's an all-NaN spacer row right after header, drop it
    data = data.dropna(how="all")

    return data.reset_index(drop=True)


def _norm(col: str) -> str:
    """Normalize a column name to a simple key: uppercase alnum only."""
    return re.sub(r"[^A-Z0-9]+", "", str(col).upper())

def _rename_by_aliases(df: pd.DataFrame, alias_map: dict) -> pd.DataFrame:
    if df is None:
        return df
    cols = {c: _norm(c) for c in df.columns}
    inv = {}
    for original, key in cols.items():
        inv.setdefault(key, []).append(original)

    rename = {}
    for target, aliases in alias_map.items():
        for a in aliases:
            akey = _norm(a)
            if akey in inv:
                # choose the first matching original column
                rename[inv[akey][0]] = target
                break
    return df.rename(columns=rename)

def _coerce_numeric(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df


def _ensure_player_name(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a canonical PLAYER column exists.

    GameChanger exports often have Number/First/Last instead of a single Player column.
    This creates PLAYER = "First Last" (and keeps Number if present).
    """
    if df is None:
        return df
    if "PLAYER" in df.columns:
        return df
    # common GC columns
    first_col = None
    last_col = None
    for c in df.columns:
        if _norm(c) in ("FIRST",):
            first_col = c
        if _norm(c) in ("LAST",):
            last_col = c
    if first_col and last_col:
        df["PLAYER"] = df[first_col].astype(str).str.strip() + " " + df[last_col].astype(str).str.strip()
        df["PLAYER"] = df["PLAYER"].str.replace(r"\s+", " ", regex=True).str.strip()
    return df


def load_dataset(mode: str, batting_file=None, pitching_file=None, fielding_file=None):
    """
    Loads either:
      - demo CSVs from ./demo/
      - uploaded CSVs (season totals)
    Returns: (bat_df, pit_df, fld_df) which may be None if missing.
    """
    if mode == "demo":
        bat = read_gamechanger_csv(f"{DEMO_DIR}/batting.csv")
        pit = read_gamechanger_csv(f"{DEMO_DIR}/pitching.csv")
        fld = read_gamechanger_csv(f"{DEMO_DIR}/fielding.csv")
    else:
        bat = read_gamechanger_csv(batting_file) if batting_file is not None else None
        pit = read_gamechanger_csv(pitching_file) if pitching_file is not None else None
        fld = read_gamechanger_csv(fielding_file) if fielding_file is not None else None

    # ---- Aliases: add to this list as you see actual GC export variants ----
    bat_aliases = {
        "PLAYER": ["PLAYER", "Player", "NAME"],
        "GP": ["GP", "G"],
        "PA": ["PA", "PLATEAPPEARANCES"],
        "AB": ["AB", "ATBATS"],
        "H": ["H", "HITS"],
        "2B": ["2B", "DOUBLES"],
        "3B": ["3B", "TRIPLES"],
        "HR": ["HR", "HOMERUNS"],
        "RBI": ["RBI"],
        "R": ["R", "RUNS"],
        "BB": ["BB", "WALKS"],
        "SO": ["SO", "K", "STRIKEOUTS"],
        "HBP": ["HBP"],
        "SAC": ["SAC", "SACBUNT"],
        "SF": ["SF", "SACFLY"],
        "SB": ["SB", "STOLENBASES"],
        "CS": ["CS", "CAUGHTSTEALING"],
        "TB": ["TB", "TOTALBASES"],
        "XBH": ["XBH", "EXTRABASEHITS"],
        "AVG": ["AVG"],
        "OBP": ["OBP"],
        "SLG": ["SLG"],
        "OPS": ["OPS"],
        "QAB": ["QAB"],
        "PS": ["PS", "PITCHESSEEN"],
    }

    pit_aliases = {
    "PLAYER": ["PLAYER", "Player", "NAME"],
    "GP": ["GP", "G"],
    "GS": ["GS"],
    "IP": ["IP", "IP_1", "INNINGSPITCHED"],
    "BF": ["BF", "BF_1", "BATTERSFACED"],
    "#P": ["#P", "P", "P_1", "PITCHES"],

    # Core pitching stats (handle GC duplicates)
    "H":  ["H_1", "H", "HITS", "HITSALLOWED"],
    "R":  ["R_1", "R", "RUNS"],
    "ER": ["ER_1", "ER", "EARNEDRUNS"],
    "BB": ["BB_1", "BB", "WALKS", "WALKSALLOWED"],
    "SO": ["SO_1", "SO", "K_1", "K", "STRIKEOUTS"],
    "HBP":["HBP_1", "HBP"],
    "HR": ["HR_1", "HR", "HOMERUNS"],

    "WP": ["WP", "WILDPITCHES"],
    "ERA": ["ERA"],
    "WHIP": ["WHIP"],
}

    fld_aliases = {
        "PLAYER": ["PLAYER", "Player", "NAME"],
        "TC": ["TC", "TOTALCHANCES"],
        "PO": ["PO", "PUTOUTS"],
        "A": ["A", "ASSISTS"],
        "E": ["E", "ERRORS"],
        "DP": ["DP", "DOUBLEPLAYS"],
        "TP": ["TP", "TRIPLEPLAYS"],
        "FPCT": ["FPCT", "FIELDINGPERCENTAGE"],
    }

    bat = _rename_by_aliases(bat, bat_aliases) if bat is not None else None
    pit = _rename_by_aliases(pit, pit_aliases) if pit is not None else None
    fld = _rename_by_aliases(fld, fld_aliases) if fld is not None else None

    # Ensure a canonical PLAYER column exists (build from First/Last if needed)
    bat = _ensure_player_name(bat)
    pit = _ensure_player_name(pit)
    fld = _ensure_player_name(fld)

    return bat, pit, fld
def compute_hitting_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or not len(df):
        return df

    df = df.copy()

    # Ensure core columns exist (compute if possible)
    for col in ["PA","AB","BB","SO","H","2B","3B","HR","HBP","SB","CS","TB","XBH","R","RBI"]:
        if col not in df.columns:
            df[col] = 0

    df = _coerce_numeric(df, ["PA","AB","BB","SO","H","2B","3B","HR","HBP","SB","CS","TB","XBH","R","RBI"])

    # Compute TB if missing/zero but components exist
    if (df["TB"] == 0).all():
        # Singles = H - 2B - 3B - HR (clip at 0)
        singles = (df["H"] - df["2B"] - df["3B"] - df["HR"]).clip(lower=0)
        df["TB"] = singles + 2*df["2B"] + 3*df["3B"] + 4*df["HR"]

    # Compute XBH if missing/zero
    if (df["XBH"] == 0).all():
        df["XBH"] = df["2B"] + df["3B"] + df["HR"]

    # If H missing but AVG provided, approximate
    if ("H" in df.columns) and (df["H"] == 0).all() and ("AVG" in df.columns) and ("AB" in df.columns):
        df["H"] = (pd.to_numeric(df["AVG"], errors="coerce").fillna(0) * df["AB"]).round().astype(int)

    # Rates (safe division)
    df["AVG"] = (df["H"] / df["AB"].replace(0, pd.NA)).fillna(0)
    df["OBP"] = ((df["H"] + df["BB"] + df["HBP"]) / df["PA"].replace(0, pd.NA)).fillna(0)
    df["SLG"] = (df["TB"] / df["AB"].replace(0, pd.NA)).fillna(0)
    df["OPS"] = df["OBP"] + df["SLG"]

    df["BB%"] = (df["BB"] / df["PA"].replace(0, pd.NA)).fillna(0)
    df["K%"] = (df["SO"] / df["PA"].replace(0, pd.NA)).fillna(0)
    df["SB%"] = (df["SB"] / (df["SB"] + df["CS"]).replace(0, pd.NA)).fillna(0)
    df["XBH%"] = (df["XBH"] / df["AB"].replace(0, pd.NA)).fillna(0)

    # Round for display
    for c in ["AVG","OBP","SLG","OPS","BB%","K%","SB%","XBH%"]:
        df[c] = df[c].astype(float)

    return df

def compute_pitching_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or not len(df):
        return df

    df = df.copy()
    for col in ["IP","BF","H","R","ER","BB","SO","HR","#P","HBP","WP"]:
        if col not in df.columns:
            df[col] = 0
    df = _coerce_numeric(df, ["IP","BF","H","R","ER","BB","SO","HR","#P","HBP","WP"])

    df["ERA"] = ((df["ER"] * 9) / df["IP"].replace(0, pd.NA)).fillna(0)
    df["WHIP"] = ((df["BB"] + df["H"]) / df["IP"].replace(0, pd.NA)).fillna(0)
    df["K/BB"] = (df["SO"] / df["BB"].replace(0, pd.NA)).fillna(df["SO"])  # if BB=0, show SO as "infinite-ish"
    df["BB/INN"] = (df["BB"] / df["IP"].replace(0, pd.NA)).fillna(0)
    df["K/BF"] = (df["SO"] / df["BF"].replace(0, pd.NA)).fillna(0)

    return df

def compute_fielding_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or not len(df):
        return df

    df = df.copy()
    for col in ["TC","PO","A","E","DP","TP"]:
        if col not in df.columns:
            df[col] = 0
    df = _coerce_numeric(df, ["TC","PO","A","E","DP","TP"])

    df["FPCT"] = ((df["PO"] + df["A"]) / df["TC"].replace(0, pd.NA)).fillna(1.0)
    return df

def top_table(df: pd.DataFrame, metric: str, n: int = 10, ascending: bool = False) -> pd.DataFrame:
    cols = ["PLAYER", metric]
    for extra in ["PA","AB","H","HR","BB","SO","IP","BF","ER","WHIP","OPS","OBP","SLG","TC","E"]:
        if extra in df.columns and extra not in cols:
            cols.append(extra)
    out = df.copy()
    out = out.sort_values(metric, ascending=ascending).head(n)
    return out[cols]

def lineup_suggestions(bat_df: pd.DataFrame, min_pa: int = 10) -> pd.DataFrame:
    df = bat_df.copy()
    if "PA" in df.columns:
        df = df[df["PA"] >= min_pa]

    # Fill missing
    for c in ["OBP","SLG","OPS","K%","SB","BB%"]:
        if c not in df.columns:
            df[c] = 0

    # Scores
    df["top_order_score"] = df["OBP"] - 0.5*df["K%"] + 0.05*df["SB"]
    df["middle_order_score"] = 0.6*df["SLG"] + 0.4*df["OPS"]
    df["bottom_order_score"] = (1 - df["K%"]) + 0.3*df["SB"] + 0.2*df["BB%"]

    top3 = df.sort_values("top_order_score", ascending=False).head(3).copy()
    mid3 = df[~df["PLAYER"].isin(top3["PLAYER"])].sort_values("middle_order_score", ascending=False).head(3).copy()
    bottom3 = df[~df["PLAYER"].isin(list(top3["PLAYER"]) + list(mid3["PLAYER"]))].sort_values("bottom_order_score", ascending=False).head(3).copy()

    def _slot_rows(slot_start, group, why):
        rows=[]
        for i, (_, r) in enumerate(group.iterrows()):
            rows.append({
                "Slot": slot_start+i,
                "PLAYER": r["PLAYER"],
                "Why": why,
                "OBP": float(r.get("OBP", 0)),
                "SLG": float(r.get("SLG", 0)),
                "OPS": float(r.get("OPS", 0)),
                "K%": float(r.get("K%", 0)),
                "SB": int(r.get("SB", 0)),
                "PA": int(r.get("PA", 0)) if "PA" in r else 0,
            })
        return rows

    rows=[]
    rows += _slot_rows(1, top3, "Top-of-order (OBP + low K% + speed)")
    rows += _slot_rows(4, mid3, "Middle-order (SLG/OPS impact)")
    rows += _slot_rows(7, bottom3, "Bottom (contact/speed/turnover)")

    return pd.DataFrame(rows).sort_values("Slot")
