# HS Baseball Analytics — Starter App (Streamlit)

## What this is
A beginner-friendly Streamlit app you can use to learn Python while building a real baseball analytics tool.

- **Demo MLB mode**: uses built-in demo CSVs (public-safe)
- **Upload mode**: upload your GameChanger exports (batting/pitching/fielding season totals)

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Folder structure
- `app.py` — Streamlit UI
- `utils.py` — loaders + metric calculations
- `demo/` — demo CSVs used for Demo MLB mode

## Notes on GameChanger exports
GameChanger exports can vary slightly by column names. This starter app uses a normalization + alias mapping approach.
If your export uses different headers, add aliases in `utils.py` under `bat_aliases`, `pit_aliases`, `fld_aliases`.

## GameChanger export note
This starter app can ingest GameChanger exports that include a section/glossary header block. If your export has rows like `Batting`/`Pitching` and `Glossary`, the loader will auto-detect the true header row.
