# NTBSC Roster Planner (Enhanced)

This package contains the enhanced Streamlit application for planning NTBSC rosters.

## Features
- Admin-only editing tabs (password protected).
- Validated dates in editors; dropdowns for `kind` and `session`.
- Add duties by date range (start → end).
- Multi-month roster building and storage under `data/rosters/YYYY-MM.csv`.
- Smart weekly preceptor selection (fixed mode) with smoothing.
- Conflicts detection and highlighted roster view.
- Editable roster grid with save.
- Overview with consultant filter.
- Calendar + Overview month/year toggle (affects both views), with public holidays shown using a 🎉 icon and shaded cells.
- Exports to CSV and Excel.
- Singapore public holidays hardwired (2024–2026).

## Usage
1. Install dependencies:
   ```bash
   pip install streamlit pandas openpyxl
   ```
2. Run the app:
   ```bash
   streamlit run app.py
   ```
3. Data persists under the `data/` folder. Monthly rosters are saved in `data/rosters/YYYY-MM.csv`.

## Notes
- Admin features are gated by a password (not shown here). Public users can view Overview and Exports.
