# NTBSC Roster Planner (Full Enhanced)

This package contains the full enhanced Streamlit application for planning NTBSC rosters.

## Features
- Password-gated admin tabs (password: `NTBSC_jy`).
- Consultants, defaults, duties editing with validated dates, dropdowns.
- Add duties by date range.
- Multi-month sustainable roster building and storage under `data/rosters`.
- Conflicts detection, highlighting, editable roster.
- Overview with consultant filter.
- Calendar and overview list toggle by month/year (independent of sidebar settings).
- Exports to CSV and Excel.
- Singapore public holidays hardwired.

## Usage
1. Install dependencies:
   ```bash
   pip install streamlit pandas openpyxl
   ```
2. Run the app:
   ```bash
   streamlit run app.py
   ```
3. Default data is stored under the `data/` folder.

## Notes
- Admin features are available only with the password.
- Public users can view Overview and Exports.
