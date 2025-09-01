# NTBSC Roster Planner (Streamlit)

This app generates clinic rosters for NTBSC with rule-based blocking and a printable overview.
It respects:
- Leave blocks
- Ward rounds (block AM)
- Blues (block PM)
- Woodlands (full-day block incl. precepting)
- Suma auto-assigned to Room 28 on 2nd & 4th Friday PM
- Paediatrics (visiting consultants) in Room 29
- Only certain consultants can precept
- A person running Room 18/28 cannot be the Preceptor in the same session
- Preceptor is kept the same across a week where possible
- Public holidays block all duties

## Quick start

```bash
python -m venv .venv
. .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Data model

CSV files live in `data/` and can be edited in-app:
- consultants.csv — name, active (Y/N), preceptor_eligible (Y/N)
- preceptor_defaults.csv — weekday, session, default_preceptor
- settings.json — month, year, week_preceptor_mode (fixed or per_session)
- leave.csv — consultant, date, session(AM/PM/Full), type, notes
- ward_rounds.csv — consultant, date, notes
- blues.csv — consultant, date, notes
- woodlands.csv — consultant, date, session(AM/PM/Full), notes
- paeds.csv — consultant, date, session(AM/PM), notes  (visiting; they are not in the pool)
- juniors.csv — date, junior_name
- public_holidays.csv — date, name
- special_slots.csv — date, session, room_name, assignee (optional free-form extras)

## Export
- Download Excel: roster, list and calendar views, flat export.
- Download CSV: roster flat file.
