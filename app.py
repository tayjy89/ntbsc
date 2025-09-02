
import streamlit as st
import pandas as pd
import json
from io import BytesIO
from pathlib import Path
from datetime import date, datetime, timedelta
import calendar

"""
NTBSC Roster Planner — consolidated single-file Streamlit app
Dependencies: streamlit, pandas, openpyxl
Persistence: plain files under ./data and ./data/rosters

Key features:
- Password-gated admin tabs (password: NTBSC_jy).
- Unified duties.csv for leave/ward/blues/woodlands with month filter & editing.
- Multi-month planning with per-month roster persistence in data/rosters/YYYY-MM.csv.
- Smarter weekly preceptor picking in "fixed" mode with availability scoring & smoothing.
- Conflicts detection and highlighted roster view.
- Editable roster grid with save.
- Overview list + calendar; overview supports consultant filter.
- Singapore public holidays hardwired for 2024–2026.
"""

APP_TITLE = "NTBSC Roster Planner"
DATA_DIR = Path("data")
ROSTER_DIR = DATA_DIR / "rosters"
DATA_DIR.mkdir(exist_ok=True)
ROSTER_DIR.mkdir(exist_ok=True)

ADMIN_PASSWORD = "NTBSC_jy"

# -------------------- Hardwired Singapore Public Holidays --------------------
def sg_public_holidays(year: int):
    # Note: keep this table current; extend years as needed.
    fixed = {
        2024: [
            "2024-01-01", "2024-02-10", "2024-02-11", "2024-03-29", "2024-04-10",
            "2024-05-01", "2024-05-22", "2024-06-17", "2024-08-09", "2024-10-31", "2024-12-25",
        ],
        2025: [
            "2025-01-01", "2025-01-29", "2025-01-30", "2025-04-18", "2025-05-01",
            "2025-05-12", "2025-06-06", "2025-08-09", "2025-10-20", "2025-12-25",
        ],
        2026: [
            "2026-01-01", "2026-02-17", "2026-02-18", "2026-04-03", "2026-05-01",
            "2026-05-22", "2026-06-26", "2026-08-09", "2026-11-09", "2026-12-25",
        ],
    }
    return set(pd.to_datetime(fixed.get(year, []), errors="coerce").dropna().date)

# -------------------- IO helpers --------------------
def ensure_seed_files():
    # consultants
    cfp = DATA_DIR / "consultants.csv"
    if not cfp.exists():
        pd.DataFrame([
            {"name": "Jun Yang", "active(Y/N)": "Y", "preceptor_eligible(Y/N)": "Y"},
            {"name": "Deborah", "active(Y/N)": "Y", "preceptor_eligible(Y/N)": "Y"},
            {"name": "Wilnard", "active(Y/N)": "Y", "preceptor_eligible(Y/N)": "Y"},
            {"name": "Suma", "active(Y/N)": "Y", "preceptor_eligible(Y/N)": "N"},
            {"name": "Matthias", "active(Y/N)": "Y", "preceptor_eligible(Y/N)": "N"},
            {"name": "Khin", "active(Y/N)": "Y", "preceptor_eligible(Y/N)": "Y"},
            {"name": "Hoi Wah", "active(Y/N)": "Y", "preceptor_eligible(Y/N)": "N"},
        ]).to_csv(cfp, index=False)

    # room28 defaults
    dfp = DATA_DIR / "room28_defaults.csv"
    if not dfp.exists():
        pd.DataFrame([
            {"weekday": "Mon", "session": "AM", "consultant": "Khin"},
            {"weekday": "Mon", "session": "PM", "consultant": "Jun Yang"},
            {"weekday": "Tue", "session": "AM", "consultant": "Matthias"},
            {"weekday": "Tue", "session": "PM", "consultant": "Wilnard"},
            {"weekday": "Wed", "session": "AM", "consultant": "Hoi Wah"},
            {"weekday": "Wed", "session": "PM", "consultant": "Deborah"},
            {"weekday": "Thu", "session": "AM", "consultant": ""},
            {"weekday": "Thu", "session": "PM", "consultant": "Khin"},
            {"weekday": "Fri", "session": "AM", "consultant": "Hoi Wah"},
            {"weekday": "Fri", "session": "PM", "consultant": "Khin"},
        ]).to_csv(dfp, index=False)

    # settings
    sfp = DATA_DIR / "settings.json"
    if not sfp.exists():
        today = date.today()
        json.dump({"month": today.month, "year": today.year, "week_preceptor_mode": "fixed"}, open(sfp, "w"))

    # unified duties
    du = DATA_DIR / "duties.csv"
    if not du.exists():
        pd.DataFrame(columns=["consultant", "date", "session", "kind", "notes"]).to_csv(du, index=False)

    # juniors / paeds / specials
    for fname, cols in [
        ("juniors.csv", ["date", "junior_name"]),
        ("paeds.csv", ["consultant", "date", "session", "notes"]),
        ("special_slots.csv", ["date", "session", "room_name", "assignee"]),
    ]:
        fp = DATA_DIR / fname
        if not fp.exists():
            pd.DataFrame(columns=cols).to_csv(fp, index=False)


def read_csv(name, parse_dates=("date",)):
    fp = DATA_DIR / name
    if not fp.exists():
        # create empty with guessed columns if needed
        if name == "duties.csv":
            pd.DataFrame(columns=["consultant", "date", "session", "kind", "notes"]).to_csv(fp, index=False)
        else:
            pd.DataFrame().to_csv(fp, index=False)
    df = pd.read_csv(fp, dtype=str).fillna("")
    for col in parse_dates:
        if col in df.columns:
            def _parse(x):
                x = str(x).strip()
                if not x:
                    return ""
                try:
                    return pd.to_datetime(x).date()
                except Exception:
                    return ""
            df[col] = df[col].map(_parse)
    return df


def write_csv(name, df):
    df2 = df.copy()
    for c in df2.columns:
        df2[c] = df2[c].map(lambda x: x.isoformat() if hasattr(x, "isoformat") else x)
    df2.to_csv(DATA_DIR / name, index=False)


def read_settings():
    return json.load(open(DATA_DIR / "settings.json"))


def write_settings(s):
    json.dump(s, open(DATA_DIR / "settings.json", "w"))

# -------------------- Date helpers --------------------

def weekdays_in_month(year: int, month: int):
    cal = calendar.Calendar(firstweekday=calendar.MONDAY)
    out = []
    for d in cal.itermonthdates(year, month):
        if d.month == month and d.weekday() < 5:
            out.append((d, ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][d.weekday()]))
    return out


def second_and_fourth_fridays(year: int, month: int):
    fridays = [d for d, wd in weekdays_in_month(year, month) if wd == "Fri"]
    fridays = [f if isinstance(f, date) else f[0] for f in fridays]
    out = []
    if len(fridays) >= 2:
        out.append(fridays[1])
    if len(fridays) >= 4:
        out.append(fridays[3])
    return out

# -------------------- Blocks / Lookups --------------------

def _room28_default_for(wd, session, defaults_df):
    m = defaults_df[(defaults_df["weekday"] == wd) & (defaults_df["session"] == session)]
    return (m.iloc[0]["consultant"].strip() if not m.empty else "")


def _is_ph(d: date, year_ph_set: set):
    return d in year_ph_set


def _has_leave(consultant, d, session, duties_df):
    rows = duties_df[(duties_df["consultant"] == consultant) & (duties_df["date"] == d) & (duties_df["kind"] == "leave")]
    for _, r in rows.iterrows():
        s = str(r.get("session", "")).upper()
        if s in ("FULL", "FD", "WHOLE", "ALL", "DAY", "FULL DAY", ""):
            return True
        if s == session.upper():
            return True
    return False


def _has_woodlands(consultant, d, duties_df):
    return not duties_df[(duties_df["consultant"] == consultant) & (duties_df["date"] == d) & (duties_df["kind"] == "woodlands")].empty


def _has_ward_am(consultant, d, duties_df):
    return not duties_df[(duties_df["consultant"] == consultant) & (duties_df["date"] == d) & (duties_df["kind"] == "ward")].empty


def _has_blues_pm(consultant, d, duties_df):
    return not duties_df[(duties_df["consultant"] == consultant) & (duties_df["date"] == d) & (duties_df["kind"] == "blues")].empty

# -------------------- Roster build --------------------

def build_roster(
    year,
    month,
    week_preceptor_mode,
    consultants_df,
    defaults_df,
    juniors_df,
    year_ph_set,
    duties_df,
    paeds_df,
    special_df,
):
    consultants_list = consultants_df[consultants_df["active(Y/N)"] == "Y"]["name"].tolist()
    preceptor_eligible = set(
        consultants_df[
            (consultants_df["active(Y/N)"] == "Y") & (consultants_df["preceptor_eligible(Y/N)"] == "Y")
        ]["name"].tolist()
    )

    rows = []
    day_slots = weekdays_in_month(year, month)
    juniors_map = {r["date"]: r["junior_name"] for _, r in juniors_df.iterrows() if r["date"] != ""}
    fri_2_4 = second_and_fourth_fridays(year, month)

    # Smart weekly preceptor picking for "fixed"
    week_monday_to_preceptor = {}
    if week_preceptor_mode == "fixed":
        fri_2_4_month = fri_2_4

        def _is_available_for_session(cand, d, sess):
            if _is_ph(d, year_ph_set):
                return False
            wd = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][d.weekday()]
            r28 = _room28_default_for(wd, sess, defaults_df)
            if wd == "Fri" and sess == "PM" and d in fri_2_4_month:
                r28 = "Suma"
            if r28 == cand:
                return False
            if _has_woodlands(cand, d, duties_df):
                return False
            if _has_leave(cand, d, sess, duties_df):
                return False
            if sess == "AM" and _has_ward_am(cand, d, duties_df):
                return False
            if sess == "PM" and _has_blues_pm(cand, d, duties_df):
                return False
            return True

        dates = [d for d, _ in day_slots]
        if dates:
            start = min(dates)
            start_monday = start - timedelta(days=start.weekday())
            end = max(dates)
            end_sunday = end + timedelta(days=(6 - end.weekday()))
            prior_counts = {c: 0 for c in preceptor_eligible}
            cur = start_monday
            while cur <= end_sunday:
                week_workdays = [cur + timedelta(days=i) for i in range(5) if (cur + timedelta(days=i)).month == month]
                if not week_workdays:
                    week_monday_to_preceptor[cur] = None
                    cur += timedelta(days=7)
                    continue
                best, best_score = None, -1
                for cand in preceptor_eligible:
                    score = 0
                    for d in week_workdays:
                        for sess in ("AM", "PM"):
                            if _is_available_for_session(cand, d, sess):
                                score += 1
                    if score > best_score:
                        best, best_score = cand, score
                    elif score == best_score and best is not None:
                        if prior_counts.get(cand, 0) < prior_counts.get(best, 0):
                            best = cand
                        elif prior_counts.get(cand, 0) == prior_counts.get(best, 0) and cand < best:
                            best = cand
                week_monday_to_preceptor[cur] = best if best_score > 0 else None
                if best and best_score > 0:
                    prior_counts[best] += 1
                cur += timedelta(days=7)

    # Build row skeleton
    for d, wd in day_slots:
        for session in ("AM", "PM"):
            notes = []
            room18 = juniors_map.get(d, "TBD")

            # Room 28 default
            room28 = _room28_default_for(wd, session, defaults_df)
            if (wd == "Fri") and (session == "PM") and (d in fri_2_4):
                room28 = "Suma"

            # Room 29 (paeds)
            room29 = ""
            for _, r in paeds_df[paeds_df["date"] == d].iterrows():
                if str(r.get("session", "")).upper() in ("", session.upper()):
                    room29 = r["consultant"]

            # Specials -> put others into notes; fill R29 if named
            specials_today = special_df[
                (special_df["date"] == d)
                & (special_df["session"].str.upper().isin(["", session.upper()]))
            ] if not special_df.empty else pd.DataFrame()
            for _, r in specials_today.iterrows():
                rn, assignee = r.get("room_name", ""), r.get("assignee", "")
                if str(rn).strip().lower() in ("room29", "room 29", "r29", "29"):
                    if not room29:
                        room29 = assignee
                else:
                    if rn or assignee:
                        notes.append(f"{rn}: {assignee}")

            # Preceptor default
            preceptor = ""
            if week_preceptor_mode == "fixed":
                monday = d - timedelta(days=d.weekday())
                candidate = week_monday_to_preceptor.get(monday, "")
                preceptor = candidate or ""
            else:
                preceptor = ""

            rows.append(
                {
                    "date": d,
                    "weekday": wd,
                    "session": session,
                    "Room 18": room18,
                    "Room 28": room28,
                    "Preceptor": preceptor,
                    "Room 29": room29,
                    "notes": "; ".join(notes) if notes else "",
                }
            )

    roster = pd.DataFrame(rows)

    # Apply blocking rules
    def _wipe(row_ref, cols):
        for c in cols:
            row_ref[c] = ""

    for i in roster.index:
        r = roster.loc[i]
        d, session = r["date"], r["session"]

        # PH
        if _is_ph(d, year_ph_set):
            _wipe(roster.loc[i], ["Room 18", "Room 28", "Preceptor", "Room 29"])
            roster.loc[i, "notes"] = (roster.loc[i, "notes"] + "; Public Holiday").strip("; ")
            continue

        # Leave / Woodlands affect Room28 & Preceptor
        for col in ("Room 28", "Preceptor"):
            name = r[col]
            if not name:
                continue
            if _has_woodlands(name, d, duties_df):
                roster.loc[i, col] = ""
                roster.loc[i, "notes"] = (roster.loc[i, "notes"] + f"; Woodlands({name})").strip("; ")
            elif _has_leave(name, d, session, duties_df):
                roster.loc[i, col] = ""
                roster.loc[i, "notes"] = (roster.loc[i, "notes"] + f"; Leave({name})").strip("; ")

        # Ward AM / Blues PM
        for col in ("Room 28", "Preceptor"):
            name = roster.loc[i, col]
            if name and session == "AM" and _has_ward_am(name, d, duties_df):
                roster.loc[i, col] = ""
                roster.loc[i, "notes"] = (roster.loc[i, "notes"] + f"; WardRounds({name})").strip("; ")
            if name and session == "PM" and _has_blues_pm(name, d, duties_df):
                roster.loc[i, col] = ""
                roster.loc[i, "notes"] = (roster.loc[i, "notes"] + f"; Blues({name})").strip("; ")

        # Exclusivity: same person cannot be Room28 & Preceptor
        if roster.loc[i, "Room 28"] and roster.loc[i, "Preceptor"] and roster.loc[i, "Room 28"] == roster.loc[i, "Preceptor"]:
            roster.loc[i, "Preceptor"] = ""

    return roster

# -------------------- Conflicts & styling --------------------

def conflicts(roster_df, consultants_df, year_ph_set, duties_df):
    rows = []
    preceptor_ok = set(
        consultants_df[
            (consultants_df["active(Y/N)"] == "Y") & (consultants_df["preceptor_eligible(Y/N)"] == "Y")
        ]["name"].tolist()
    )

    def _add(idx, d, sess, field, name, reason):
        rows.append(
            {
                "row_index": idx,
                "date": d,
                "session": sess,
                "field": field,
                "name": name,
                "conflict_reason": reason,
            }
        )

    for idx, r in roster_df.reset_index().iterrows():
        d, sess = r["date"], r["session"]

        # PH scheduling
        if d in year_ph_set:
            for field in ["Room 18", "Room 28", "Preceptor", "Room 29"]:
                if str(r[field]).strip():
                    _add(r["index"], d, sess, field, r[field], "Scheduled on public holiday")

        # Duty violations
        for field in ["Room 28", "Preceptor"]:
            name = str(r[field]).strip()
            if not name:
                continue
            if any(
                (
                    row["consultant"] == name
                    and row["date"] == d
                    and row["kind"] == "leave"
                    and (str(row["session"]).upper() in ("FULL", "FD", "") or str(row["session"]).upper() == sess)
                )
                for _, row in duties_df.iterrows()
            ):
                _add(r["index"], d, sess, field, name, "On leave")
            if any((row["consultant"] == name and row["date"] == d and row["kind"] == "woodlands") for _, row in duties_df.iterrows()):
                _add(r["index"], d, sess, field, name, "Woodlands day")
            if sess == "AM" and any((row["consultant"] == name and row["date"] == d and row["kind"] == "ward") for _, row in duties_df.iterrows()):
                _add(r["index"], d, sess, field, name, "Ward rounds (AM)")
            if sess == "PM" and any((row["consultant"] == name and row["date"] == d and row["kind"] == "blues") for _, row in duties_df.iterrows()):
                _add(r["index"], d, sess, field, name, "Blues (PM)")

        # Preceptor eligibility
        name = str(r["Preceptor"]).strip()
        if name and name not in preceptor_ok:
            _add(r["index"], d, sess, "Preceptor", name, "Not preceptor-eligible")

        # Exclusivity conflicts
        if str(r["Room 28"]).strip() and str(r["Preceptor"]).strip() and r["Room 28"] == r["Preceptor"]:
            _add(r["index"], d, sess, "Preceptor", r["Preceptor"], "Same as Room 28 (exclusivity)")
        if str(r["Room 18"]).strip() and str(r["Preceptor"]).strip() and r["Room 18"] == r["Preceptor"]:
            _add(r["index"], d, sess, "Preceptor", r["Preceptor"], "Same as Room 18 (exclusivity)")

    return pd.DataFrame(rows)


def style_roster(roster_df: pd.DataFrame, conf_df: pd.DataFrame):
    # Build a per-cell style DataFrame highlighting conflicts & empty slots
    styles = pd.DataFrame("", index=roster_df.index, columns=roster_df.columns)
    conflict_map = {(r["date"], r["session"], r["field"]): True for _, r in conf_df.iterrows()}

    for i, r in roster_df.iterrows():
        for col in ["Room 18", "Room 28", "Preceptor", "Room 29"]:
            if (r["date"], r["session"], col) in conflict_map:
                styles.at[i, col] = "background-color: #ffe6e6; border: 1px solid #ff4d4f;"
            elif (col == "Room 18" and r[col] == "TBD") or (col != "Room 18" and r[col] == ""):
                styles.at[i, col] = "background-color: #fff7e6;"
        if r["session"] == "PM":
            for col in roster_df.columns:
                styles.at[i, col] += " border-top: 1px solid #f0f0f0;"

    base_style = {"border": "1px solid #eee", "font-size": "12px", "padding": "4px 6px"}
    return (
        roster_df.style
        .set_properties(**base_style)
        .apply(lambda _: styles, axis=None)
    )

# -------------------- Overview & Exports --------------------

def overview_list(roster_df: pd.DataFrame):
    def _compact(day_df):
        rec = {"date": day_df.iloc[0]["date"], "weekday": day_df.iloc[0]["weekday"]}
        for _, r in day_df.iterrows():
            sfx = r["session"]
            rec[f"Room18_{sfx}"] = r["Room 18"]
            rec[f"Room28_{sfx}"] = r["Room 28"]
            rec[f"Preceptor_{sfx}"] = r["Preceptor"]
            rec[f"Room29_{sfx}"] = r["Room 29"]
        rec["notes"] = "; ".join([x for x in day_df["notes"].tolist() if x])
        return rec

    out = []
    for d, grp in roster_df.groupby("date", sort=True):
        out.append(_compact(grp.sort_values("session")))
    return pd.DataFrame(out).sort_values("date")


def calendar_markdown(roster_df: pd.DataFrame, year_ph_set: set):
    if roster_df.empty:
        return "_No roster_"
    year = roster_df["date"].iloc[0].year
    month = roster_df["date"].iloc[0].month
    cal = calendar.Calendar(firstweekday=calendar.MONDAY)
    weeks = cal.monthdatescalendar(year, month)
    idx = {(r["date"], r["session"]): r for _, r in roster_df.iterrows()}

    md = []
    md.append("| Mon | Tue | Wed | Thu | Fri | Sat | Sun |")
    md.append("|-----|-----|-----|-----|-----|-----|-----|")
    for week in weeks:
        cells = []
        for d in week:
            if d.month != month:
                cells.append(" ")
                continue
            am = idx.get((d, "AM"))
            pm = idx.get((d, "PM"))
            ph_badge = " **PH**" if d in year_ph_set else ""

            def fmt(slot):
                if slot is None:
                    return ""
                a = []
                if slot["Room 28"]:
                    a.append(f"R28: {slot['Room 28']}")
                if slot["Preceptor"]:
                    a.append(f"P: {slot['Preceptor']}")
                if slot["Room 29"]:
                    a.append(f"R29: {slot['Room 29']}")
                return "; ".join(a)

            cell = f"**{d.day}**{ph_badge}<br/>AM: {fmt(am)}<br/>PM: {fmt(pm)}"
            cells.append(cell)
        md.append("| " + " | ".join(cells) + " |")
    return "\n".join(md)


def excel_bytes(roster_df: pd.DataFrame):
    from openpyxl import Workbook
    from openpyxl.utils.dataframe import dataframe_to_rows

    wb = Workbook()
    ws1 = wb.active
    ws1.title = "Roster"
    for r in dataframe_to_rows(roster_df, index=False, header=True):
        ws1.append(r)

    ws2 = wb.create_sheet("Overview_List")
    ov = overview_list(roster_df)
    for r in dataframe_to_rows(ov, index=False, header=True):
        ws2.append(r)

    ws3 = wb.create_sheet("Export_Flat")
    flat = roster_df.copy()
    for r in dataframe_to_rows(flat, index=False, header=True):
        ws3.append(r)

    bio = BytesIO()
    wb.save(bio)
    bio.seek(0)
    return bio.getvalue()

# -------------------- Streamlit UI --------------------

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    ensure_seed_files()

    # Settings & month selection (supports multi-month planning)
    settings = read_settings()
    with st.sidebar:
        st.header("Settings")
        col1, col2 = st.columns(2)
        month = col1.selectbox("Month", list(range(1, 13)), index=settings.get("month", date.today().month) - 1, key="month")
        year = col2.number_input("Year", min_value=2020, max_value=2100, value=settings.get("year", date.today().year), step=1, key="year")
        mode = st.radio(
            "Preceptor assignment",
            options=["fixed", "per_session"],
            index=0 if settings.get("week_preceptor_mode", "fixed") == "fixed" else 1,
            key="mode",
        )
        admin_pw = st.text_input("Admin password (leave blank for public view)", type="password", key="pw")
        unlocked = admin_pw == ADMIN_PASSWORD
        if st.button("Save Settings", key="save_settings"):
            write_settings({"month": int(month), "year": int(year), "week_preceptor_mode": mode})
            st.success("Settings saved.")
        st.caption("Public view shows only Overview & Exports. Enter password to unlock editors.")

    # Base inputs
    consultants_df = read_csv("consultants.csv", parse_dates=())
    defaults_df = read_csv("room28_defaults.csv", parse_dates=())
    juniors_df = read_csv("juniors.csv")
    duties_df = read_csv("duties.csv")
    paeds_df = read_csv("paeds.csv")
    special_df = read_csv("special_slots.csv")

    year_ph = sg_public_holidays(int(year))

    # Build or load roster for selected month
    roster_fp = ROSTER_DIR / f"{int(year)}-{int(month):02d}.csv"
    if roster_fp.exists():
        roster_df = pd.read_csv(roster_fp, dtype=str)
        roster_df["date"] = pd.to_datetime(roster_df["date"]).dt.date
    else:
        roster_df = build_roster(
            int(year),
            int(month),
            mode,
            consultants_df,
            defaults_df,
            juniors_df,
            year_ph,
            duties_df,
            paeds_df,
            special_df,
        )

    # Tabs (gated)
    if unlocked:
        tabs = st.tabs(["Consultants & Defaults", "Duties (All-in-one)", "Build / Edit Roster", "Overview", "Exports"])
    else:
        tabs = st.tabs(["Overview", "Exports"])

    # ---------------- Admin Tabs ----------------
    if unlocked:
        with tabs[0]:
            st.subheader("Consultants")
            cedit = st.data_editor(consultants_df, num_rows="dynamic", width="stretch", key="edit_consultants")
            if st.button("Save Consultants", key="save_consultants"):
                write_csv("consultants.csv", cedit)
                st.success("Saved consultants.csv")

            st.divider()
            st.subheader("Room 28 Defaults (fixed senior assignments)")
            ddef = st.data_editor(defaults_df, num_rows="dynamic", width="stretch", key="edit_room28_defaults")
            if st.button("Save Room28 Defaults", key="save_room28_defaults"):
                write_csv("room28_defaults.csv", ddef)
                st.success("Saved room28_defaults.csv")

        with tabs[1]:
            st.subheader("Duties (leave, ward rounds, blues, woodlands)")
            # Month filter for duties
            month_dates = pd.to_datetime(duties_df["date"], errors="coerce")
            duties_df["_month"] = month_dates.dt.month
            duties_df["_year"] = month_dates.dt.year
            filt = (duties_df["_month"] == int(month)) & (duties_df["_year"] == int(year))
            month_view = duties_df[filt].drop(columns=["_month", "_year"], errors="ignore")
            st.caption("Kinds: leave / ward / blues / woodlands. Session: AM, PM, Full (blank = Full).")
            dedit = st.data_editor(month_view, num_rows="dynamic", width="stretch", key="edit_duties")
            if st.button("Save Duties (for this month)", key="save_duties"):
                keep = ~filt
                remaining = duties_df[keep].drop(columns=["_month", "_year"], errors="ignore")
                write_csv("duties.csv", pd.concat([remaining, dedit], ignore_index=True))
                st.success("Saved duties.csv for selected month.")

            st.divider()
            st.subheader("Juniors & Paeds & Specials")
            jedit = st.data_editor(juniors_df, num_rows="dynamic", width="stretch", key="edit_juniors")
            if st.button("Save Juniors", key="save_juniors"):
                write_csv("juniors.csv", jedit)
                st.success("Saved juniors.csv")
            pedit = st.data_editor(paeds_df, num_rows="dynamic", width="stretch", key="edit_paeds")
            if st.button("Save Paeds", key="save_paeds"):
                write_csv("paeds.csv", pedit)
                st.success("Saved paeds.csv")
            sedit = st.data_editor(special_df, num_rows="dynamic", width="stretch", key="edit_specials")
            if st.button("Save Specials", key="save_specials"):
                write_csv("special_slots.csv", sedit)
                st.success("Saved special_slots.csv")

        with tabs[2]:
            st.subheader("Build / Edit Roster")
            if st.button("Rebuild from inputs for this month", key="rebuild_roster"):
                roster_df = build_roster(
                    int(year),
                    int(month),
                    mode,
                    read_csv("consultants.csv", parse_dates=()),
                    read_csv("room28_defaults.csv", parse_dates=()),
                    read_csv("juniors.csv"),
                    sg_public_holidays(int(year)),
                    read_csv("duties.csv"),
                    read_csv("paeds.csv"),
                    read_csv("special_slots.csv"),
                )
                st.success("Roster regenerated. You can edit below and save.")

            # Conflicts on current roster view
            conf_df = conflicts(
                roster_df,
                read_csv("consultants.csv", parse_dates=()),
                sg_public_holidays(int(year)),
                read_csv("duties.csv"),
            )

            # Highlighted static view
            st.markdown("**Roster (highlighted)**")
            styled = style_roster(roster_df, conf_df)
            st.markdown(styled.to_html(), unsafe_allow_html=True)

            st.markdown("**Edit roster (inline)**")
            redit = st.data_editor(roster_df, num_rows="dynamic", width="stretch", key="edit_roster")
            if st.button("Save Roster for this Month", key="save_roster"):
                df2 = redit.copy()
                df2["date"] = pd.to_datetime(df2["date"]).dt.date
                df2.to_csv(ROSTER_DIR / f"{int(year)}-{int(month):02d}.csv", index=False)
                st.success(f"Saved to rosters/{int(year)}-{int(month):02d}.csv")

            st.markdown("**Conflicts**")
            st.dataframe(conf_df, width="stretch", height=240)

    # ---------------- Public Tabs ----------------
    with tabs[-2 if unlocked else 0]:
        st.subheader("Overview")
        # Ensure we have the roster_df for the view
        if not (ROSTER_DIR / f"{int(year)}-{int(month):02d}.csv").exists():
            roster_df = build_roster(
                int(year), int(month), mode,
                consultants_df, defaults_df, juniors_df, year_ph, duties_df, paeds_df, special_df,
            )
        ov = overview_list(roster_df)

        # Filter by consultant
        all_people = sorted(set(consultants_df["name"].tolist()))
        who = st.multiselect(
            "Filter by consultant(s) (matches any of Room18/Room28/Preceptor/Room29)",
            options=all_people,
            default=[],
            key="ov_filter_who",
        )
        if who:
            mask = pd.Series(False, index=ov.index)
            cols_to_search = [c for c in ov.columns if any(x in c for x in ["Room18_", "Room28_", "Preceptor_", "Room29_"])]
            for person in who:
                m = ov[cols_to_search].apply(lambda s: s.astype(str).str.contains(fr"\b{person}\b", na=False))
                mask = mask | m.any(axis=1)
            ovf = ov[mask]
        else:
            ovf = ov
        st.dataframe(ovf, width="stretch", height=480)

        st.subheader("Calendar")
        md = calendar_markdown(roster_df, year_ph)
        st.markdown(md, unsafe_allow_html=True)

    with tabs[-1]:
        st.subheader("Exports")
        if not (ROSTER_DIR / f"{int(year)}-{int(month):02d}.csv").exists():
            roster_df = build_roster(
                int(year), int(month), mode,
                consultants_df, defaults_df, juniors_df, year_ph, duties_df, paeds_df, special_df,
            )
        csv_bytes = roster_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Roster CSV",
            data=csv_bytes,
            file_name=f"Roster_{int(year)}-{int(month):02d}.csv",
            mime="text/csv",
            key="dl_csv",
        )
        xlsx = excel_bytes(roster_df)
        st.download_button(
            "Download Excel Workbook",
            data=xlsx,
            file_name=f"NTBSC_Roster_{int(year)}-{int(month):02d}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_xlsx",
        )
        st.caption("Workbook contains: Roster, Overview_List, Export_Flat")


if __name__ == "__main__":
    main()
