
import streamlit as st
import pandas as pd
import json
from io import BytesIO
from pathlib import Path
from datetime import datetime, date, timedelta
import calendar

APP_TITLE = "NTBSC Roster Planner"
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# -------------------- Utilities: IO --------------------
def ensure_seed_files():
    consultants_fp = DATA_DIR / "consultants.csv"
    if not consultants_fp.exists():
        pd.DataFrame([
            {"name": "Jun Yang", "active(Y/N)": "Y", "preceptor_eligible(Y/N)": "Y"},
            {"name": "Deborah", "active(Y/N)": "Y", "preceptor_eligible(Y/N)": "Y"},
            {"name": "Wilnard", "active(Y/N)": "Y", "preceptor_eligible(Y/N)": "Y"},
            {"name": "Suma", "active(Y/N)": "Y", "preceptor_eligible(Y/N)": "N"},
            {"name": "Matthias", "active(Y/N)": "Y", "preceptor_eligible(Y/N)": "N"},
            {"name": "Khin", "active(Y/N)": "Y", "preceptor_eligible(Y/N)": "Y"},
            {"name": "Hoi Wah", "active(Y/N)": "Y", "preceptor_eligible(Y/N)": "N"},
        ]).to_csv(consultants_fp, index=False)

    defaults_fp = DATA_DIR / "room28_defaults.csv"
    if not defaults_fp.exists():
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
        ]).to_csv(defaults_fp, index=False)

    settings_fp = DATA_DIR / "settings.json"
    if not settings_fp.exists():
        today = date.today()
        json.dump({"month": today.month, "year": today.year, "week_preceptor_mode": "fixed"},
                  open(settings_fp, "w"))

    for fname, cols in [
        ("leave.csv", ["consultant","date","session","type","notes"]),
        ("ward_rounds.csv", ["consultant","date","notes"]),
        ("blues.csv", ["consultant","date","notes"]),
        ("woodlands.csv", ["consultant","date","session","notes"]),
        ("paeds.csv", ["consultant","date","session","notes"]),
        ("juniors.csv", ["date","junior_name"]),
        ("public_holidays.csv", ["date","name"]),
        ("special_slots.csv", ["date","session","room_name","assignee"]),
    ]:
        fp = DATA_DIR / fname
        if not fp.exists():
            pd.DataFrame(columns=cols).to_csv(fp, index=False)

def read_csv(name, parse_dates=("date",)):
    fp = DATA_DIR / name
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
    (DATA_DIR / name).parent.mkdir(exist_ok=True, parents=True)
    df2 = df.copy()
    for c in df2.columns:
        df2[c] = df2[c].map(lambda x: x.isoformat() if hasattr(x, "isoformat") else x)
    df2.to_csv(DATA_DIR / name, index=False)

def read_settings():
    return json.load(open(DATA_DIR / "settings.json"))

def write_settings(s):
    json.dump(s, open(DATA_DIR / "settings.json","w"))

# -------------------- Helpers --------------------
def weekdays_in_month(year:int, month:int):
    cal = calendar.Calendar(firstweekday=calendar.MONDAY)
    out = []
    for d in cal.itermonthdates(year, month):
        if d.month==month and d.weekday()<5:
            out.append((d, ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][d.weekday()]))
    return out

def second_and_fourth_fridays(year:int, month:int):
    fridays = [d for d, wd in weekdays_in_month(year, month) if wd=="Fri"]
    fridays = [f[0] if isinstance(f, tuple) else f for f in fridays]
    out = []
    if len(fridays)>=2: out.append(fridays[1])
    if len(fridays)>=4: out.append(fridays[3])
    return out

def _room28_default_for(wd, session, defaults_df):
    m = defaults_df[(defaults_df["weekday"]==wd) & (defaults_df["session"]==session)]
    return (m.iloc[0]["consultant"].strip() if not m.empty else "")

def _is_ph(d, ph_df):
    return any((r==d) for r in ph_df["date"] if r!="")

def _has_leave(consultant, d, session, leave_df):
    rows = leave_df[(leave_df["consultant"]==consultant) & (leave_df["date"]==d)]
    for _, r in rows.iterrows():
        s = str(r.get("session","")).upper()
        if s in ("FULL","FD","WHOLE","ALL","DAY","FULL DAY"):
            return True
        if s == session.upper():
            return True
    return False

def _has_woodlands(consultant, d, wood_df):
    return not wood_df[(wood_df["consultant"]==consultant) & (wood_df["date"]==d)].empty

def _has_ward_am(consultant, d, wr_df):
    return not wr_df[(wr_df["consultant"]==consultant) & (wr_df["date"]==d)].empty

def _has_blues_pm(consultant, d, blues_df):
    return not blues_df[(blues_df["consultant"]==consultant) & (blues_df["date"]==d)].empty

# -------------------- Core: build roster --------------------
def build_roster(year, month, week_preceptor_mode,
                 consultants_df, defaults_df, juniors_df, ph_df, leave_df,
                 wr_df, blues_df, wood_df, paeds_df, special_df):
    consultants_list = consultants_df[consultants_df["active(Y/N)"]=="Y"]["name"].tolist()
    preceptor_eligible = set(
        consultants_df[(consultants_df["active(Y/N)"]=="Y") & (consultants_df["preceptor_eligible(Y/N)"]=="Y")]["name"].tolist()
    )
    rows = []
    day_slots = weekdays_in_month(year, month)
    juniors_map = {r["date"]: r["junior_name"] for _, r in juniors_df.iterrows() if r["date"] != ""}
    fri_2_4 = second_and_fourth_fridays(year, month)

    # Smart weekly preceptor picking for "fixed"
    week_monday_to_preceptor = {}
    if week_preceptor_mode == "fixed":
        fri_2_4_month = fri_2_4
        ph_set = set([d for d in ph_df["date"].tolist() if d!=""])
        def _is_available_for_session(cand, d, sess):
            if d in ph_set:
                return False
            wd = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][d.weekday()]
            r28 = _room28_default_for(wd, sess, defaults_df)
            if wd=="Fri" and sess=="PM" and d in fri_2_4_month:
                r28 = "Suma"
            if r28 == cand:
                return False
            if _has_woodlands(cand, d, wood_df):
                return False
            if _has_leave(cand, d, sess, leave_df):
                return False
            if sess=="AM" and _has_ward_am(cand, d, wr_df):
                return False
            if sess=="PM" and _has_blues_pm(cand, d, blues_df):
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
                    cur += timedelta(days=7); continue
                best, best_score = None, -1
                for cand in preceptor_eligible:
                    score = 0
                    for d in week_workdays:
                        for sess in ("AM","PM"):
                            if _is_available_for_session(cand, d, sess):
                                score += 1
                    if score > best_score:
                        best, best_score = cand, score
                    elif score == best_score and best is not None:
                        if prior_counts.get(cand,0) < prior_counts.get(best,0):
                            best = cand
                        elif prior_counts.get(cand,0) == prior_counts.get(best,0) and cand < best:
                            best = cand
                week_monday_to_preceptor[cur] = best if best_score > 0 else None
                if best and best_score > 0:
                    prior_counts[best] += 1
                cur += timedelta(days=7)

    for d, wd in day_slots:
        for session in ("AM","PM"):
            notes = []
            room18 = juniors_map.get(d, "TBD")
            room28 = _room28_default_for(wd, session, defaults_df)
            if (wd=="Fri") and (session=="PM") and (d in fri_2_4):
                room28 = "Suma"
            room29 = ""
            for _, r in paeds_df[paeds_df["date"]==d].iterrows():
                if str(r.get("session","")).upper() in ("", session.upper()):
                    room29 = r["consultant"]
            specials_today = special_df[(special_df["date"]==d) & (special_df["session"].str.upper().isin(["", session.upper()]))]
            for _, r in specials_today.iterrows():
                rn, assignee = r.get("room_name",""), r.get("assignee","")
                if str(rn).strip().lower() in ("room29", "room 29", "r29", "29"):
                    if not room29:
                        room29 = assignee
                else:
                    if rn or assignee:
                        notes.append(f"{rn}: {assignee}")

            preceptor = ""
            if week_preceptor_mode == "fixed":
                monday = d - timedelta(days=d.weekday())
                candidate = week_monday_to_preceptor.get(monday, "")
                preceptor = candidate or ""
            else:
                preceptor = ""

            rows.append({
                "date": d,
                "weekday": wd,
                "session": session,
                "Room 18": room18,
                "Room 28": room28,
                "Preceptor": preceptor,
                "Room 29": room29,
                "notes": "; ".join(notes) if notes else ""
            })

    roster = pd.DataFrame(rows)

    # Apply blocking
    def _wipe(r, cols):
        for c in cols: r[c] = ""

    for i in roster.index:
        r = roster.loc[i]
        d, session = r["date"], r["session"]
        if _is_ph(d, ph_df):
            _wipe(roster.loc[i], ["Room 18","Room 28","Preceptor","Room 29"])
            roster.loc[i,"notes"] = (roster.loc[i,"notes"] + "; Public Holiday").strip("; "); continue

        for col in ("Room 28","Preceptor"):
            name = r[col]
            if not name: continue
            if _has_woodlands(name, d, wood_df):
                roster.loc[i, col] = ""
                roster.loc[i,"notes"] = (roster.loc[i,"notes"] + f"; Woodlands({name})").strip("; ")
            elif _has_leave(name, d, session, leave_df):
                roster.loc[i, col] = ""
                roster.loc[i,"notes"] = (roster.loc[i,"notes"] + f"; Leave({name})").strip("; ")
        for col in ("Room 28","Preceptor"):
            name = roster.loc[i, col]
            if name and session=="AM" and _has_ward_am(name, d, wr_df):
                roster.loc[i, col] = ""
                roster.loc[i,"notes"] = (roster.loc[i,"notes"] + f"; WardRounds({name})").strip("; ")
            if name and session=="PM" and _has_blues_pm(name, d, blues_df):
                roster.loc[i, col] = ""
                roster.loc[i,"notes"] = (roster.loc[i,"notes"] + f"; Blues({name})").strip("; ")
        if roster.loc[i,"Room 28"] and roster.loc[i,"Preceptor"] and roster.loc[i,"Room 28"]==roster.loc[i,"Preceptor"]:
            roster.loc[i,"Preceptor"] = ""
    return roster

# -------------------- Conflicts --------------------
def conflicts(roster_df, consultants_df, ph_df, leave_df, wr_df, blues_df, wood_df):
    rows = []
    preceptor_ok = set(
        consultants_df[(consultants_df["active(Y/N)"]=="Y") & (consultants_df["preceptor_eligible(Y/N)"]=="Y")]["name"].tolist()
    )
    def _add(idx, d, sess, field, name, reason):
        rows.append({"row_index": idx,"date": d,"session": sess,"field": field,"name": name,"conflict_reason": reason})
    for idx, r in roster_df.reset_index().iterrows():
        d, sess = r["date"], r["session"]
        if ph_df.shape[0] and any((x==d) for x in ph_df["date"] if x!=""):
            for field in ["Room 18","Room 28","Preceptor","Room 29"]:
                if str(r[field]).strip():
                    _add(r["index"], d, sess, field, r[field], "Scheduled on public holiday")
        for field in ["Room 28","Preceptor"]:
            name = str(r[field]).strip()
            if not name: continue
            if not leave_df.empty and any((row["consultant"]==name and row["date"]==d and (str(row["session"]).upper() in ("FULL","FD") or str(row["session"]).upper()==sess)) for _, row in leave_df.iterrows()):
                _add(r["index"], d, sess, field, name, "On leave")
            if not wood_df.empty and any((row["consultant"]==name and row["date"]==d) for _, row in wood_df.iterrows()):
                _add(r["index"], d, sess, field, name, "Woodlands day")
            if sess=="AM" and not wr_df.empty and any((row["consultant"]==name and row["date"]==d) for _, row in wr_df.iterrows()):
                _add(r["index"], d, sess, field, name, "Ward rounds (AM)")
            if sess=="PM" and not blues_df.empty and any((row["consultant"]==name and row["date"]==d) for _, row in blues_df.iterrows()):
                _add(r["index"], d, sess, field, name, "Blues (PM)")
        name = str(r["Preceptor"]).strip()
        if name and name not in preceptor_ok:
            _add(r["index"], d, sess, "Preceptor", name, "Not preceptor-eligible")
        if str(r["Room 28"]).strip() and str(r["Preceptor"]).strip() and r["Room 28"]==r["Preceptor"]:
            _add(r["index"], d, sess, "Preceptor", r["Preceptor"], "Same as Room 28 (exclusivity)")
        if str(r["Room 18"]).strip() and str(r["Preceptor"]).strip() and r["Room 18"]==r["Preceptor"]:
            _add(r["index"], d, sess, "Preceptor", r["Preceptor"], "Same as Room 18 (exclusivity)")
    return pd.DataFrame(rows)

# -------------------- Overview --------------------
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

def calendar_markdown(roster_df: pd.DataFrame, ph_df: pd.DataFrame):
    if roster_df.empty:
        return "_No roster_"
    year = roster_df["date"].iloc[0].year
    month = roster_df["date"].iloc[0].month
    cal = calendar.Calendar(firstweekday=calendar.MONDAY)
    weeks = cal.monthdatescalendar(year, month)
    idx = {(r["date"], r["session"]): r for _, r in roster_df.iterrows()}
    ph_set = set([d for d in ph_df["date"].tolist() if d!=""])
    md = []
    md.append("| Mon | Tue | Wed | Thu | Fri | Sat | Sun |")
    md.append("|-----|-----|-----|-----|-----|-----|-----|")
    for week in weeks:
        cells = []
        for d in week:
            if d.month != month:
                cells.append(" ")
                continue
            am = idx.get((d, "AM")); pm = idx.get((d, "PM"))
            ph_badge = " **PH**" if d in ph_set else ""
            def fmt(slot):
                if not slot: return ""
                a = []
                if slot["Room 28"]: a.append(f"R28: {slot['Room 28']}")
                if slot["Preceptor"]: a.append(f"P: {slot['Preceptor']}")
                if slot["Room 29"]: a.append(f"R29: {slot['Room 29']}")
                return "; ".join(a)
            cell = f"**{d.day}**{ph_badge}<br/>AM: {fmt(am)}<br/>PM: {fmt(pm)}"
            cells.append(cell)
        md.append("| " + " | ".join(cells) + " |")
    return "\n".join(md)

def excel_bytes(roster_df: pd.DataFrame):
    from openpyxl import Workbook
    from openpyxl.utils.dataframe import dataframe_to_rows
    wb = Workbook()
    ws1 = wb.active; ws1.title = "Roster"
    for r in dataframe_to_rows(roster_df, index=False, header=True): ws1.append(r)
    ws2 = wb.create_sheet("Overview_List")
    for r in dataframe_to_rows(overview_list(roster_df), index=False, header=True): ws2.append(r)
    ws3 = wb.create_sheet("Export_Flat")
    for r in dataframe_to_rows(roster_df.copy(), index=False, header=True): ws3.append(r)
    bio = BytesIO(); wb.save(bio); bio.seek(0)
    return bio.getvalue()

def run_acceptance_checks(roster_df, inputs, settings):
    results = []
    def add(check, ok, detail): results.append({"check": check, "status": "PASS" if ok else "FAIL", "detail": detail})
    fri_days = sorted([d for d, wd in weekdays_in_month(settings["year"], settings["month"]) if wd=="Fri"])
    if len(fri_days) >= 4:
        c2 = roster_df[(roster_df["date"]==fri_days[1]) & (roster_df["session"]=="PM")]["Room 28"].iloc[0] if not roster_df.empty else ""
        c4 = roster_df[(roster_df["date"]==fri_days[3]) & (roster_df["session"]=="PM")]["Room 28"].iloc[0] if not roster_df.empty else ""
        add("Suma on 2nd & 4th Friday PM", (c2=="Suma" and c4=="Suma"), f"2nd Fri PM={c2}, 4th Fri PM={c4}")
    else:
        add("Suma rule (insufficient Fridays)", True, "Skipped")
    wood = inputs["woodlands"]
    if not wood.empty:
        for _, r in wood.iterrows():
            if r["consultant"]=="Khin" and r["date"]!="":
                day = r["date"]
                day_df = roster_df[roster_df["date"]==day]
                forbidden = any((x=="Khin") for x in day_df[["Room 28","Preceptor"]].values.flatten())
                add("Woodlands blocks R28/Preceptor (Khin)", not forbidden, f"{day}: {'blocked' if not forbidden else 'assigned'}")
    if settings.get("week_preceptor_mode")=="fixed":
        week_groups, ok_all = {}, True
        for _, r in roster_df.iterrows():
            d = r["date"]; monday = d - timedelta(days=d.weekday())
            week_groups.setdefault(monday, []).append(r)
        for wk, rows in week_groups.items():
            pre = [str(x['Preceptor']).strip() for x in rows if str(x['Preceptor']).strip()]
            if pre: ok_all = ok_all and (len(set(pre))==1)
        add("Fixed mode -> single preceptor per week", ok_all, "Consistent where assigned")
    exc = all((not r["Room 28"] or not r["Preceptor"] or r["Room 28"]!=r["Preceptor"]) for _, r in roster_df.iterrows())
    add("Exclusivity (Room28 != Preceptor)", exc, "No same-name overlaps")
    ph = inputs["public_holidays"]
    for _, r in ph.iterrows():
        if r["date"]!="":
            day_df = roster_df[roster_df["date"]==r["date"]]
            if not day_df.empty:
                empty = all((not str(x).strip()) for x in day_df[["Room 18","Room 28","Preceptor","Room 29"]].values.flatten())
                add(f"Public holiday {r['date']} empty", empty, "All empty")
    return pd.DataFrame(results)

# -------------------- UI --------------------
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    ensure_seed_files()

    settings = read_settings()
    with st.sidebar:
        st.header("Settings")
        col1, col2 = st.columns(2)
        month = col1.selectbox("Month", list(range(1,13)), index=settings.get("month", date.today().month)-1)
        year = col2.number_input("Year", min_value=2020, max_value=2100, value=settings.get("year", date.today().year), step=1)
        mode = st.radio("Preceptor assignment", options=["fixed","per_session"],
                        index=0 if settings.get("week_preceptor_mode","fixed")=="fixed" else 1)
        if st.button("Save Settings"):
            write_settings({"month": int(month), "year": int(year), "week_preceptor_mode": mode})
            st.success("Settings saved.")

    tabs = st.tabs(["Consultants & Defaults", "Inputs", "Build Roster", "Overview", "Exports"])

    with tabs[0]:
        st.subheader("Consultants")
        consultants_df = read_csv("consultants.csv", parse_dates=())
        edited = st.data_editor(consultants_df, num_rows="dynamic", width='stretch', key="edit_consultants")
        if st.button("Save Consultants"):
            write_csv("consultants.csv", edited); st.success("Saved consultants.csv")

        st.divider()
        st.subheader("Room 28 Defaults (fixed senior assignments)")
        defaults_df = read_csv("room28_defaults.csv", parse_dates=())
        edited_d = st.data_editor(defaults_df, num_rows="dynamic", width='stretch', key="edit_room28_defaults")
        if st.button("Save Room28 Defaults"):
            write_csv("room28_defaults.csv", edited_d); st.success("Saved room28_defaults.csv")

    with tabs[1]:
        st.subheader("Operational Inputs")
        input_files = [
            ("leave.csv", ["consultant","date","session","type","notes"]),
            ("ward_rounds.csv", ["consultant","date","notes"]),
            ("blues.csv", ["consultant","date","notes"]),
            ("woodlands.csv", ["consultant","date","session","notes"]),
            ("paeds.csv", ["consultant","date","session","notes"]),
            ("juniors.csv", ["date","junior_name"]),
            ("public_holidays.csv", ["date","name"]),
            ("special_slots.csv", ["date","session","room_name","assignee"]),
        ]
        cols = st.columns(2)
        for idx, (fname, _) in enumerate(input_files):
            with cols[idx % 2]:
                st.markdown(f"**{fname}**")
                df = read_csv(fname)
                edited = st.data_editor(df, num_rows="dynamic", width='stretch', key=f"edit_{fname}")
                if st.button(f"Save {fname}", key=f"save_{fname}"):
                    write_csv(fname, edited); st.success(f"Saved {fname}")

    with tabs[2]:
        st.subheader("Generate Roster")
        consultants_df = read_csv("consultants.csv", parse_dates=())
        defaults_df = read_csv("room28_defaults.csv", parse_dates=())
        juniors_df = read_csv("juniors.csv")
        ph_df = read_csv("public_holidays.csv")
        leave_df = read_csv("leave.csv")
        wr_df = read_csv("ward_rounds.csv")
        blues_df = read_csv("blues.csv")
        wood_df = read_csv("woodlands.csv")
        paeds_df = read_csv("paeds.csv")
        special_df = read_csv("special_slots.csv")

        roster_df = build_roster(settings["year"], settings["month"], settings["week_preceptor_mode"],
                                 consultants_df, defaults_df, juniors_df, ph_df, leave_df,
                                 wr_df, blues_df, wood_df, paeds_df, special_df)

        st.dataframe(roster_df, width='stretch', height=500)

        st.markdown("**Conflicts**")
        conf_df = conflicts(roster_df, consultants_df, ph_df, leave_df, wr_df, blues_df, wood_df)
        st.dataframe(conf_df, width='stretch', height=240)

        if st.button("Run Acceptance Checks"):
            checks = run_acceptance_checks(roster_df, {
                "woodlands": wood_df,
                "leave": leave_df,
                "public_holidays": ph_df,
            }, settings)
            st.dataframe(checks, width='stretch')

        st.info("Adjust inputs above and regenerate to update assignments.")

    with tabs[3]:
        st.subheader("List view")
        roster_df = build_roster(settings["year"], settings["month"], settings["week_preceptor_mode"],
                                 read_csv("consultants.csv", parse_dates=()),
                                 read_csv("room28_defaults.csv", parse_dates=()),
                                 read_csv("juniors.csv"),
                                 read_csv("public_holidays.csv"),
                                 read_csv("leave.csv"),
                                 read_csv("ward_rounds.csv"),
                                 read_csv("blues.csv"),
                                 read_csv("woodlands.csv"),
                                 read_csv("paeds.csv"),
                                 read_csv("special_slots.csv"))
        ov = overview_list(roster_df)
        st.dataframe(ov, width='stretch', height=480)

        st.subheader("Calendar")
        md = calendar_markdown(roster_df, read_csv("public_holidays.csv"))
        st.markdown(md, unsafe_allow_html=True)

    with tabs[4]:
        st.subheader("Downloads")
        roster_df = build_roster(settings["year"], settings["month"], settings["week_preceptor_mode"],
                                 read_csv("consultants.csv", parse_dates=()),
                                 read_csv("room28_defaults.csv", parse_dates=()),
                                 read_csv("juniors.csv"),
                                 read_csv("public_holidays.csv"),
                                 read_csv("leave.csv"),
                                 read_csv("ward_rounds.csv"),
                                 read_csv("blues.csv"),
                                 read_csv("woodlands.csv"),
                                 read_csv("paeds.csv"),
                                 read_csv("special_slots.csv"))
        csv_bytes = roster_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Roster CSV", data=csv_bytes, file_name="roster.csv", mime="text/csv")

        xlsx = excel_bytes(roster_df)
        st.download_button("Download Excel Workbook", data=xlsx, file_name="NTBSC_Roster.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.caption("Workbook contains: Roster, Overview_List, Export_Flat")

if __name__ == "__main__":
    main()
