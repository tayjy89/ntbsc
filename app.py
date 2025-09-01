
import streamlit as st
import pandas as pd
from datetime import date, datetime
import calendar, io, json
from dateutil.parser import parse as dtparse

st.set_page_config(page_title="NTBSC Roster Planner", layout="wide")

# ---------- Helpers ----------
def load_csv(path, dtype=None):
    try:
        return pd.read_csv(path, dtype=dtype)
    except Exception:
        return pd.DataFrame()

def save_csv(df, path):
    df.to_csv(path, index=False)

@st.cache_data(show_spinner=False)
def weekdays_in_month(year, month):
    last = calendar.monthrange(year, month)[1]
    days = [date(year, month, d) for d in range(1, last+1) if date(year, month, d).weekday() < 5]
    return days

def parse_date_col(series):
    out = []
    for v in series.fillna(""):
        if isinstance(v, (datetime, date)):
            out.append(v.date() if isinstance(v, datetime) else v)
        else:
            v = str(v).strip()
            if not v:
                out.append(None); continue
            try:
                out.append(dtparse(v, dayfirst=False).date())
            except Exception:
                out.append(None)
    return out

def second_and_fourth_fridays(year, month):
    fridays = [d for d in weekdays_in_month(year, month) if d.weekday()==4]
    out = []
    if len(fridays) >= 2: out.append(fridays[1])
    if len(fridays) >= 4: out.append(fridays[3])
    return set(out)

def is_ph(d, ph_df):
    if ph_df.empty: return False
    dates = set(parse_date_col(ph_df["date"]))
    return d in dates

def has_leave(name, d, session, leave_df):
    if leave_df.empty: return False
    df = leave_df.copy()
    df["__date"] = parse_date_col(df["date"])
    df["__name"] = df["consultant"].fillna("").str.strip().str.lower()
    df["__session"] = df["session"].fillna("").str.strip().str.upper()
    mask = (df["__date"]==d) & (df["__name"]==name.lower()) & (df["__session"].isin(["FULL", session.upper()]))
    return bool(mask.any())

def on_wardrounds(name, d, ward_df):
    if ward_df.empty: return False
    df = ward_df.copy()
    df["__date"] = parse_date_col(df["date"])
    df["__name"] = df["consultant"].fillna("").str.strip().str.lower()
    return bool(((df["__date"]==d) & (df["__name"]==name.lower())).any())

def on_blues(name, d, blues_df):
    if blues_df.empty: return False
    df = blues_df.copy()
    df["__date"] = parse_date_col(df["date"])
    df["__name"] = df["consultant"].fillna("").str.strip().str.lower()
    return bool(((df["__date"]==d) & (df["__name"]==name.lower())).any())

def on_woodlands(name, d, session, wood_df):
    if wood_df.empty: return False
    df = wood_df.copy()
    df["__date"] = parse_date_col(df["date"])
    df["__name"] = df["consultant"].fillna("").str.strip().str.lower()
    df["__session"] = df["session"].fillna("Full").str.title()
    mask = (df["__date"]==d) & (df["__name"]==name.lower()) & (df["__session"].isin(["Full", session.title()]))
    return bool(mask.any())

def paeds_label(d, session, paeds_df):
    if paeds_df.empty: return ""
    df = paeds_df.copy()
    df["__date"] = parse_date_col(df["date"])
    df["__sess"] = df["session"].fillna("").str.upper()
    subset = df[(df["__date"]==d) & (df["__sess"]==session.upper())]
    if subset.empty: return ""
    names = subset["consultant"].fillna("").tolist()
    names = [n for n in names if n]
    return "Paeds: " + ", ".join(names) if names else ""

def junior_for_date(d, juniors_df):
    if juniors_df.empty: return "TBD"
    df = juniors_df.copy()
    df["__date"] = parse_date_col(df["date"])
    subset = df[df["__date"]==d]
    if subset.empty: return "TBD"
    return ", ".join(subset["junior_name"].fillna("TBD").tolist())

def build_roster(month, year, consultants_df, preceptor_defaults_df, leave_df, ward_df, blues_df, wood_df, paeds_df, juniors_df, ph_df, settings):
    days = weekdays_in_month(year, month)
    preceptor_eligible = set(consultants_df.query("preceptor_eligible=='Y' and active=='Y'")["name"].str.lower())
    active = set(consultants_df.query("active=='Y'")["name"].str.lower())

    # Defaults map
    def_map = {}
    for _, r in preceptor_defaults_df.iterrows():
        key = (r["weekday"], r["session"])
        def_map[key] = (r["default_preceptor"] or "").strip()

    # Baseline rows
    rows = []
    suma_fridays = second_and_fourth_fridays(year, month)
    for d in days:
        wd = d.strftime("%A")
        for sess in ["AM","PM"]:
            r18 = junior_for_date(d, juniors_df)
            r28 = ""
            if (d in suma_fridays) and sess=="PM":
                r28 = "Suma"
            pre_def = def_map.get((wd, sess), "")
            # block preceptor default if PH or duties
            blocked = False
            if pre_def:
                if is_ph(d, ph_df): blocked = True
                elif has_leave(pre_def, d, sess, leave_df): blocked = True
                elif on_woodlands(pre_def, d, sess, wood_df): blocked = True
                elif sess=="AM" and on_wardrounds(pre_def, d, ward_df): blocked = True
                elif sess=="PM" and on_blues(pre_def, d, blues_df): blocked = True
            preceptor = "" if blocked else pre_def
            r29 = paeds_label(d, sess, paeds_df)
            rows.append({
                "date": d, "weekday": wd, "session": sess,
                "room18": r18, "room28": r28, "preceptor": preceptor, "room29": r29, "notes": ""
            })

    roster = pd.DataFrame(rows)

    # Week-level preceptor smoothing if enabled
    if settings.get("week_preceptor_mode","fixed") == "fixed":
        # anchor by Monday AM preceptor; apply to week
        # Week boundaries Monday..Friday; keep same preceptor when valid
        roster["week_index"] = roster["date"].apply(lambda d: d.isocalendar().week)
        for wk, sub in roster.groupby("week_index"):
            # pick anchor: first non-empty default preceptor in that week
            anchor = None
            for i, r in sub.iterrows():
                if r["preceptor"]:
                    anchor = r["preceptor"]
                    break
            if not anchor:
                continue
            # propagate anchor unless blocked or running a room that session
            for i, r in sub.iterrows():
                sess = r["session"]
                d = r["date"]
                # if anchor invalid for this slot, skip
                if is_ph(d, ph_df): continue
                if has_leave(anchor, d, sess, leave_df): continue
                if on_woodlands(anchor, d, sess, wood_df): continue
                if (sess=="AM" and on_wardrounds(anchor, d, ward_df)) or (sess=="PM" and on_blues(anchor, d, blues_df)):
                    continue
                # cannot precept if also roomed
                if str(r["room28"]).strip().lower() == anchor.lower():
                    continue
                if str(r["room18"]).strip().lower() == anchor.lower():
                    continue
                roster.loc[i, "preceptor"] = anchor

        roster = roster.drop(columns=["week_index"])

    return roster

def conflicts(roster, consultants_df, leave_df, ward_df, blues_df, wood_df, ph_df):
    preceptor_eligible = set(consultants_df.query("preceptor_eligible=='Y' and active=='Y'")["name"].str.lower())
    issues = []
    for i, r in roster.iterrows():
        d, sess = r["date"], r["session"]
        r18 = (r["room18"] or "").strip()
        r28 = (r["room28"] or "").strip()
        prec = (r["preceptor"] or "").strip()

        if is_ph(d, ph_df):
            issues.append((i, d, sess, "All", "", "Public holiday – all duties blocked"))

        if r28:
            if has_leave(r28, d, sess, leave_df): issues.append((i,d,sess,"Room 28", r28,"On leave"))
            if on_woodlands(r28, d, sess, wood_df): issues.append((i,d,sess,"Room 28", r28,"On Woodlands duty"))
            if sess=="AM" and on_wardrounds(r28, d, ward_df): issues.append((i,d,sess,"Room 28", r28,"On ward rounds (AM)"))
            if sess=="PM" and on_blues(r28, d, blues_df): issues.append((i,d,sess,"Room 28", r28,"On Blues (PM)"))

        if prec:
            if prec.lower() not in preceptor_eligible:
                issues.append((i,d,sess,"Preceptor", prec,"Not preceptor-eligible"))
            if has_leave(prec, d, sess, leave_df): issues.append((i,d,sess,"Preceptor", prec,"On leave"))
            if on_woodlands(prec, d, sess, wood_df): issues.append((i,d,sess,"Preceptor", prec,"On Woodlands duty"))
            if sess=="AM" and on_wardrounds(prec, d, ward_df): issues.append((i,d,sess,"Preceptor", prec,"On ward rounds (AM)"))
            if sess=="PM" and on_blues(prec, d, blues_df): issues.append((i,d,sess,"Preceptor", prec,"On Blues (PM)"))
            if r28 and prec.strip().lower()==r28.strip().lower():
                issues.append((i,d,sess,"Preceptor", prec,"Cannot be the same as Room 28 in the same session"))
            if r18 and prec.strip().lower()==r18.strip().lower():
                issues.append((i,d,sess,"Preceptor", prec,"Cannot be the same as Room 18 in the same session"))

    if not issues:
        return pd.DataFrame(columns=["row","date","session","field","name","conflict"])
    df = pd.DataFrame(issues, columns=["row","date","session","field","name","conflict"])
    return df

def overview_list(roster):
    rows = []
    for d, sub in roster.groupby("date"):
        wd = d.strftime("%A")
        am = sub[sub["session"]=="AM"].iloc[0] if not sub[sub["session"]=="AM"].empty else None
        pm = sub[sub["session"]=="PM"].iloc[0] if not sub[sub["session"]=="PM"].empty else None
        rows.append({
            "date": d, "weekday": wd,
            "AM: Room 18": am["room18"] if am is not None else "",
            "AM: Room 28": am["room28"] if am is not None else "",
            "AM: Preceptor": am["preceptor"] if am is not None else "",
            "AM: Room 29": am["room29"] if am is not None else "",
            "PM: Room 18": pm["room18"] if pm is not None else "",
            "PM: Room 28": pm["room28"] if pm is not None else "",
            "PM: Preceptor": pm["preceptor"] if pm is not None else "",
            "PM: Room 29": pm["room29"] if pm is not None else "",
        })
    return pd.DataFrame(rows).sort_values("date")

def calendar_markdown(roster, ph_df):
    # Simple weekly grid
    if roster.empty: return "No data"
    any_date = roster["date"].iloc[0]
    year = any_date.year; month = any_date.month
    first_weekday, num_days = calendar.monthrange(year, month)  # 0=Mon
    md = f"### {calendar.month_name[month]} {year}\n\n"
    md += "| Mon | Tue | Wed | Thu | Fri | Sat | Sun |\n"
    md += "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n"
    dnum = 1
    for week in range(6):
        cells = []
        for dow in range(7):
            day_index = week*7 + dow
            if day_index < first_weekday or dnum > num_days:
                cells.append(" ")
                continue
            d = date(year, month, dnum)
            dnum += 1
            sub = roster[roster["date"]==d]
            am = sub[sub["session"]=="AM"]
            pm = sub[sub["session"]=="PM"]
            am_text = ""
            pm_text = ""
            if not am.empty:
                r = am.iloc[0]
                am_text = f"AM R18 {r['room18'] or ''}<br>R28 {r['room28'] or ''}<br>Prec {r['preceptor'] or ''}"
            if not pm.empty:
                r = pm.iloc[0]
                pm_text = f"PM R18 {r['room18'] or ''}<br>R28 {r['room28'] or ''}<br>Prec {r['preceptor'] or ''}"
            ph_badge = "🟪 PH<br>" if is_ph(d, ph_df) else ""
            cell = f"<b>{d.day}</b><br>{ph_badge}{am_text}<br>{pm_text}"
            cells.append(cell)
        md += "|" + "|".join(cells) + "|\n"
    return md

def excel_bytes(roster, overview, flat):
    import openpyxl
    from openpyxl.styles import Alignment
    from openpyxl.utils.dataframe import dataframe_to_rows
    wb = openpyxl.Workbook()
    ws = wb.active; ws.title = "Roster"
    for r in dataframe_to_rows(roster, index=False, header=True):
        ws.append(r)
    ws2 = wb.create_sheet("Overview_List")
    for r in dataframe_to_rows(overview, index=False, header=True):
        ws2.append(r)
    ws3 = wb.create_sheet("Export_Flat")
    for r in dataframe_to_rows(flat, index=False, header=True):
        ws3.append(r)
    bio = io.BytesIO()
    wb.save(bio)
    bio.seek(0)
    return bio.getvalue()

# ---------- Load data ----------
DATA_DIR = "data"
consultants = load_csv(f"{DATA_DIR}/consultants.csv")
preceptor_defaults = load_csv(f"{DATA_DIR}/preceptor_defaults.csv")
leave = load_csv(f"{DATA_DIR}/leave.csv")
ward_rounds = load_csv(f"{DATA_DIR}/ward_rounds.csv")
blues = load_csv(f"{DATA_DIR}/blues.csv")
woodlands = load_csv(f"{DATA_DIR}/woodlands.csv")
paeds = load_csv(f"{DATA_DIR}/paeds.csv")
juniors = load_csv(f"{DATA_DIR}/juniors.csv")
ph = load_csv(f"{DATA_DIR}/public_holidays.csv")
special = load_csv(f"{DATA_DIR}/special_slots.csv")

with open(f"{DATA_DIR}/settings.json","r") as f:
    settings = json.load(f)

st.sidebar.header("Settings")
c1, c2 = st.sidebar.columns(2)
month = c1.number_input("Month", 1, 12, int(settings.get("month", date.today().month)))
year = c2.number_input("Year", 2000, 2100, int(settings.get("year", date.today().year)))
week_mode = st.sidebar.selectbox("Preceptor assignment", ["fixed","per_session"], index=0 if settings.get("week_preceptor_mode","fixed")=="fixed" else 1)
settings["month"] = int(month); settings["year"] = int(year); settings["week_preceptor_mode"] = week_mode

if st.sidebar.button("Save settings"):
    with open(f"{DATA_DIR}/settings.json","w") as f:
        json.dump(settings, f, indent=2)
    st.sidebar.success("Settings saved.")

st.title("NTBSC Roster Planner")

tabs = st.tabs(["Consultants & Defaults","Inputs (Leave/Rounds/etc.)","Build Roster","Overview","Exports"])

with tabs[0]:
    st.subheader("Consultants")
    st.caption("Only those with preceptor_eligible='Y' and active='Y' can precept.")
    edited = st.data_editor(consultants, num_rows="dynamic", use_container_width=True)
    if st.button("Save consultants"):
        save_csv(edited, f"{DATA_DIR}/consultants.csv")
        st.success("Consultants saved. Reload the page to refresh dropdowns.")

    st.subheader("Preceptor Defaults (editable)")
    edited_pd = st.data_editor(preceptor_defaults, num_rows="dynamic", use_container_width=True)
    if st.button("Save preceptor defaults"):
        save_csv(edited_pd, f"{DATA_DIR}/preceptor_defaults.csv")
        st.success("Preceptor defaults saved.")

with tabs[1]:
    st.subheader("Core inputs")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Leave**")
        e = st.data_editor(leave, num_rows="dynamic", use_container_width=True)
        if st.button("Save leave"):
            save_csv(e, f"{DATA_DIR}/leave.csv"); st.success("Leave saved.")
        st.markdown("---")
        st.markdown("**Ward rounds (AM block)**")
        e = st.data_editor(ward_rounds, num_rows="dynamic", use_container_width=True)
        if st.button("Save ward rounds"):
            save_csv(e, f"{DATA_DIR}/ward_rounds.csv"); st.success("Ward rounds saved.")
        st.markdown("---")
        st.markdown("**Public holidays (block all)**")
        e = st.data_editor(ph, num_rows="dynamic", use_container_width=True)
        if st.button("Save public holidays"):
            save_csv(e, f"{DATA_DIR}/public_holidays.csv"); st.success("Public holidays saved.")
    with col2:
        st.markdown("**Blues (PM block)**")
        e = st.data_editor(blues, num_rows="dynamic", use_container_width=True)
        if st.button("Save blues"):
            save_csv(e, f"{DATA_DIR}/blues.csv"); st.success("Blues saved.")
        st.markdown("---")
        st.markdown("**Woodlands (Full or AM/PM)**")
        e = st.data_editor(woodlands, num_rows="dynamic", use_container_width=True)
        if st.button("Save woodlands"):
            save_csv(e, f"{DATA_DIR}/woodlands.csv"); st.success("Woodlands saved.")
        st.markdown("---")
        st.markdown("**Paediatrics (Room 29)**")
        e = st.data_editor(paeds, num_rows="dynamic", use_container_width=True)
        if st.button("Save paeds"):
            save_csv(e, f"{DATA_DIR}/paeds.csv"); st.success("Paeds saved.")

    st.markdown("---")
    st.subheader("Juniors (Room 18)")
    e = st.data_editor(juniors, num_rows="dynamic", use_container_width=True)
    if st.button("Save juniors"):
        save_csv(e, f"{DATA_DIR}/juniors.csv"); st.success("Juniors saved.")

    st.markdown("---")
    st.subheader("Special slots (optional)")
    st.caption("e.g. Respi Caroline at Room 29 or other rooms.")
    e = st.data_editor(special, num_rows="dynamic", use_container_width=True)
    if st.button("Save special slots"):
        save_csv(e, f"{DATA_DIR}/special_slots.csv"); st.success("Special slots saved.")

with tabs[2]:
    st.subheader("Build roster")
    consultants = load_csv(f"{DATA_DIR}/consultants.csv")
    preceptor_defaults = load_csv(f"{DATA_DIR}/preceptor_defaults.csv")
    leave = load_csv(f"{DATA_DIR}/leave.csv")
    ward_rounds = load_csv(f"{DATA_DIR}/ward_rounds.csv")
    blues = load_csv(f"{DATA_DIR}/blues.csv")
    woodlands = load_csv(f"{DATA_DIR}/woodlands.csv")
    paeds = load_csv(f"{DATA_DIR}/paeds.csv")
    juniors = load_csv(f"{DATA_DIR}/juniors.csv")
    ph = load_csv(f"{DATA_DIR}/public_holidays.csv")

    roster = build_roster(int(month), int(year), consultants, preceptor_defaults, leave, ward_rounds, blues, woodlands, paeds, juniors, ph, settings)
    st.dataframe(roster, use_container_width=True, height=500)

    st.markdown("**Conflicts**")
    cf = conflicts(roster, consultants, leave, ward_rounds, blues, woodlands, ph)
    if cf.empty:
        st.success("No conflicts detected.")
    else:
        st.dataframe(cf, use_container_width=True, height=240)

    st.session_state["roster_df"] = roster
    st.session_state["conflicts_df"] = cf

with tabs[3]:
    st.subheader("Overview")
    roster = st.session_state.get("roster_df")
    if roster is None or roster.empty:
        st.info("Build the roster first.")
    else:
        lst = overview_list(roster)
        st.dataframe(lst, use_container_width=True, height=420)
        st.markdown("---")
        st.markdown(calendar_markdown(roster, ph), unsafe_allow_html=True)

with tabs[4]:
    st.subheader("Export")
    roster = st.session_state.get("roster_df")
    if roster is None or roster.empty:
        st.info("Build the roster first.")
    else:
        # flat CSV
        flat_rows = []
        for _, r in roster.iterrows():
            d = r["date"]; sess = r["session"]
            flat_rows.append({"date": d, "session": sess, "role": "Room 18 (Junior)", "assigned": r["room18"]})
            flat_rows.append({"date": d, "session": sess, "role": "Room 28 (Senior)", "assigned": r["room28"]})
            flat_rows.append({"date": d, "session": sess, "role": "Preceptor", "assigned": r["preceptor"]})
            if r["room29"]:
                flat_rows.append({"date": d, "session": sess, "role": "Room 29 (Paeds)", "assigned": r["room29"]})
        flat = pd.DataFrame(flat_rows)
        c1, c2 = st.columns(2)
        with c1:
            csv_bytes = flat.to_csv(index=False).encode()
            st.download_button("Download flat CSV", data=csv_bytes, file_name=f"NTBSC_roster_flat_{year}-{month:02d}.csv", mime="text/csv")
        with c2:
            xlsx = excel_bytes(roster, overview_list(roster), flat)
            st.download_button("Download Excel", data=xlsx, file_name=f"NTBSC_roster_{year}-{month:02d}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
