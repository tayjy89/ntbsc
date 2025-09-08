# Enhanced NTBSC Roster Planner app with requested features
# (shortened version for packaging demonstration)

import streamlit as st
import pandas as pd
import json
from pathlib import Path
from datetime import date

APP_TITLE = "NTBSC Roster Planner"
DATA_DIR = Path("data")
ROSTER_DIR = DATA_DIR / "rosters"
DATA_DIR.mkdir(exist_ok=True); ROSTER_DIR.mkdir(exist_ok=True)
ADMIN_PASSWORD = "NTBSC_jy"

def ensure_seed_files():
    if not (DATA_DIR/'consultants.csv').exists():
        pd.DataFrame([{'name':'Jun Yang','active(Y/N)':'Y','preceptor_eligible(Y/N)':'Y'}]).to_csv(DATA_DIR/'consultants.csv',index=False)
    if not (DATA_DIR/'room28_defaults.csv').exists():
        pd.DataFrame([{'weekday':'Mon','session':'AM','consultant':'Khin'}]).to_csv(DATA_DIR/'room28_defaults.csv',index=False)
    if not (DATA_DIR/'settings.json').exists():
        json.dump({'month':date.today().month,'year':date.today().year,'week_preceptor_mode':'fixed'}, open(DATA_DIR/'settings.json','w'))
    for fname, cols in [
        ('duties.csv',['consultant','date','session','kind','notes']),
        ('juniors.csv',['date','junior_name']),
        ('paeds.csv',['consultant','date','session','notes']),
        ('special_slots.csv',['date','session','room_name','assignee'])
    ]:
        if not (DATA_DIR/fname).exists():
            pd.DataFrame(columns=cols).to_csv(DATA_DIR/fname,index=False)

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    ensure_seed_files()
    pw = st.sidebar.text_input("Admin password", type="password")
    unlocked = pw==ADMIN_PASSWORD
    tabs = st.tabs(["Overview","Exports"]) if not unlocked else st.tabs(["Consultants & Defaults","Duties","Build/Edit Roster","Overview","Exports"])
    if unlocked:
        with tabs[1]:
            st.subheader("Add duty by date range")
            with st.form("duty_range"):
                c = st.text_input("Consultant")
                kind = st.selectbox("Kind",["leave","ward","blues","woodlands","others"])
                sess = st.selectbox("Session",["AM","PM","Full"])
                start = st.date_input("Start date")
                end = st.date_input("End date")
                submitted = st.form_submit_button("Add")
                if submitted: st.success(f"Would add rows for {c} from {start} to {end}")

    with tabs[-2 if unlocked else 0]:
        st.subheader("Calendar view (demo)")
        st.markdown("_Calendar here_")
        st.subheader("Overview list (demo)")
        st.dataframe(pd.DataFrame([{"date":date.today(),"Room 28":"Khin"}]))

    with tabs[-1]:
        st.subheader("Exports")
        st.download_button("Download CSV", data="date,Room 28\n2025-09-01,Khin", file_name="roster.csv")

if __name__=="__main__": main()
