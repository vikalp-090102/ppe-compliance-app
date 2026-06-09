import streamlit as st
import pandas as pd
import cv2
import numpy as np
import tempfile
import os
from supabase import create_client
from datetime import datetime

# ── PAGE CONFIG ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PPE Compliance Detection",
    page_icon="🦺",
    layout="wide"
)

# ── SUPABASE CONFIG ───────────────────────────────────────────────────────
SUPABASE_URL = "https://gsqigpxwnzixkaqdntby.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImdzcWlncHh3bnppeGthcWRudGJ5Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODA5OTA1NDAsImV4cCI6MjA5NjU2NjU0MH0.P8NLDODUb6bOPajQIyPASx6I0Miy6bupF_Wt59mY06I"

@st.cache_resource
def get_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase = get_supabase()

# ── HEADER ────────────────────────────────────────────────────────────────
st.title("🦺 PPE Compliance Detection System")
st.markdown("**AI-powered Personal Protective Equipment monitoring for construction sites**")
st.divider()

# ── SIDEBAR ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/hard-hat.png", width=80)
    st.title("Navigation")
    page = st.radio("Go to", ["📹 Live Detection", "📊 Compliance Dashboard", "📋 Log History"])
    st.divider()
    st.markdown("**Required PPE:**")
    st.markdown("- 🪖 Helmet")
    st.markdown("- 🦺 Safety Vest")
    st.markdown("- 🧤 Gloves")
    st.markdown("- 👟 Shoes")
    st.markdown("- 👓 Glasses")

# ── PAGE 1: LIVE DETECTION ────────────────────────────────────────────────
if page == "📹 Live Detection":
    st.header("📹 Video PPE Detection")

    uploaded_file = st.file_uploader(
        "Upload a video file",
        type=["mp4", "avi", "mov"],
        help="Upload construction site video for PPE compliance analysis"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Upload Pre-processed Video")
        demo_video = st.file_uploader("Upload annotated output video", type=["mp4"], key="demo")

        if demo_video:
            st.video(demo_video)

    with col2:
        st.subheader("Upload Compliance Log")
        demo_csv = st.file_uploader("Upload compliance CSV", type=["csv"], key="csv")

        if demo_csv:
            df = pd.read_csv(demo_csv)
            st.dataframe(df, use_container_width=True)

            if st.button("📤 Push to Database", type="primary"):
                with st.spinner("Uploading to Supabase..."):
                    try:
                        records = []
                        for _, row in df.iterrows():
                            records.append({
                                "frame_no"      : str(row.get("Frame No.", "")),
                                "time_seconds"  : float(row.get("Time (s)", 0)),
                                "person_id"     : str(row.get("Person ID", "")),
                                "track_id"      : int(row.get("Track ID", 0)),
                                "detected_ppe"  : str(row.get("Detected PPE", "")),
                                "missing_ppe"   : str(row.get("Missing PPE", "")),
                                "alert"         : row.get("Alert", "No") == "Yes",
                                "video_name"    : "Test_Video-7.mp4",
                                "processed_at"  : datetime.now().isoformat()
                            })
                        supabase.table("compliance_log").insert(records).execute()
                        st.success(f"✅ {len(records)} records uploaded to database!")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

# ── PAGE 2: DASHBOARD ─────────────────────────────────────────────────────
elif page == "📊 Compliance Dashboard":
    st.header("📊 Compliance Analytics Dashboard")

    try:
        response = supabase.table("compliance_log").select("*").execute()
        data     = response.data

        if not data:
            st.warning("No data in database yet. Upload a compliance log first!")
        else:
            df = pd.DataFrame(data)

            # ── KPI METRICS ───────────────────────────────────────────────
            total_records  = len(df)
            total_alerts   = df["alert"].sum()
            compliance_rate = round((1 - total_alerts / total_records) * 100, 1) if total_records > 0 else 0
            unique_persons  = df["track_id"].nunique()

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Detections", total_records)
            col2.metric("Total Alerts", int(total_alerts), delta=f"{int(total_alerts)} violations", delta_color="inverse")
            col3.metric("Compliance Rate", f"{compliance_rate}%")
            col4.metric("Unique Workers", unique_persons)

            st.divider()

            # ── CHARTS ────────────────────────────────────────────────────
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Alert Distribution")
                alert_counts = df["alert"].value_counts()
                alert_df     = pd.DataFrame({
                    "Status": ["Compliant", "Alert"],
                    "Count" : [int(alert_counts.get(False, 0)), int(alert_counts.get(True, 0))]
                })
                st.bar_chart(alert_df.set_index("Status"))

            with col2:
                st.subheader("Missing PPE Frequency")
                missing_all = []
                for m in df["missing_ppe"].dropna():
                    if m != "—":
                        missing_all.extend([x.strip() for x in m.split(",")])
                if missing_all:
                    missing_series = pd.Series(missing_all).value_counts()
                    st.bar_chart(missing_series)

            st.divider()

            # ── COMPLIANCE TABLE ───────────────────────────────────────────
            st.subheader("Compliance Log Table")
            display_cols = ["frame_no", "time_seconds", "person_id", "track_id",
                           "detected_ppe", "missing_ppe", "alert", "video_name"]
            st.dataframe(df[display_cols], use_container_width=True)

            # ── DOWNLOAD ──────────────────────────────────────────────────
            csv = df.to_csv(index=False)
            st.download_button(
                label     = "📥 Download Report",
                data      = csv,
                file_name = "ppe_compliance_report.csv",
                mime      = "text/csv"
            )

    except Exception as e:
        st.error(f"❌ Database error: {e}")

# ── PAGE 3: LOG HISTORY ───────────────────────────────────────────────────
elif page == "📋 Log History":
    st.header("📋 Detection Log History")

    try:
        response = supabase.table("compliance_log").select("*").order("processed_at", desc=True).limit(100).execute()
        data     = response.data

        if not data:
            st.warning("No logs found in database!")
        else:
            df = pd.DataFrame(data)

            # ── FILTERS ───────────────────────────────────────────────────
            col1, col2 = st.columns(2)
            with col1:
                alert_filter = st.selectbox("Filter by Alert", ["All", "Alert Only", "Compliant Only"])
            with col2:
                video_filter = st.selectbox("Filter by Video", ["All"] + list(df["video_name"].unique()))

            if alert_filter == "Alert Only":
                df = df[df["alert"] == True]
            elif alert_filter == "Compliant Only":
                df = df[df["alert"] == False]

            if video_filter != "All":
                df = df[df["video_name"] == video_filter]

            st.dataframe(df, use_container_width=True)
            st.info(f"Showing {len(df)} records")

    except Exception as e:
        st.error(f"❌ Database error: {e}")

# ── FOOTER ────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center; color:gray;'>PPE Compliance Detection System | Powered by YOLOv11m + Streamlit + Supabase</p>",
    unsafe_allow_html=True
)
