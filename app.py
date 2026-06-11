import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import os
import time
from supabase import create_client
from datetime import datetime
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

st.set_page_config(page_title="PPE Compliance Detection", page_icon="🦺", layout="wide")

SUPABASE_URL = "https://gsqigpxwnzixkaqdntby.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImdzcWlncHh3bnppeGthcWRudGJ5Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODA5OTA1NDAsImV4cCI6MjA5NjU2NjU0MH0.P8NLDODUb6bOPajQIyPASx6I0Miy6bupF_Wt59mY06I"

@st.cache_resource
def get_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase = get_supabase()

@st.cache_resource
def load_models():
    with st.spinner("Loading AI models... please wait"):
        det_path  = hf_hub_download(repo_id="vikalp090/ppe-compliance-yolo11m", filename="best (9).pt")
        pose_path = hf_hub_download(repo_id="vikalp090/ppe-compliance-yolo11m", filename="yolo11m.pt")
        det_model  = YOLO(det_path)
        pose_model = YOLO(pose_path)
    return det_model, pose_model

CLASS_NAMES  = {0:"person",1:"head",2:"face",3:"glasses",4:"face-mask-medical",5:"face-guard",6:"ear",7:"earmuffs",8:"hands",9:"gloves",10:"foot",11:"shoes",12:"safety-vest",13:"tools",14:"helmet",15:"medical-suit",16:"safety-suit"}
REQUIRED_PPE = {14:"helmet",12:"safety-vest",9:"gloves",11:"shoes",3:"glasses"}
PPE_CLASSES  = [3,4,5,7,9,11,12,14,15,16]
HEAD_PPE     = [3,7,4,5]
HAND_PPE     = [9,8]
FEET_PPE     = [11,10]

def process_video(video_path, det_model, pose_model, video_name, progress_bar, status_text):
    import cv2
    cap    = cv2.VideoCapture(video_path)
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    output_path = tempfile.mktemp(suffix=".mp4")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    frame_idx = 0
    compliance_log = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        det_results  = det_model.track(source=frame, persist=True, conf=0.25, iou=0.5, tracker="botsort.yaml", verbose=False)
        pose_results = pose_model(frame, conf=0.3, verbose=False)

        persons   = []
        ppe_items = []

        if det_results[0].boxes is not None:
            for box in det_results[0].boxes:
                cls_id   = int(box.cls[0])
                conf     = float(box.conf[0])
                track_id = int(box.id[0]) if box.id is not None else -1
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                cx = (x1+x2)//2
                cy = (y1+y2)//2
                if cls_id == 0:
                    persons.append({"track_id":track_id,"box":(x1,y1,x2,y2),"cx":cx,"cy":cy,"keypoints":None,"ppe":[]})
                elif cls_id in PPE_CLASSES:
                    ppe_items.append({"cls_id":cls_id,"conf":conf,"cx":cx,"cy":cy,"box":(x1,y1,x2,y2)})

        if pose_results[0].keypoints is not None:
            kps_data  = pose_results[0].keypoints.xy.cpu().numpy()
            kps_confs = pose_results[0].keypoints.conf.cpu().numpy()
            for i, person in enumerate(persons):
                if i < len(kps_data):
                    valid_kps = [kps_data[i][kp_idx] for kp_idx in [0,9,10,15,16] if kps_confs[i][kp_idx] > 0.1]
                    persons[i]["keypoints"] = valid_kps if valid_kps else None

        if len(persons) > 0 and len(ppe_items) > 0:
            for ppe in ppe_items:
                best_person = None
                best_dist   = 9999
                for j, person in enumerate(persons):
                    if ppe["cls_id"] in HEAD_PPE:
                        if person["keypoints"]:
                            dist = np.sqrt((ppe["cx"]-person["keypoints"][0][0])**2+(ppe["cy"]-person["keypoints"][0][1])**2)
                        else:
                            dist = np.sqrt((ppe["cx"]-person["cx"])**2+(ppe["cy"]-person["cy"])**2)
                    elif ppe["cls_id"] in HAND_PPE:
                        if person["keypoints"] and len(person["keypoints"]) >= 3:
                            wrists = person["keypoints"][1:3]
                            dists  = [np.sqrt((ppe["cx"]-kp[0])**2+(ppe["cy"]-kp[1])**2) for kp in wrists if kp[0]>0]
                            dist   = min(dists) if dists else 9999
                        else:
                            dist = np.sqrt((ppe["cx"]-person["cx"])**2+(ppe["cy"]-person["cy"])**2)
                    elif ppe["cls_id"] in FEET_PPE:
                        if person["keypoints"] and len(person["keypoints"]) > 3:
                            ankles = person["keypoints"][3:]
                            dists  = [np.sqrt((ppe["cx"]-kp[0])**2+(ppe["cy"]-kp[1])**2) for kp in ankles if kp[0]>0]
                            dist   = min(dists) if dists else 9999
                        else:
                            dist = np.sqrt((ppe["cx"]-person["cx"])**2+(ppe["cy"]-person["cy"])**2)
                    else:
                        dist = np.sqrt((ppe["cx"]-person["cx"])**2+(ppe["cy"]-person["cy"])**2)
                    if dist < best_dist:
                        best_dist   = dist
                        best_person = j
                if best_person is not None and best_dist < 400:
                    persons[best_person]["ppe"].append(ppe["cls_id"])

        for person in persons:
            x1,y1,x2,y2 = person["box"]
            detected_ppe = list(set(person["ppe"]))
            missing_ppe  = [REQUIRED_PPE[k] for k in REQUIRED_PPE if k not in detected_ppe]
            has_alert    = len(missing_ppe) > 0
            color        = (0,0,255) if has_alert else (0,255,0)
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            label = f"ID:{person['track_id']} {'ALERT' if has_alert else 'OK'}"
            cv2.rectangle(frame,(x1,y1-30),(x1+len(label)*15,y1),color,-1)
            cv2.putText(frame,label,(x1+3,y1-8),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
            y_off = y1+20
            for cls_id in detected_ppe:
                cv2.putText(frame,f"+ {CLASS_NAMES[cls_id]}",(x1+5,y_off),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
                y_off += 20
            for m in missing_ppe:
                cv2.putText(frame,f"- {m}",(x1+5,y_off),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)
                y_off += 20

        for ppe in ppe_items:
            x1,y1,x2,y2 = ppe["box"]
            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,165,0),1)

        out.write(frame)

        for person in persons:
            detected_ppe = list(set(person["ppe"]))
            missing_ppe  = [REQUIRED_PPE[k] for k in REQUIRED_PPE if k not in detected_ppe]
            compliance_log.append({
                "Frame No."   : str(frame_idx).zfill(5),
                "Time (s)"    : round(frame_idx/fps,1),
                "Person ID"   : f"Person {person['track_id']}",
                "Track ID"    : person["track_id"],
                "Detected PPE": ", ".join([CLASS_NAMES[k] for k in detected_ppe]) if detected_ppe else "—",
                "Missing PPE" : ", ".join(missing_ppe) if missing_ppe else "—",
                "Alert"       : "Yes" if missing_ppe else "No"
            })

        progress_bar.progress(frame_idx/total)
        status_text.text(f"Processing frame {frame_idx}/{total}...")

    cap.release()
    out.release()

    converted_path = tempfile.mktemp(suffix=".mp4")
    os.system(f"ffmpeg -i {output_path} -vcodec libx264 -acodec aac {converted_path} -y -loglevel quiet")
    if os.path.exists(converted_path) and os.path.getsize(converted_path) > 0:
        os.unlink(output_path)
        return converted_path, pd.DataFrame(compliance_log)
    return output_path, pd.DataFrame(compliance_log)

# ── UI ────────────────────────────────────────────────────────────────────
st.title("🦺 PPE Compliance Detection System")
st.markdown("**AI-powered Personal Protective Equipment monitoring for construction sites**")
st.divider()

with st.sidebar:
    st.image("https://img.icons8.com/color/96/hard-hat.png", width=80)
    st.title("Navigation")
    page = st.radio("Go to", ["📹 Run Detection", "📊 Compliance Dashboard", "📋 Log History"])
    st.divider()
    st.markdown("**Required PPE:**")
    st.markdown("- 🪖 Helmet\n- 🦺 Safety Vest\n- 🧤 Gloves\n- 👟 Shoes\n- 👓 Glasses")

if page == "📹 Run Detection":
    st.header("📹 Run PPE Detection")
    st.info("Upload a construction site video. The system will detect PPE compliance for each worker.")
    uploaded_video = st.file_uploader("Upload video file", type=["mp4","avi","mov"])

    if uploaded_video:
        st.video(uploaded_video)
        if st.button("🚀 Run PPE Detection", type="primary"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uploaded_video.read())
                tmp_path = tmp.name

            det_model, pose_model = load_models()
            st.subheader("Processing...")
            progress_bar = st.progress(0)
            status_text  = st.empty()
            start_time   = time.time()

            output_path, df = process_video(tmp_path, det_model, pose_model, uploaded_video.name, progress_bar, status_text)
            elapsed = round(time.time()-start_time, 1)
            status_text.text(f"✅ Done! Processed in {elapsed} seconds")
            progress_bar.progress(1.0)

            st.subheader("✅ Processing Complete!")
            with open(output_path,"rb") as f:
                st.download_button("📥 Download Annotated Video", f.read(), "ppe_annotated_output.mp4", "video/mp4")

            st.subheader("Compliance Log")
            st.dataframe(df, use_container_width=True)

            if st.button("📤 Push to Database"):
                with st.spinner("Uploading to Supabase..."):
                    try:
                        records = [{"frame_no":str(row["Frame No."]),"time_seconds":float(row["Time (s)"]),"person_id":str(row["Person ID"]),"track_id":int(row["Track ID"]),"detected_ppe":str(row["Detected PPE"]),"missing_ppe":str(row["Missing PPE"]),"alert":row["Alert"]=="Yes","video_name":uploaded_video.name,"processed_at":datetime.now().isoformat()} for _,row in df.iterrows()]
                        supabase.table("compliance_log").insert(records).execute()
                        st.success(f"✅ {len(records)} records uploaded!")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

            st.download_button("📥 Download CSV", df.to_csv(index=False), "compliance_log.csv", "text/csv")
            os.unlink(tmp_path)

elif page == "📊 Compliance Dashboard":
    st.header("📊 Compliance Analytics Dashboard")
    try:
        data = supabase.table("compliance_log").select("*").execute().data
        if not data:
            st.warning("No data yet. Run detection first!")
        else:
            df = pd.DataFrame(data)
            col1,col2,col3,col4 = st.columns(4)
            total_alerts = df["alert"].sum()
            col1.metric("Total Detections", len(df))
            col2.metric("Total Alerts", int(total_alerts), delta=f"{int(total_alerts)} violations", delta_color="inverse")
            col3.metric("Compliance Rate", f"{round((1-total_alerts/len(df))*100,1)}%")
            col4.metric("Unique Workers", df["track_id"].nunique())
            st.divider()
            col1,col2 = st.columns(2)
            with col1:
                st.subheader("Alert Distribution")
                alert_counts = df["alert"].value_counts()
                st.bar_chart(pd.DataFrame({"Status":["Compliant","Alert"],"Count":[int(alert_counts.get(False,0)),int(alert_counts.get(True,0))]}).set_index("Status"))
            with col2:
                st.subheader("Missing PPE Frequency")
                missing_all = []
                for m in df["missing_ppe"].dropna():
                    if m != "—":
                        missing_all.extend([x.strip() for x in m.split(",")])
                if missing_all:
                    st.bar_chart(pd.Series(missing_all).value_counts())
            st.divider()
            st.subheader("Compliance Log Table")
            st.dataframe(df[["frame_no","time_seconds","person_id","track_id","detected_ppe","missing_ppe","alert","video_name"]], use_container_width=True)
            st.download_button("📥 Download Report", df.to_csv(index=False), "ppe_report.csv", "text/csv")
    except Exception as e:
        st.error(f"❌ Database error: {e}")

elif page == "📋 Log History":
    st.header("📋 Detection Log History")
    try:
        data = supabase.table("compliance_log").select("*").order("processed_at", desc=True).limit(100).execute().data
        if not data:
            st.warning("No logs found!")
        else:
            df = pd.DataFrame(data)
            col1,col2 = st.columns(2)
            with col1:
                alert_filter = st.selectbox("Filter by Alert", ["All","Alert Only","Compliant Only"])
            with col2:
                video_filter = st.selectbox("Filter by Video", ["All"]+list(df["video_name"].unique()))
            if alert_filter == "Alert Only":
                df = df[df["alert"]==True]
            elif alert_filter == "Compliant Only":
                df = df[df["alert"]==False]
            if video_filter != "All":
                df = df[df["video_name"]==video_filter]
            st.dataframe(df, use_container_width=True)
            st.info(f"Showing {len(df)} records")
    except Exception as e:
        st.error(f"❌ Database error: {e}")

st.divider()
st.markdown("<p style='text-align:center;color:gray;'>PPE Compliance Detection System | YOLOv11m + Streamlit + Supabase</p>", unsafe_allow_html=True)
