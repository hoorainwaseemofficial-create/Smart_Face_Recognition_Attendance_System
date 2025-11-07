import streamlit as st
import cv2
import numpy as np
import face_recognition
import os
import pandas as pd
import pickle
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import time

# Page Configuration
st.set_page_config(
    page_title="Smart Face Attendance System",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern UI
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    /* Global Styles */
    * {
        font-family: 'Poppins', sans-serif;
    }

    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }

    /* Header Styling - CHANGED TO DARK NAVY FOR CONTRAST */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        margin-bottom: 2rem;
        animation: fadeInDown 0.8s ease-in-out;
    }

    .main-header h1 {
        color: white;
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }

    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
    }

    /* Dashboard Cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 20px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.1);
        text-align: center;
        transition: all 0.3s ease;
        margin: 1rem 0;
    }

    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 30px rgba(0,0,0,0.15);
    }

    .metric-card h3 {
        color: #667eea;
        font-size: 2.5rem;
        margin: 0.5rem 0;
        font-weight: 700;
    }

    .metric-card p {
        color: #666;
        font-size: 1rem;
        margin: 0;
    }

    .metric-icon {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }

    /* Section Headers */
    .section-header {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem 1.5rem;
        border-radius: 15px;
        color: white;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(102,126,234,0.4);
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(102,126,234,0.6);
    }

    /* Sidebar Styling */
    .css-1d391kg {
        background: white !important;
        color: black !important;
        box-shadow: 2px 0 10px rgba(0, 0, 0, 0.05);
    }

    /* New Streamlit sidebar class */
    [data-testid="stSidebar"] {
        background: white !important;
        color: black !important;
        box-shadow: 2px 0 10px rgba(0, 0, 0, 0.05);
    }

    /* Sidebar text and button colors */
    [data-testid="stSidebar"] * {
        color: #333 !important;
        font-weight: 500;
    }

    /* Webcam Frame */
    .webcam-container {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }

    /* Notification Banner */
    .notification {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        animation: slideInRight 0.5s ease-in-out;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }

    /* Footer - CHANGED TO DARK NAVY FOR CONTRAST */
    .footer {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin-top: 3rem;
        color: white;
        box-shadow: 0 -5px 15px rgba(0,0,0,0.2);
    }

    .footer p {
        margin: 0.5rem 0;
        font-size: 0.95rem;
    }

    /* Dataframe Styling */
    .dataframe {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }

    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }

    /* Animations */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(100px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }

    /* Status Indicators */
    .status-present {
        background: #38ef7d;
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-weight: 600;
    }

    .status-absent {
        background: #f5576c;
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-weight: 600;
    }

    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- Folder setup ---
KNOWN_FACE_DIR = "known_face"
os.makedirs(KNOWN_FACE_DIR, exist_ok=True)

# Initialize session state
if 'last_marked' not in st.session_state:
    st.session_state.last_marked = None
if 'notification' not in st.session_state:
    st.session_state.notification = None


# --- Helper Functions ---
def find_encodings(images, names):
    """Encode faces from images with progress tracking"""
    encodeList = []
    validNames = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, (img, name) in enumerate(zip(images, names)):
        try:
            status_text.text(f"🔄 Encoding face: {name}...")
            img = np.array(img, dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encodes = face_recognition.face_encodings(img)
            if len(encodes) > 0:
                encodeList.append(encodes[0])
                validNames.append(name)
            else:
                st.warning(f"⚠️ No face detected in {name}, skipping.")
            progress_bar.progress((idx + 1) / len(images))
        except Exception as e:
            st.error(f"❌ Error encoding {name}: {e}")

    progress_bar.empty()
    status_text.empty()
    return encodeList, validNames


def mark_attendance(name, status="Present"):
    """Mark attendance for a student in CSV files"""
    date_today = datetime.now().strftime("%Y-%m-%d")
    time_now = datetime.now().strftime("%H:%M:%S")

    # Main file
    if not os.path.exists('attendance.csv'):
        df = pd.DataFrame(columns=["Name", "Date", "Time", "Status"])
        df.to_csv('attendance.csv', index=False)

    df = pd.read_csv('attendance.csv')
    if not ((df['Name'] == name) & (df['Date'] == date_today)).any():
        new_entry = pd.DataFrame([[name, date_today, time_now, status]],
                                 columns=["Name", "Date", "Time", "Status"])
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv('attendance.csv', index=False)

        # Set notification
        st.session_state.notification = f"✅ {name} marked {status} at {time_now}"
        st.session_state.last_marked = name

    # Daily file
    today_file = f"attendance_{date_today}.csv"
    if os.path.exists(today_file):
        df_today = pd.read_csv(today_file)
    else:
        df_today = pd.DataFrame(columns=["Name", "Date", "Time", "Status"])

    if not ((df_today['Name'] == name) & (df_today['Date'] == date_today)).any():
        new_entry_today = pd.DataFrame([[name, date_today, time_now, status]],
                                       columns=["Name", "Date", "Time", "Status"])
        df_today = pd.concat([df_today, new_entry_today], ignore_index=True)
        df_today.to_csv(today_file, index=False)


def mark_absentees(known_faces):
    """Mark all students not present as absent - with safety checks"""
    date_today = datetime.now().strftime("%Y-%m-%d")

    # Create attendance.csv if it doesn't exist
    if not os.path.exists('attendance.csv'):
        df = pd.DataFrame(columns=["Name", "Date", "Time", "Status"])
        df.to_csv('attendance.csv', index=False)

    df = pd.read_csv('attendance.csv')

    absent_count = 0
    for name in known_faces:
        if not ((df['Name'] == name) & (df['Date'] == date_today)).any():
            mark_attendance(name, "Absent")
            absent_count += 1

    return absent_count


# --- Header ---
st.markdown("""
<div class="main-header">
    <h1>📸 Smart Face Recognition Attendance System</h1>
    <p>AI-Powered Real-Time Attendance Tracking | Current Date: {}</p>
</div>
""".format(datetime.now().strftime("%B %d, %Y - %H:%M")), unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("### 🧭 Navigation")
    page = st.radio("", ["🏠 Dashboard", "👩‍🎓 Add Students", "🎥 Live Camera", "📊 Attendance Logs"],
                    label_visibility="collapsed")

    st.markdown("---")
    st.markdown("### ⚙️ Quick Actions")

    # Quick stats in sidebar
    if os.path.exists('attendance.csv'):
        df = pd.read_csv('attendance.csv')
        date_today = datetime.now().strftime("%Y-%m-%d")
        present_today = len(df[(df['Date'] == date_today) & (df['Status'] == 'Present')])
        st.metric("Present Today", present_today)

# --- Load Known Faces ---
images, classNames = [], []
for filename in os.listdir(KNOWN_FACE_DIR):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        img = cv2.imread(os.path.join(KNOWN_FACE_DIR, filename))
        if img is not None:
            images.append(img)
            classNames.append(os.path.splitext(filename)[0])

# --- Load Encodings ---
if os.path.exists("encodings.pkl"):
    with open("encodings.pkl", "rb") as f:
        data = pickle.load(f)
    encodeListKnown = data["encodings"]
    known_face_names = data["names"]
else:
    encodeListKnown = []
    known_face_names = []

# --- Dashboard Page ---
if page == "🏠 Dashboard":
    # Display notification
    if st.session_state.notification:
        st.markdown(f'<div class="notification">{st.session_state.notification}</div>', unsafe_allow_html=True)
        time.sleep(0.1)

    # Metrics Dashboard
    col1, col2, col3, col4 = st.columns(4)

    # Calculate metrics
    date_today = datetime.now().strftime("%Y-%m-%d")
    if os.path.exists('attendance.csv'):
        df = pd.read_csv('attendance.csv')
        present_today = len(df[(df['Date'] == date_today) & (df['Status'] == 'Present')])
        total_sessions = len(df['Date'].unique())
    else:
        present_today = 0
        total_sessions = 0

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">✅</div>
            <h3>{present_today}</h3>
            <p>Present Today</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">🧠</div>
            <h3>{len(classNames)}</h3>
            <p>Known Faces</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">📅</div>
            <h3>{total_sessions}</h3>
            <p>Total Sessions</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">⏰</div>
            <h3>{datetime.now().strftime("%H:%M")}</h3>
            <p>Current Time</p>
        </div>
        """, unsafe_allow_html=True)

    # Today's Attendance Summary
    st.markdown('<div class="section-header">📋 Today\'s Attendance Summary</div>', unsafe_allow_html=True)

    if os.path.exists('attendance.csv'):
        df = pd.read_csv('attendance.csv')
        df_today = df[df['Date'] == date_today]

        if not df_today.empty:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### ✅ Present Students")
                present_df = df_today[df_today['Status'] == 'Present']
                if not present_df.empty:
                    for _, row in present_df.iterrows():
                        st.markdown(f"**{row['Name']}** - {row['Time']}")
                else:
                    st.info("No students marked present yet")

            with col2:
                st.markdown("#### ❌ Absent Students")
                absent_df = df_today[df_today['Status'] == 'Absent']
                if not absent_df.empty:
                    for _, row in absent_df.iterrows():
                        st.markdown(f"**{row['Name']}**")
                else:
                    st.info("No absentees marked yet")
        else:
            st.info("No attendance records for today yet. Start the camera to begin tracking!")
    else:
        st.info("No attendance records yet. Start the camera to begin tracking!")

# --- Add Students Page ---
# --- Add Students Page ---
elif page == "👩‍🎓 Add Students":
    st.markdown('<div class="section-header">📁 Upload Student Images</div>', unsafe_allow_html=True)

    st.info("💡 **Tip:** Upload clear, well-lit photos with visible faces. Name the files as 'StudentName.jpg' for automatic naming.")

    uploaded_files = st.file_uploader(
        "Upload Images (JPG, JPEG, PNG)",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True
    )

    # --- Save Uploaded Images ---
    if uploaded_files:
        cols = st.columns(3)
        for idx, file in enumerate(uploaded_files):
            with cols[idx % 3]:
                st.image(file, caption=file.name, use_container_width=True)

        if st.button("💾 Save Images"):
            for file in uploaded_files:
                with open(os.path.join(KNOWN_FACE_DIR, file.name), "wb") as f:
                    f.write(file.getbuffer())
            st.markdown(
                f'<div class="notification">✅ {len(uploaded_files)} images saved successfully! Please encode faces next.</div>',
                unsafe_allow_html=True)
            time.sleep(1)
            st.rerun()

    # --- Dropdown for Saved Images ---
    if classNames:
        st.markdown(f'<div class="notification">✅ Found {len(classNames)} saved student images</div>', unsafe_allow_html=True)

        selected_name = st.selectbox(
            "🧑‍🎓 Select a Student to View or Delete:",
            options=classNames,
            index=None,
            placeholder="Choose a saved student..."
        )

        if selected_name:
            # Find and show the selected student's image
            img_path = None
            for filename in os.listdir(KNOWN_FACE_DIR):
                if os.path.splitext(filename)[0] == selected_name:
                    img_path = os.path.join(KNOWN_FACE_DIR, filename)
                    break

            if img_path:
                img = cv2.imread(img_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                st.image(img_rgb, caption=selected_name, use_container_width=False, width=250)

                col1, col2 = st.columns([1, 3])
                with col1:
                    if st.button(f"🗑️ Delete {selected_name}", key=f"delete_{selected_name}"):
                        os.remove(img_path)
                        st.markdown(
                            f'<div class="notification">🗑️ {selected_name} deleted successfully!</div>',
                            unsafe_allow_html=True)
                        time.sleep(1)
                        st.rerun()

    # --- Encode Section ---
    st.markdown('<div class="section-header">🧠 Encode Student Faces</div>', unsafe_allow_html=True)

    if classNames:
        st.markdown(
            f'<div class="notification">✅ Ready to encode {len(classNames)} faces: {", ".join(classNames)}</div>',
            unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔄 Start Encoding Process", use_container_width=True):
                with st.spinner("Encoding faces..."):
                    encodeListKnown, known_face_names = find_encodings(images, classNames)
                    with open("encodings.pkl", "wb") as f:
                        pickle.dump({"encodings": encodeListKnown, "names": known_face_names}, f)

                    st.markdown(
                        f'<div class="notification">✅ Face encodings completed! {len(known_face_names)} faces encoded successfully.</div>',
                        unsafe_allow_html=True)
                    time.sleep(1)
                    st.rerun()

        with col2:
            if st.button("🗑️ Delete All Encodings", use_container_width=True):
                if os.path.exists("encodings.pkl"):
                    os.remove("encodings.pkl")
                    st.markdown(
                        '<div class="notification">🗑️ All encodings deleted! Re-encode after fixing duplicates.</div>',
                        unsafe_allow_html=True)
                    time.sleep(1)
                    st.rerun()
    else:
        st.warning("⚠️ No images found. Please upload student images first.")

# --- Live Camera Page ---
elif page == "🎥 Live Camera":
    st.markdown('<div class="section-header">🎥 Live Face Recognition</div>', unsafe_allow_html=True)

    if not encodeListKnown:
        st.error("❌ No face encodings found. Please add and encode student images first!")
        st.info("👉 Go to 'Add Students' page to upload images and encode faces.")
    else:
        st.info(f"🧠 Ready to recognize {len(known_face_names)} students: {', '.join(known_face_names)}")


        class FaceRecognitionProcessor(VideoProcessorBase):
            def __init__(self):
                self.encodeListKnown = encodeListKnown
                self.known_face_names = known_face_names

            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")

                # Resize and convert to RGB
                imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
                imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

                facesCurFrame = face_recognition.face_locations(imgS)
                encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

                for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                    matches = face_recognition.compare_faces(self.encodeListKnown, encodeFace)
                    faceDis = face_recognition.face_distance(self.encodeListKnown, encodeFace)
                    matchIndex = np.argmin(faceDis)

                    y1, x2, y2, x1 = [v * 4 for v in faceLoc]

                    if matches[matchIndex]:
                        name = self.known_face_names[matchIndex]
                        mark_attendance(name, "Present")
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                        cv2.putText(img, name.upper(), (x1 + 6, y2 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    else:
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
                        cv2.putText(img, "UNKNOWN", (x1 + 6, y2 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                return av.VideoFrame.from_ndarray(img, format="bgr24")


        st.markdown('<div class="webcam-container">', unsafe_allow_html=True)
        webrtc_streamer(key="face-recognition", video_processor_factory=FaceRecognitionProcessor)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("📝 Mark Remaining as Absent", use_container_width=True):
                absent_count = mark_absentees(known_face_names)
                if absent_count > 0:
                    st.success(f"✅ {absent_count} students marked as absent!")
                else:
                    st.info("All students already have attendance records for today.")

        with col2:
            if st.button("🔄 Refresh Page", use_container_width=True):
                st.rerun()

# --- Attendance Logs Page ---
elif page == "📊 Attendance Logs":
    st.markdown('<div class="section-header">📊 Attendance Records</div>', unsafe_allow_html=True)

    if os.path.exists("attendance.csv"):
        df = pd.read_csv("attendance.csv")

        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            date_filter = st.date_input("Filter by Date", value=datetime.now())
        with col2:
            name_filter = st.multiselect("Filter by Student", options=sorted(df['Name'].unique()))
        with col3:
            status_filter = st.selectbox("Filter by Status", ["All", "Present", "Absent"])

        # Apply filters
        filtered_df = df.copy()
        if date_filter:
            filtered_df = filtered_df[filtered_df['Date'] == date_filter.strftime("%Y-%m-%d")]
        if name_filter:
            filtered_df = filtered_df[filtered_df['Name'].isin(name_filter)]
        if status_filter != "All":
            filtered_df = filtered_df[filtered_df['Status'] == status_filter]

        # Display results
        st.markdown(f"**Showing {len(filtered_df)} records**")
        st.dataframe(filtered_df, use_container_width=True, height=400)

        # Download buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button(
                "📥 Download Full Attendance CSV",
                df.to_csv(index=False),
                "attendance_full.csv",
                "text/csv",
                use_container_width=True
            )
        with col2:
            st.download_button(
                "📥 Download Filtered CSV",
                filtered_df.to_csv(index=False),
                "attendance_filtered.csv",
                "text/csv",
                use_container_width=True
            )
        with col3:
            st.download_button(
                "📥 Download Today's CSV",
                df[df['Date'] == datetime.now().strftime("%Y-%m-%d")].to_csv(index=False),
                f"attendance_{datetime.now().strftime('%Y-%m-%d')}.csv",
                "text/csv",
                use_container_width=True
            )

        # Statistics
        st.markdown('<div class="section-header">📈 Attendance Statistics</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Total Students", df['Name'].nunique())
        with col3:
            st.metric("Total Days", df['Date'].nunique())

    else:
        st.warning("⚠️ No attendance records yet. Start the camera to begin tracking!")
        st.info("👉 Go to 'Live Camera' page to start marking attendance.")

# --- Footer ---
st.markdown("""
<div class="footer">
    <p>© 2025 Smart Face Recognition Attendance System</p>
    <p>Developed with ❤️ by <strong>HOORAIN WASEEM</strong> | Powered by OpenCV, face_recognition & Streamlit 🚀</p>
    <p>📧 Contact: hoorainwaseem.official@gmail.com| 🌐 GitHub:github.com/hoorainwaseemofficial-create/Smart_Face_Recognition_Attendance_System.git</p>
</div>
""", unsafe_allow_html=True)