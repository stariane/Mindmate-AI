import streamlit as st
from google import genai
import os
from dotenv import load_dotenv
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Load secret API Key
load_dotenv(dotenv_path=".env")

# --- 1. SET UP THE VISUAL PAGE ---
st.set_page_config(page_title="MindMate AI", page_icon="logo.png", layout="centered")

# --- CUSTOM BEAUTIFUL THEME CSS ---
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #e0f2f1 0%, #e1f5fe 50%, #f3e5f5 100%);
        background-attachment: fixed;
    }
    [data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.6) !important;
        backdrop-filter: blur(10px);
    }
    .stChatMessage {
        background-color: rgba(255, 255, 255, 0.8) !important;
        border-radius: 15px !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- 2. GLOBAL TRANSLATION DICTIONARY ---
translations = {
    "English": {
        "title": "MindMate AI Support",
        "caption": "Safe, Multilingual AI Support Platform",
        "settings": "Settings",
        "lang_label": "Select Language:",
        "scanner_title": " Auto Face Scanner",
        "cam_on": "Activate Face Detection",
        "manual_mood": "Mood Tracker (Manual)",
        "mood_label": "How are you today?:",
        "active_ctx": "Active Context:",
        "quick_tools": "Quick Tools",
        "breath_btn": "Breathing Exercise",
        "affirm_btn": "Positive Affirmation",
        "input_placeholder": "Type what's on your mind...",
        "breath_prompt": "Give me a very short breathing exercise because the face scanner detected I am feeling ",
        "affirm_prompt": "Give me a single powerful comforting phrase based on my current emotional context of ",
        "cam_msg": "*Auto-captured facial scan data sent to AI*"
    },
    "Kiswahili": {
        "title": " Msaada wa MindMate AI",
        "caption": "Jukwaa la Msaada wa AI la Lugha Nyingi",
        "settings": " Mipangilio",
        "lang_label": "Chagua Lugha:",
        "scanner_title": "Kitambuzi Kiotomatiki",
        "cam_on": "Washa Utambuzi wa Uso",
        "manual_mood": "Jinsi unavyohisi",
        "mood_label": "Leo uko vipi?:",
        "active_ctx": "Hali ya Sasa:",
        "quick_tools": "Njia za Mkato",
        "breath_btn": "Kinga ya Pumzi",
        "affirm_btn": "Maneno ya Kutia Moyo",
        "input_placeholder": "Andika hapa...",
        "breath_prompt": "Nipe zoezi fupi sana la kupumua kwa sababu kitambuzi kimegundua ninahisi ",
        "affirm_prompt": "Nipe kauli moja thabiti ya kunifariji kulingana na hali yangu ya sasa ya ",
        "cam_msg": "*Data ya sura imenaswa kiotomatiki na kutumwa kwa AI*"
    },
    "Dholuo": {
        "title": " Kony ji mar MindMate AI",
        "caption": "Mbalari mar kony kod AI e dhok mathoth",
        "settings": " Chenro",
        "lang_label": "Yier Dhok:",
        "scanner_title": "Kup picha mar Wang'",
        "cam_on": "Chak Utambuzi mar Wang'",
        "manual_mood": "Kaka iwinjo e chunyi",
        "mood_label": "Tinende iwinjo nade?:",
        "active_ctx": "Kaka iwinjo sani:",
        "quick_tools": "Yo mar yudo kony mapiyo",
        "breath_btn": "Lwasi mar Yuyo Muya",
        "affirm_btn": " Jiwo Chuny",
        "input_placeholder": "kona kaka iwinjo ...",
        "breath_prompt": "Miya lwasi mapiyo mar yuyo muya nikech kitambuzi mabor oneno ni awinjo ",
        "affirm_prompt": "Miya wach achiel mamit mar konyo chunya kaluwore gi kaka awinjo sani mar ",
        "cam_msg": " *Oseor weche mag picha ne AI kiotomatiki*"
    }
}

# Load OpenCV's official face detection coordinates tool
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- BACKGROUND FACE CAMERA DETECTOR CLASS ---
class FaceDetectorTransformer(VideoTransformerBase):
    def __init__(self):
        self.face_detected = False
        self.saved_frame = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Scan for faces in the video frame stream
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        # If a face appears on camera, draw a bounding box and snap it!
        if len(faces) > 0 and not self.face_detected:
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Lock the detection flag and save the snapshot data matrix
            self.face_detected = True
            st.session_state["auto_captured_frame"] = img
            
        return img

# --- 3. SIDEBAR SELECTION & LAYOUT ---
with st.sidebar:
    selected_language = st.selectbox(
        "Select Language / Chagua Lugha / Yier Dhok:",
        ["English", "Kiswahili", "Dholuo"]
    )
    lang = translations[selected_language]
    
    st.markdown("---")
    st.header(lang["scanner_title"])
    enable_camera = st.checkbox(lang["cam_on"])
    
    camera_mood_detected = ""
    photo_triggered = False
    
    if enable_camera:
        # Stream the webcam live directly into the app framework
        ctx = webrtc_streamer(key="face-detection", video_transformer_factory=FaceDetectorTransformer)
        
        if "auto_captured_frame" in st.session_state:
            st.success(" " + lang["scan_success"])
            photo_triggered = True
            
            # Analyze lighting / context from the captured frame matrix
            captured_img = st.session_state["auto_captured_frame"]
            mean_brightness = np.mean(captured_img)
            
            if mean_brightness < 50:
                camera_mood_detected = "Sad/Tired" if selected_language == "English" else ("Huzuni" if selected_language == "Kiswahili" else "Kuyo Oloi")
            else:
                camera_mood_detected = "Neutral/Expressive" if selected_language == "English" else ("Kawaida" if selected_language == "Kiswahili" else "Maber")
            
            st.info(f"{lang['active_ctx']} {camera_mood_detected}")
            
            # Provide a clear reset button if they want to capture their face again
            if st.button("Clear & Re-scan"):
                del st.session_state["auto_captured_frame"]
                st.rerun()

    st.markdown("---")
    st.header(lang["manual_mood"])
    user_mood = st.select_slider(lang["mood_label"], options=["😢 Overwhelmed", "😨 Anxious", "😐 Okay", "🙂 Good", "😄 Excellent"])
    final_detected_mood = camera_mood_detected if camera_mood_detected else user_mood

# --- 4. RENDER PAGE MAIN TITLES ---
st.title(lang["title"])
st.caption(lang["caption"])

# --- 5. CONNECT TO GEMINI CLIENT ---
try:
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
except Exception:
    st.error("API Key missing! Check your .env file.")
    st.stop()

# --- 6. CHAT HISTORY CORNERSTONE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# If a face was automatically locked on, trigger an instant AI checkup!
if photo_triggered and "last_auto_photo" not in st.session_state:
    st.session_state.messages = []
    st.session_state.last_auto_photo = camera_mood_detected
    
    camera_prompt = f"I just positioned my face in front of the scanner. It auto-captured a frame and categorized the layout as: {camera_mood_detected}."
    st.session_state.messages.append({"role": "user", "content": lang["cam_msg"]})
    
    system_instruction = (
        f"You are a warm, intuitive community mental health companion. You MUST speak entirely in {selected_language}. "
        f"The system automatically scanned the user's face and detected an expression that looks: {camera_mood_detected}. "
        f"Acknowledge their facial scan results gently. Keep it short (2-3 sentences), warm, and ask how they are feeling deep down."
    )
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=camera_prompt,
            config={"system_instruction": system_instruction}
        )
        st.session_state.messages.append({"role": "assistant", "content": response.text})
    except Exception:
        st.error("Server glitch. Please rerun.")

if not enable_camera and "last_auto_photo" in st.session_state:
    del st.session_state.last_auto_photo

# Print ongoing discussion boxes
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 7. TRANSLATED QUICK INTERACTIVE PANELS ---
st.markdown(f"### {lang['quick_tools']}")
col1, col2 = st.columns(2)
quick_prompt = ""

with col1:
    if st.button(lang["breath_btn"]):
        quick_prompt = f"{lang['breath_prompt']} {final_detected_mood}."
with col2:
    if st.button(lang["affirm_btn"]):
        quick_prompt = f"{lang['affirm_prompt']} {final_detected_mood}."

# --- 8. BASE CHAT RUNNER ---
user_input = st.chat_input(lang["input_placeholder"])
if quick_prompt:
    user_input = quick_prompt

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Crisis keyword checker
    crisis_keywords = ["suicide", "harm myself", "kujiua", "nataka kufa", "nego ra", "tho"]
    if any(word in user_input.lower() for word in crisis_keywords):
        st.error("Please reach out to Befrienders Kenya at +254 722 178 177 immediately.")
    else:
        with st.chat_message("assistant"):
            system_instruction = (
                f"You are a caring companion talking completely in {selected_language}. "
                f"Current context: User feels {final_detected_mood}. Keep answers warm and short (2-3 sentences)."
            )
            try:
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=user_input,
                    config={"system_instruction": system_instruction}
                )
                st.markdown(response.text)
                st.session_state.messages.append({"role": "assistant", "content": response.text})
            except Exception:
                st.error("Server glitch. Please rerun.")