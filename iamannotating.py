import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os
import tempfile
import shutil

# --- Configuration ---
st.set_page_config(
    page_title="YOLO Annotation Assistant",
    page_icon="🤖",
    layout="wide"
)

# --- Path Fixing ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(BASE_DIR, "dog.pt")

# --- Model Loading ---
@st.cache_resource
def load_yolo_model(model_path):
    try:
        if not os.path.exists(model_path):
            st.error(f"Model not found at: {model_path}")
            return None
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- Core Processing Function ---
def process_frame(frame, model):
    # result.plot() returns BGR image
    results = model(frame, verbose=False)
    result = results[0]
    annotated_frame = result.plot()
    
    labels = []
    if result.boxes:
        # xywhn = center_x, center_y, width, height (normalized 0-1)
        boxes = result.boxes.xywhn
        classes = result.boxes.cls
        for i in range(len(boxes)):
            c_id = int(classes[i])
            x, y, w, h = boxes[i]
            labels.append(f"{c_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
            
    return annotated_frame, labels

# --- Session State ---
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = []

# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Settings")
    m_path = st.text_input("YOLO Model Path", DEFAULT_MODEL)
    model = load_yolo_model(m_path)
    
    dataset_name = st.text_input("Dataset Name", "my_dataset")
    
    st.write(f"📁 Items collected: **{len(st.session_state.processed_data)}**")
    
    if st.button("📦 Download Zip"):
        if not st.session_state.processed_data:
            st.warning("No data to download.")
        else:
            with tempfile.TemporaryDirectory() as tmpdir:
                img_dir = os.path.join(tmpdir, 'images')
                lbl_dir = os.path.join(tmpdir, 'labels')
                os.makedirs(img_dir); os.makedirs(lbl_dir)
                
                for i, data in enumerate(st.session_state.processed_data):
                    fname = data['filename']
                    # Save Image
                    data['original_image'].save(os.path.join(img_dir, fname))
                    # Save Label
                    if data['labels']:
                        l_name = os.path.splitext(fname)[0] + ".txt"
                        with open(os.path.join(lbl_dir, l_name), 'w') as f:
                            f.write("\n".join(data['labels']))
                
                zip_base = os.path.join(tempfile.gettempdir(), dataset_name)
                shutil.make_archive(zip_base, 'zip', tmpdir)
                
                with open(f"{zip_base}.zip", "rb") as f:
                    st.download_button("⬇️ Download Now", f, f"{dataset_name}.zip", "application/zip")

    if st.button("🗑️ Clear All"):
        st.session_state.processed_data = []
        st.rerun()

# --- UI ---
st.title("🤖 YOLO Annotation Assistant")

tab1, tab2 = st.tabs(["🖼️ Image Upload", "📹 Live Webcam"])

with tab1:
    files = st.file_uploader("Upload Images", type=["jpg","png","jpeg"], accept_multiple_files=True)
    if files and model:
        for f in files:
            # Check if already processed to avoid duplicates on rerun
            if not any(d['filename'] == f.name for d in st.session_state.processed_data):
                orig = Image.open(f).convert("RGB")
                # Convert for OpenCV
                frame_bgr = cv2.cvtColor(np.array(orig), cv2.COLOR_RGB2BGR)
                anno, labs = process_frame(frame_bgr, model)
                
                st.image(anno, caption=f.name, channels="BGR")
                
                st.session_state.processed_data.append({
                    'original_image': orig,
                    'labels': labs,
                    'filename': f.name
                })
        st.success("Images processed!")

with tab2:
    st.write("Webcam requires local browser permissions.")
    # Simple webcam implementation
    run_cam = st.checkbox("Toggle Camera")
    if run_cam and model:
        cam = cv2.VideoCapture(0)
        st_frame = st.empty()
        while run_cam:
            ret, frame = cam.read()
            if not ret: break
            anno, _ = process_frame(frame, model)
            st_frame.image(anno, channels="BGR")
        cam.release()
