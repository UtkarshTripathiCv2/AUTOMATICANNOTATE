import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os
import tempfile
import shutil
import zipfile

# --- Configuration ---
st.set_page_config(
    page_title="YOLO Annotation Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Model Loading ---
@st.cache_resource
def load_yolo_model(model_path):
    """Loads the YOLO model from the specified path, caching it for performance."""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- Core Processing Function ---
def process_frame(frame, model):
    """
    Performs object detection on a single frame (image) and returns the annotated frame
    and a list of YOLO-formatted label strings.
    """
    results = model(frame, verbose=False)
    result = results[0]
    
    annotated_frame = result.plot()
    
    labels = []
    if len(result.boxes) > 0:
        boxes = result.boxes.xywhn
        classes = result.boxes.cls
        for i in range(len(boxes)):
            class_id = int(classes[i])
            x_center, y_center, width, height = boxes[i]
            labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
    return annotated_frame, labels

# --- Session State Initialization ---
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = []

# --- UI Sidebar ---
with st.sidebar:
    st.title("⚙ Configuration")
    
    model_path = st.text_input("YOLO Model Path", "dog.pt")
    model = load_yolo_model(model_path)

    st.markdown("---")
    
    dataset_name = st.text_input("Enter Dataset Name", "my_dataset")
    
    if st.session_state.processed_data:
        st.info(f"Collected {len(st.session_state.processed_data)} items for '{dataset_name}'.")
        
        # --- Download Logic ---
        if st.button("📦 Prepare & Download Dataset"):
            with tempfile.TemporaryDirectory() as tmpdir:
                images_dir = os.path.join(tmpdir, dataset_name, 'images')
                labels_dir = os.path.join(tmpdir, dataset_name, 'labels')
                os.makedirs(images_dir, exist_ok=True)
                os.makedirs(labels_dir, exist_ok=True)
                
                for i, data in enumerate(st.session_state.processed_data):
                    original_image = data['original_image']
                    labels = data['labels']
                    base_filename = data['filename']
                    
                    # Save image
                    img_path = os.path.join(images_dir, base_filename)
                    original_image.save(img_path)
                    
                    # Save labels
                    if labels:
                        label_filename = os.path.splitext(base_filename)[0] + '.txt'
                        label_path = os.path.join(labels_dir, label_filename)
                        with open(label_path, 'w') as f:
                            f.write("\n".join(labels))

                # Create Zip file
                zip_path = os.path.join(tempfile.gettempdir(), f"{dataset_name}.zip")
                shutil.make_archive(zip_path.replace('.zip', ''), 'zip', os.path.join(tmpdir, dataset_name))

                with open(zip_path, "rb") as fp:
                    st.download_button(
                        label="✅ Click to Download ZIP",
                        data=fp,
                        file_name=f"{dataset_name}.zip",
                        mime="application/zip",
                        key='download_zip_button'
                    )
                    st.success("Zip file ready! Click above to download.")

    if st.button("🗑 Clear Collected Data"):
        st.session_state.processed_data = []
        st.rerun()

# --- Main App Interface ---
st.title("🤖 YOLO Annotation Assistant")
st.write("Upload images or use your webcam to detect objects. The app will generate YOLO labels and package them into a downloadable zip file for training.")

tab1, tab2 = st.tabs(["🖼 Image Upload", "📹 Live Video"])

# --- Image Upload Tab ---
with tab1:
    uploaded_files = st.file_uploader(
        "Choose one or more images...",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    
    if uploaded_files and model:
        st.info("Processing uploaded images...")
        
        for uploaded_file in uploaded_files:
            original_image = Image.open(uploaded_file).convert("RGB")
            frame = np.array(original_image)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            annotated_frame, labels = process_frame(frame_bgr, model)

            st.image(annotated_frame, caption=f"Annotated: {uploaded_file.name}", use_container_width=True)

            # Store results in session state
            st.session_state.processed_data.append({
                'original_image': original_image,
                'labels': labels,
                'filename': uploaded_file.name
            })
        
        st.success(f"Processed and added {len(uploaded_files)} images to the collection. Check the sidebar to download.")
        st.rerun() # Rerun to update the sidebar count immediately

# --- Live Video Tab ---
with tab2:
    st.header("Live Webcam Feed")
    run = st.checkbox('Start Webcam', key='run_webcam')
    
    if run and model:
        FRAME_WINDOW = st.image([])
        camera = cv2.VideoCapture(0)
        
        if not camera.isOpened():
            st.error("Could not open webcam. Please grant permission.")
        else:
            while run:
                success, frame_bgr = camera.read()
                if not success:
                    st.error("Failed to capture frame from webcam.")
                    break
                
                annotated_frame, labels = process_frame(frame_bgr, model)
                FRAME_WINDOW.image(annotated_frame, channels="BGR")
                
                # We need a unique key for the button inside the loop
                capture_key = f"capture_{camera.get(cv2.CAP_PROP_POS_FRAMES)}" 
                if st.button("📸 Capture & Add Frame", key=capture_key):
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    original_image = Image.fromarray(frame_rgb)
                    
                    # Create a unique filename
                    timestamp = f"frame_{len(st.session_state.processed_data) + 1}.jpg"
                    
                    st.session_state.processed_data.append({
                        'original_image': original_image,
                        'labels': labels,
                        'filename': timestamp
                    })
                    st.success(f"Frame '{timestamp}' captured and added to collection!")

            camera.release()
    elif not model:
        st.warning("Please provide a valid model path in the sidebar to start the webcam.")
