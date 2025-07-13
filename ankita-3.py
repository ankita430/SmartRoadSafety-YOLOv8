import PIL
import streamlit as st
from ultralytics import YOLO
import numpy as np
import time

# Path to your trained YOLO model
model_path = r"F:\SIT_MTECH\SEM3\ankita_best.pt"

# Set up page layout and configuration
st.set_page_config(page_title="Speed Bump Detection", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for styling and animations
st.markdown("""
    <style>
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #f4f4f4;
    }
    .stSidebar header, .stSidebar h2 {
        color: #2E86C1;
    }

    /* Title and subtitle */
    h1 {
        font-family: 'Helvetica Neue', sans-serif;
        color: #2980B9;
        text-align: center;
        margin-top: -10px;
    }
    h3, p {
        color: #2C3E50;
    }

    /* Formal welcome message styling */
    .welcome-message {
        font-size: 20px;
        font-weight: bold;
        color: #34495E;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 20px;
        font-family: 'Arial', sans-serif;
    }

    /* Animated button */
    .animated-button {
        display: inline-block;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
        color: white;
        background-color: #2980B9;
        border-radius: 5px;
        cursor: pointer;
        transition: background-color 0.3s, transform 0.3s;
    }
    .animated-button:hover {
        background-color: #3498DB;
        transform: scale(1.05);
    }

    /* Slide-in animation for results */
    .slide-in {
        animation: slideIn 1s ease-out forwards;
    }
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(-100%);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }

    /* Footer styling */
    .footer {
        text-align: center;
        font-size: 14px;
        color: gray;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Formal welcome message
st.markdown("<div class='welcome-message'>Welcome to the Speed Bump Detection Project</div>", unsafe_allow_html=True)

# Main title with icon
st.markdown("<h1>🚧 Speed Bump Detection System</h1>", unsafe_allow_html=True)

# Instructions
st.markdown("<h3>Instructions:</h3>", unsafe_allow_html=True)
st.markdown("""
- **Step 1:** Upload an image in the sidebar.
- **Step 2:** Adjust the confidence level if needed.
- **Step 3:** Click **Detect Speed Bump** to analyze the image.
""")

# Sidebar for image upload and settings
with st.sidebar:
    st.markdown("<h2>Image Upload & Settings</h2>", unsafe_allow_html=True)
    source_img = st.file_uploader("📤 Upload an Image", type=("jpg", "jpeg", "png"))
    confidence = float(st.slider("🔍 Confidence Threshold", 0.1, 1.0, 0.5))  # Default 50% confidence

# Load and display uploaded image
col1, col2 = st.columns(2)
with col1:
    if source_img:
        uploaded_image = PIL.Image.open(source_img).resize((640, 640))  # Resize for YOLO model
        st.image(uploaded_image, caption="Uploaded Image", use_container_width=True, output_format="JPEG")
    else:
        st.info("Please upload an image to start detection.", icon="ℹ️")

# Load the YOLO model
try:
    model = YOLO(model_path)
except Exception as ex:
    st.error(f"🚨 Unable to load the model from the path: {model_path}")
    st.error(ex)

# Function to filter overlapping bounding boxes
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    return intersection / union

def filter_duplicates(boxes, iou_threshold=0.4):
    filtered_boxes = []
    boxes_array = np.array([box.xyxy[0].tolist() for box in boxes])
    for box in boxes_array:
        if all(calculate_iou(box, other_box[:4]) < iou_threshold for other_box in filtered_boxes):
            filtered_boxes.append(box)
    return filtered_boxes

# Detection button
detect_button = st.sidebar.button("🚧 Detect Speed Bump", help="Click to start detection!")

# Perform detection and display results
with col2:
    if detect_button:
        if source_img:
            with st.spinner("🔍 Analyzing the image..."):
                # Perform prediction
                res = model.predict(uploaded_image, conf=confidence, iou=0.4)
                boxes = res[0].boxes

                # Ensure boxes are processed safely
                if boxes:
                    # Filter duplicates
                    filtered_boxes = filter_duplicates(boxes)

                    # Display bounding boxes
                    if filtered_boxes:
                        res_plotted = res[0].plot()[:, :, ::-1]
                        st.image(res_plotted, caption="Detection Result", use_container_width=True, output_format="JPEG")
                        st.success(f"✅ Detected {len(filtered_boxes)} speed bumps!")
                    else:
                        st.warning("🚫 No speed bumps detected.", icon="⚠️")

                    # Show detailed results
                    with st.expander("📋 Detection Details"):
                        for box in filtered_boxes:
                            x_min, y_min, x_max, y_max = box[:4]
                            st.write(f"Coordinates: {[x_min, y_min, x_max, y_max]}")
                else:
                    st.warning("🚫 No bounding boxes detected.", icon="⚠️")
        else:
            st.warning("🚫 Please upload an image first.", icon="⚠️")

# Footer with information
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p class='footer'>Developed with ❤️ using YOLOv8 and Streamlit</p>", unsafe_allow_html=True)
