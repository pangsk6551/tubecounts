import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from streamlit_image_coordinates import streamlit_image_coordinates

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Tube Counter",
    page_icon="🍩",
    layout="centered"
)

st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 0rem;
            padding-left: 0.5rem;
            padding-right: 0.5rem;
        }
        h1 { text-align: center; font-size: 1.5rem !important; }
    </style>
""", unsafe_allow_html=True)

st.title("🍩 Insulation Tube Counter")


# --- 2. SETUP STATE & MODEL ---

@st.cache_resource
def load_model():
    return YOLO("best.pt")


try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Initialize Session State
# Points now stores tuples of (x, y, w, h)
if "points" not in st.session_state:
    st.session_state["points"] = []
if "current_img_id" not in st.session_state:
    st.session_state["current_img_id"] = None
if "click_key" not in st.session_state:
    st.session_state["click_key"] = 0

# --- 3. SCOREBOARD ---
head_col1, head_col2 = st.columns([2, 1], gap="small")

with head_col1:
    metric_placeholder = st.empty()
    metric_placeholder.metric("Total Tubes", len(st.session_state["points"]))

with head_col2:
    if st.button("🗑️ Reset", use_container_width=True):
        st.session_state["points"] = []
        st.rerun()

# --- 4. INPUT SECTION ---
with st.expander("📸 Change Image / Settings", expanded=(st.session_state["current_img_id"] is None)):
    input_method = st.radio("Input:", ["Camera", "Upload"], horizontal=True, label_visibility="collapsed")

    img_file_buffer = None
    if input_method == "Camera":
        img_file_buffer = st.camera_input("Take a picture")
    else:
        img_file_buffer = st.file_uploader("Upload image", type=["jpg", "png"])

# --- 5. PROCESSING & DRAWING ---
if img_file_buffer is not None:
    # Open and Resize
    image = Image.open(img_file_buffer).convert("RGB")
    image.thumbnail((800, 800))

    file_id = f"{img_file_buffer.name}-{img_file_buffer.size}"

    # Run AI on new image
    if st.session_state["current_img_id"] != file_id:
        st.session_state["points"] = []
        st.session_state["current_img_id"] = file_id

        with st.spinner('Counting...'):
            results = model.predict(image, conf=0.15)
            # Store (x, y, width, height)
            for box in results[0].boxes.xywh:
                x, y, w, h = box
                st.session_state["points"].append((float(x), float(y), float(w), float(h)))

        metric_placeholder.metric("Total Tubes", len(st.session_state["points"]))
        st.session_state["click_key"] += 1

    # --- DRAWING THE DOTS & NUMBERS ---
    draw_img = image.copy()
    draw = ImageDraw.Draw(draw_img)

    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except IOError:
        font = ImageFont.load_default()

    # Draw loops
    for i, p in enumerate(st.session_state["points"]):
        x, y, w, h = p

        # Draw Ellipse matching the EXACT size of the detected tube
        # Bounding box is [x - w/2, y - h/2, x + w/2, y + h/2]
        draw.ellipse(
            [x - w / 2, y - h / 2, x + w / 2, y + h / 2],
            outline="blue", width=4
        )

        # Draw Number
        text = str(i + 1)
        try:
            left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
            text_w = right - left
            text_h = bottom - top
        except AttributeError:
            text_w, text_h = draw.textsize(text, font=font)

        draw.text((x - text_w / 2, y - text_h / 2 - 2), text, fill="blue", font=font, stroke_width=1,
                  stroke_fill="white")

    # --- INTERACTION ---
    st.caption("👇 Tap INSIDE a blue box to Remove. Tap empty space to Add.")

    value = streamlit_image_coordinates(
        draw_img,
        key=f"pil_{st.session_state['click_key']}",
        use_column_width=True
    )

    # --- HIT BOX LOGIC ---
    if value:
        click_x = value["x"]
        click_y = value["y"]

        target_point = None

        # Check if click is INSIDE any existing box
        for p in st.session_state["points"]:
            px, py, pw, ph = p

            # Check boundaries (Hit Box Test)
            left = px - pw / 2
            right = px + pw / 2
            top = py - ph / 2
            bottom = py + ph / 2

            if left <= click_x <= right and top <= click_y <= bottom:
                target_point = p
                break  # Found the target!

        if target_point:
            # REMOVE MODE: User tapped INSIDE a box
            st.session_state["points"].remove(target_point)

            metric_placeholder.metric("Total Tubes", len(st.session_state["points"]))
            st.session_state["click_key"] += 1
            st.rerun()
        else:
            # ADD MODE: User tapped empty space
            # Calculate average size of existing tubes to make the new one match
            if len(st.session_state["points"]) > 0:
                avg_w = np.mean([p[2] for p in st.session_state["points"]])
                avg_h = np.mean([p[3] for p in st.session_state["points"]])
            else:
                avg_w, avg_h = 50.0, 50.0  # Default if list is empty

            st.session_state["points"].append((click_x, click_y, avg_w, avg_h))

            metric_placeholder.metric("Total Tubes", len(st.session_state["points"]))
            st.session_state["click_key"] += 1
            st.rerun()

    metric_placeholder.metric("Total Tubes", len(st.session_state["points"]))

else:

    st.info("Please upload a photo to start.")
