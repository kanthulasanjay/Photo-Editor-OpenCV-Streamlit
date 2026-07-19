import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# ----------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="PixelForge — Photo Editor",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# CUSTOM STYLING
# ----------------------------------------------------------------------
st.markdown("""
<style>

.main-header{
background:linear-gradient(135deg,#0f2027,#203a43,#2c5364,#6a11cb,#2575fc);
padding:45px;
border-radius:30px;
text-align:center;
box-shadow:0 15px 40px rgba(0,0,0,0.3);
margin-bottom:35px;
animation: fadeIn 1s ease-in-out;
}

.main-header h1{
font-size:60px;
font-weight:900;
color:white;
margin-bottom:10px;
}

.main-header h3{
font-size:24px;
font-weight:400;
color:#EAF4FF;
margin-bottom:20px;
}

.features{
display:flex;
justify-content:center;
gap:15px;
flex-wrap:wrap;
}

.features span{
background:rgba(255,255,255,0.15);
padding:10px 18px;
border-radius:30px;
color:white;
font-size:16px;
}

@keyframes fadeIn{
from{opacity:0;transform:translateY(-25px);}
to{opacity:1;transform:translateY(0);}
}

</style>

<div class="main-header">

<h1>🎨 PixelForge Studio</h1>

<h3>Professional AI Photo Editing Experience</h3>

<div class="features">
<span>📤 Upload</span>
<span>🌞 Brightness</span>
<span>🎚 Contrast</span>
<span>🌈 Filters</span>
<span>🖼 Portrait Blur</span>
<span>✨ Sharpen</span>
<span>📥 Download</span>
</div>

</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# HERO HEADER
# ----------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>🖼️ PixelForge</h1>
    <p>Resize, adjust, filter, and transform your photos — all in your browser.</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------
def to_display(img_array):
    """Ensure array is safe to hand to st.image (uint8, valid range)."""
    return np.clip(img_array, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------
# UPLOAD
# ----------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload an image to get started",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG",
)

if uploaded_file is None:
    st.info("👆 Upload an image above to unlock the editing tools.")
    st.stop()

# Work in RGB throughout (PIL gives RGB; avoid BGR/RGB mixups from cv2)
image = Image.open(uploaded_file).convert("RGB")
original_rgb = np.array(image)

# ----------------------------------------------------------------------
# SIDEBAR — CONTROLS
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎛️ Controls")

    with st.expander("📐 Resize", expanded=True):
        width = st.slider("Width", 100, 2000, original_rgb.shape[1])
        height = st.slider("Height", 100, 2000, original_rgb.shape[0])
        keep_aspect = st.checkbox("Lock aspect ratio", value=False)

    with st.expander("☀️ Brightness & Contrast", expanded=True):
        brightness = st.slider("Brightness", -100, 100, 0)
        contrast = st.slider("Contrast", 0.5, 3.0, 1.0, step=0.05)

    with st.expander("🎨 Filters"):
        grayscale = st.checkbox("Grayscale")
        blur = st.checkbox("Blur")
        sharpen = st.checkbox("Sharpen")
        warm = st.checkbox("Warm filter")
        portrait = st.checkbox("Portrait blur")

    with st.expander("✨ Extra Effects"):
        edge = st.checkbox("Edge detection")
        rotate = st.slider("Rotate (°)", 0, 360, 0)

# ----------------------------------------------------------------------
# RESIZE
# ----------------------------------------------------------------------
if keep_aspect:
    ratio = original_rgb.shape[1] / original_rgb.shape[0]
    height = int(width / ratio)

resized_img = cv2.resize(original_rgb, (width, height))

# ----------------------------------------------------------------------
# BRIGHTNESS / CONTRAST
# ----------------------------------------------------------------------
edited = cv2.convertScaleAbs(resized_img, alpha=contrast, beta=brightness)

# ----------------------------------------------------------------------
# FILTERS
# ----------------------------------------------------------------------
is_single_channel = False

if warm and edited.ndim == 3:
    increase = np.array([80, 40, 10], dtype=np.int16)  # more R, some G, little B (RGB order)
    edited = to_display(edited.astype(np.int16) + increase)

if grayscale:
    edited = cv2.cvtColor(edited, cv2.COLOR_RGB2GRAY)
    is_single_channel = True

if blur:
    edited = cv2.GaussianBlur(edited, (15, 15), 0)

if sharpen:
    kernel = np.array([[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]])
    edited = cv2.filter2D(edited, -1, kernel)

if portrait and not is_single_channel:
    mask = np.zeros(edited.shape[:2], dtype=np.uint8)
    h, w = mask.shape
    cv2.circle(mask, (w // 2, h // 2), min(w, h) // 4, 255, -1)
    blurred = cv2.GaussianBlur(edited, (25, 25), 0)
    edited = np.where(mask[:, :, None] == 255, edited, blurred)
elif portrait and is_single_channel:
    st.sidebar.caption("⚠️ Portrait blur skipped (needs a color image — disable Grayscale).")

# ----------------------------------------------------------------------
# EXTRA EFFECTS
# ----------------------------------------------------------------------
if edge:
    gray_for_edge = edited if is_single_channel else cv2.cvtColor(edited, cv2.COLOR_RGB2GRAY)
    edited = cv2.Canny(gray_for_edge, 100, 200)
    is_single_channel = True

if rotate != 0:
    (h, w) = edited.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, rotate, 1.0)
    border_value = 0 if is_single_channel else (0, 0, 0)
    edited = cv2.warpAffine(edited, matrix, (w, h), borderValue=border_value)

edited = to_display(edited)

# ----------------------------------------------------------------------
# METRICS ROW
# ----------------------------------------------------------------------
m1, m2, m3, m4 = st.columns(4)
m1.metric("Original size", f"{original_rgb.shape[1]} × {original_rgb.shape[0]}")
m2.metric("Edited size", f"{edited.shape[1]} × {edited.shape[0]}")
m3.metric("File type", uploaded_file.type.split("/")[-1].upper())
active_filters = sum([grayscale, blur, sharpen, warm, portrait, edge]) + (1 if rotate else 0)
m4.metric("Effects applied", active_filters)

st.markdown("<br>", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# BEFORE / AFTER PANELS
# ----------------------------------------------------------------------
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="panel-title">📷 Original</div>', unsafe_allow_html=True)
    st.image(original_rgb, use_container_width=True)

with col2:
    st.markdown('<div class="panel-title">✨ Edited</div>', unsafe_allow_html=True)
    st.image(edited, use_container_width=True, clamp=True)

# ----------------------------------------------------------------------
# DOWNLOAD
# ----------------------------------------------------------------------
result = Image.fromarray(edited)
buf = io.BytesIO()
result.save(buf, format="PNG")

st.markdown("<br>", unsafe_allow_html=True)
dcol1, dcol2, dcol3 = st.columns([1, 1, 1])
with dcol2:
    st.download_button(
        label="⬇️ Download Edited Image",
        data=buf.getvalue(),
        file_name="edited_image.png",
        mime="image/png",
    )

st.markdown(
    '<div class="footer-note">Built with Streamlit + OpenCV</div>',
    unsafe_allow_html=True,
)