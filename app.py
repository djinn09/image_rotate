import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

from src.rotators.tesseract_rotator import ImageRotate
from src.rotators.histogram_rotator import align_image
from src.rotators.deskew_rotator import align_by_deskew
from src.rotators.east_hough_rotator import rotate_by_east_hough
from src.utils import rotate_image_cv2

# Set page config to wide layout
st.set_page_config(layout="wide")

# --- Streamlit UI ---
st.title("Image Rotation with Multiple Methods")
st.write("Upload an image and select a rotation method to correct its orientation.")
st.write("**Available Methods:**")
st.write("- **cv2-histogram**: Analyzes histogram sharpness to align the image.")
st.write("- **tesseract**: Uses Tesseract OCR to detect orientation.")
st.write("- **deskew**: Corrects skew using the deskew library.")
st.write("- **east-hough**: Combines EAST text detection with Hough Transform.")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
method = st.radio("Select Rotation Method:", ("cv2-histogram", "tesseract", "deskew", "east-hough"), horizontal=True)

if uploaded_image is not None:
    try:
        # Open and display original image
        image = Image.open(uploaded_image)
        original_display = image.resize((500, 500))
        col1, col2 = st.columns(2)
        col1.image(original_display, caption="Original Image", use_column_width=True)

        # Convert to OpenCV format
        image_np = np.array(image)
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Process image based on selected method
        with st.spinner(f"Processing with {method} method..."):
            if method == "tesseract":
                img_rotator = ImageRotate()
                angle = img_rotator.rotate_by_pytesseract(image_cv)
                rotated_img = img_rotator.rotate_image(image_cv, angle)
                st.write(f"Tesseract detected rotation angle: {angle} degrees")
            elif method == "cv2-histogram":
                rotated_img, angle = align_image(image_cv)
                st.write(f"Histogram-based rotation angle: {angle} degrees")
            elif method == "deskew":
                rotated_img, angle = align_by_deskew(image_cv)
                st.write(f"Deskew detected skew angle: {angle} degrees")
            elif method == "east-hough":
                angle, rotated_img = rotate_by_east_hough(image_cv)
                st.write(f"EAST + Hough detected rotation angle: {angle} degrees")
            else:
                st.error("Invalid method selected.")
                st.stop()

        # Convert rotated image to RGB for display
        rotated_img_rgb = cv2.cvtColor(rotated_img, cv2.COLOR_BGR2RGB)
        rotated_img_pil = Image.fromarray(rotated_img_rgb)
        rotated_display = rotated_img_pil.resize((500, 500))
        col2.image(rotated_display, caption="Rotated Image", use_column_width=True)

        # Download button for rotated image
        format = st.selectbox("Download format", ["PNG", "JPEG", "BMP"])
        buf = io.BytesIO()
        if format == "JPEG" and rotated_img_pil.mode == "RGBA":
            rotated_img_pil = rotated_img_pil.convert("RGB")
        rotated_img_pil.save(buf, format=format)
        byte_im = buf.getvalue()
        mime = f"image/{format.lower()}"
        st.download_button(
            label=f"Download rotated image as {format}",
            data=byte_im,
            file_name=f"rotated_image.{format.lower()}",
            mime=mime
        )
    except Exception as e:
        st.error(f"An error occurred: {e}")
