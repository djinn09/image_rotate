import io

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.evaluation import calculate_accuracy, calculate_mae, calculate_rmse
from src.realtime_adaptation import update_ensemble_weights
from src.rotators.deskew_rotator import align_by_deskew
from src.rotators.east_hough_rotator import rotate_by_east_hough
from src.rotators.ensemble_rotator import ensemble_rotation
from src.rotators.histogram_rotator import align_image
from src.rotators.tesseract_rotator import ImageRotate
from src.rl_env import OrientationEnv

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
st.write("- **ensemble**: Combines the predictions from multiple models.")
st.write("- **rl**: Uses a trained reinforcement learning agent to predict the orientation.")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
method = st.radio(
    "Select Rotation Method:",
    ("cv2-histogram", "tesseract", "deskew", "east-hough", "ensemble", "rl"),
    horizontal=True,
)

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
            elif method == "ensemble":
                rotated_img, angle, _ = ensemble_rotation(
                    image_cv, st.session_state.ensemble_weights
                )
                st.write(f"Ensemble detected rotation angle: {angle} degrees")
            elif method == "rl":
                q_table = np.load("q_table.npy")
                action = np.argmax(q_table[0])
                rotated_img = ImageRotate().rotate_image(image_cv, action)
                st.write(
                    f"Reinforcement learning agent predicted rotation angle: {action} degrees"
                )
                angle = action
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
            mime=mime,
        )

        # Initialize session state for ensemble weights
        if "ensemble_weights" not in st.session_state:
            st.session_state.ensemble_weights = np.array([1.0, 1.0, 1.0]) / 3.0

        # --- Evaluation Section ---
        st.write("---")
        st.header("Evaluation")
        st.write(
            "Note: The MAE and RMSE metrics are calculated based on the single image uploaded above."
        )
        ground_truth_angle = st.number_input(
            "Enter the ground truth angle (in degrees):", value=0.0
        )
        if st.button("Evaluate All Models"):
            st.write("**Evaluation Results:**")
            predicted_angles = []
            ground_truth_angles = [ground_truth_angle]

            # Tesseract
            img_rotator = ImageRotate()
            tesseract_angle = img_rotator.rotate_by_pytesseract(image_cv)
            predicted_angles.append(tesseract_angle)
            tesseract_accuracy = calculate_accuracy(tesseract_angle, ground_truth_angle)
            st.write(f"**Tesseract**")
            st.write(f"- Accuracy: {tesseract_accuracy}")

            # Histogram
            _, histogram_angle = align_image(image_cv)
            predicted_angles.append(histogram_angle)
            histogram_accuracy = calculate_accuracy(histogram_angle, ground_truth_angle)
            st.write(f"**Histogram**")
            st.write(f"- Accuracy: {histogram_accuracy}")

            # Deskew
            _, deskew_angle = align_by_deskew(image_cv)
            predicted_angles.append(deskew_angle)
            deskew_accuracy = calculate_accuracy(deskew_angle, ground_truth_angle)
            st.write(f"**Deskew**")
            st.write(f"- Accuracy: {deskew_accuracy}")

            # EAST + Hough
            east_hough_angle, _ = rotate_by_east_hough(image_cv)
            predicted_angles.append(east_hough_angle)
            east_hough_accuracy = calculate_accuracy(
                east_hough_angle, ground_truth_angle
            )
            st.write(f"**EAST + Hough**")
            st.write(f"- Accuracy: {east_hough_accuracy}")

            # Ensemble
            rotated_img, ensemble_angle, individual_angles = ensemble_rotation(
                image_cv, st.session_state.ensemble_weights
            )
            predicted_angles.append(ensemble_angle)
            ensemble_accuracy = calculate_accuracy(ensemble_angle, ground_truth_angle)
            st.write(f"**Ensemble**")
            st.write(f"- Accuracy: {ensemble_accuracy}")

            # Calculate and display MAE and RMSE
            mae = calculate_mae(np.array(predicted_angles), np.array(ground_truth_angles))
            rmse = calculate_rmse(
                np.array(predicted_angles), np.array(ground_truth_angles)
            )
            st.write("---")
            st.write(f"**Overall Metrics:**")
            st.write(f"- MAE: {mae:.2f} degrees")
            st.write(f"- RMSE: {rmse:.2f} degrees")

        # --- Real-time Adaptation Section ---
        st.write("---")
        st.header("Real-time Adaptation")
        st.write(
            "Provide feedback on the corrected image to update the ensemble model."
        )
        corrected_angle = st.number_input(
            "Enter the corrected angle (in degrees):", value=0.0
        )
        if st.button("Update Ensemble Model"):
            _, _, individual_angles = ensemble_rotation(
                image_cv, st.session_state.ensemble_weights
            )
            st.session_state.ensemble_weights = update_ensemble_weights(
                st.session_state.ensemble_weights, individual_angles, corrected_angle
            )
            st.success("Ensemble model updated successfully!")
            st.write("New ensemble weights:")
            st.write(st.session_state.ensemble_weights)

    except Exception as e:
        st.error(f"An error occurred: {e}")
