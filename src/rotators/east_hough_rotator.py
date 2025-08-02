import cv2
import numpy as np
import streamlit as st
from src.utils import rotate_image_cv2

def rotate_by_east_hough(image, east_model_path="frozen_east_text_detection.pb"):
    """Detects text orientation using EAST model and Hough Transform, returning the rotation angle."""
    try:
        # Load EAST model
        net = cv2.dnn.readNet(east_model_path)
        # Prepare image for EAST
        blob = cv2.dnn.blobFromImage(image, 1.0, (320, 320), (123.68, 116.78, 103.94), swapRB=True, crop=False)
        net.setInput(blob)
        scores, geometry = net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])

        # Decode EAST output to get bounding boxes
        (numRows, numCols) = scores.shape[2:4]
        rects = []
        confidences = []
        for y in range(numRows):
            scoresData = scores[0, 0, y]
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]
            for x in range(numCols):
                if scoresData[x] < 0.5:
                    continue
                (offsetX, offsetY) = (x * 4.0, y * 4.0)
                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)
                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]
                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)
                rects.append((startX, startY, endX, endY))
                confidences.append(scoresData[x])

        # Apply non-maxima suppression
        boxes = cv2.dnn.NMSBoxes(rects, confidences, 0.5, 0.4)
        angles = []

        # Calculate angles from bounding boxes
        for i in boxes:
            (startX, startY, endX, endY) = rects[i]
            angle = np.arctan2(endY - startY, endX - startX)
            angles.append(np.degrees(angle))

        # Determine median angle
        median_angle = np.median(angles) if angles else 0
        return median_angle, rotate_image_cv2(image, median_angle)
    except Exception as e:
        st.error(f"EAST + Hough method failed: {e}. Defaulting to 0 angle.")
        return 0, image
