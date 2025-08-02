import numpy as np

from src.rotators.deskew_rotator import align_by_deskew
from src.rotators.histogram_rotator import align_image
from src.rotators.tesseract_rotator import ImageRotate
from src.utils import rotate_image_cv2


def ensemble_rotation(image):
    """
    Combines the predictions from multiple models to determine the rotation angle.
    """
    # Get predictions from individual models
    img_rotator = ImageRotate()
    tesseract_angle = img_rotator.rotate_by_pytesseract(image)
    _, histogram_angle = align_image(image)
    _, deskew_angle = align_by_deskew(image)

    # Combine the predictions using the median
    angles = [tesseract_angle, histogram_angle, deskew_angle]
    median_angle = np.median(angles)

    # Rotate the image by the median angle
    rotated_image = rotate_image_cv2(image, median_angle)

    return rotated_image, median_angle
