import cv2
import imutils
import pytesseract
from pytesseract import Output


class ImageRotate:
    def rotate_by_pytesseract(self, image) -> float:
        """Detects image orientation using Tesseract OCR."""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pytesseract.image_to_osd(rgb, output_type=Output.DICT)
        rotation_angle = results["rotate"]
        return rotation_angle

    def rotate_image(self, image, rotation_angle):
        """Rotates the image by the specified angle using imutils."""
        rotated = imutils.rotate_bound(image, angle=rotation_angle)
        return rotated
