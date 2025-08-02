import cv2
import numpy as np

def random_rotation(image):
    """
    Apply a random rotation to the image.
    """
    angle = np.random.uniform(-15, 15)
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated, angle

def random_scaling(image):
    """
    Apply a random scaling to the image.
    """
    scale = np.random.uniform(0.8, 1.2)
    (h, w) = image.shape[:2]
    resized = cv2.resize(image, (int(w * scale), int(h * scale)))
    return resized

def random_noise(image):
    """
    Add random noise to the image.
    """
    noise = np.random.normal(0, 20, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image

def augment_image(image):
    """
    Apply a sequence of data augmentation techniques to the image.
    """
    rotated_image, angle = random_rotation(image)
    scaled_image = random_scaling(rotated_image)
    noisy_image = random_noise(scaled_image)
    return noisy_image, angle
