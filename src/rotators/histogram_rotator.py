import numpy as np
from joblib import Parallel, delayed

from src.utils import rotate_image_cv2


def eval_image(image):
    """Evaluates the sharpness of an image's horizontal histogram."""
    if image.size == 0:
        raise ValueError("Empty image provided.")
    hist = np.sum(np.mean(image, axis=1), axis=1)
    bef, aft = 0, 0
    err = 0.0
    for pos in range(hist.shape[0]):
        if pos == aft:
            bef = pos
            while aft + 1 < hist.shape[0] and abs(hist[aft + 1] - hist[pos]) >= abs(
                hist[aft] - hist[pos]
            ):
                aft += 1
        err += min(abs(hist[bef] - hist[pos]), abs(hist[aft] - hist[pos]))
    if err <= 0:
        raise ValueError("Invalid histogram evaluation.")
    return err


def sweep_angles(image, start_angle=-10, end_angle=10, step=0.25):
    """Sweeps through a range of angles to evaluate image sharpness."""
    num_angles = int((end_angle - start_angle) / step) + 1

    def process_angle(i):
        angle = start_angle + i * step
        rotated = rotate_image_cv2(image, angle)
        err = eval_image(rotated)
        return angle, err

    results = Parallel(n_jobs=-1)(delayed(process_angle)(i) for i in range(num_angles))
    return np.array(results)


def find_alignment_angle(image):
    """Finds the angle that aligns the image based on its horizontal histogram."""
    results = sweep_angles(image)
    best_gain = 0
    best_angle = 0.0
    for i in range(2, results.shape[0] - 2):
        ave = np.mean(results[i - 2 : i + 3, 1])
        gain = ave - results[i, 1]
        if gain > best_gain:
            best_gain = gain
            best_angle = results[i, 0]
    return best_angle


def align_image(image):
    """Aligns an image based on its horizontal histogram."""
    angle = find_alignment_angle(image)
    return rotate_image_cv2(image, angle), angle
