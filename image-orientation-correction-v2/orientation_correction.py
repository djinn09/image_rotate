import cv2
import imutils
import numpy as np
from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
import config

class Orientation:

    def __init__(self, image_path,file_properties ,conf_threshold=50, lang='eng'):

        self.image_path     = image_path
        self.image          = cv2.imread(image_path)
        self.file_properties = file_properties
        self.conf_threshold = int(conf_threshold)

        self.timer = {'net': 0, 'restore': 0, 'nms': 0}
        self.text = {}
        self.lang = lang

    def rotate_bound(self,image, angle):
        (h, w) = image.shape[:2]
        (cX, cY) = (w / 2, h / 2)

        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        return cv2.warpAffine(image, M, (nW, nH),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    def east_detect(self,image,args):

        (H, W) = image.shape[:2]

        (newW, newH) = (args["width"], args["height"])
        rW = W / float(newW)
        rH = H / float(newH)

        image = cv2.resize(image, (newW, newH))
        (H, W) = image.shape[:2]

        layerNames = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"]

        net = cv2.dnn.readNet(args["east"])

        blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
            (123.68, 116.78, 103.94), swapRB=True, crop=False)
        net.setInput(blob)
        (scores, geometry) = net.forward(layerNames)

        (numRows, numCols) = scores.shape[2:4]
        angl = []

        for y in range(0, numRows):

            scoresData = scores[0, 0, y]
            anglesData = geometry[0, 4, y]

            for x in range(0, numCols):
                if scoresData[x] < args["min_confidence"]:
                    continue

                angle = anglesData[x]
                angl.append(angle*180/(np.pi))

        return np.median(angl)

    def east(self,image_path,args):

        image = cv2.imread(image_path)
        angle = Orientation.east_detect(self,image,args)

        return image,angle

    def hough_transforms(self,image):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.GaussianBlur(gray,(11,11),0)
        edges = canny(thresh)
        tested_angles = np.deg2rad(np.arange(0.1, 180.0))
        h, theta, d = hough_line(edges, theta=tested_angles)
        accum, angles, dists = hough_line_peaks(h, theta, d)

        return accum, angles, dists

    def east_hough_line(self,image,args):
        image,angle = Orientation.east(self,image,args)
        h, theta, d = Orientation.hough_transforms(self,image)
        theta = np.rad2deg(np.pi/2-theta)
        margin = args['margin_tollerance']
        low_thresh = angle-margin
        high_thresh = angle+margin
        filter_theta = theta[theta>low_thresh]
        filter_theta = filter_theta[filter_theta < high_thresh]

        return image,np.median(filter_theta)

    def re_orient_east(self):
        lang = 'hi'

        args = {
            "image": self.image_path,
            "east": config.EAST_MODEL,
            "min_confidence": config.MIN_CONFIDENCE,
            "margin_tollerance":config.MARGIN_TOLLERANCE,
            "width": config.EAST_WIDTH,
            "height": config.EAST_HEIGHT
        }

        image,angle = Orientation.east_hough_line(self,args['image'],args)

        if abs(angle) > config.ANGLE_TOLLERANCE:
            image = Orientation.rotate_bound(self,image, angle)
            cv2.imwrite(self.image_path, image)

        print("Angle detectd is  {} ".format(angle))

        return image,angle
