import cv2
from PIL import Image
import numpy as np
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
haar = os.path.join(f'{dir_path}/../haarcascade_frontalface_default.xml')
FACE_CASCADE = cv2.CascadeClassifier(haar)


def crop_image(im, coords, optSize=0):
    left = coords[0] - optSize
    top = coords[1] - optSize
    right = left + coords[2] + optSize*2
    bottom = top + coords[3] + optSize*2
    return im.crop((left, top, right, bottom))


def draw_boundary(img, scale_factor, min_neighbors, color, text):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = FACE_CASCADE.detectMultiScale(
        gray_img, scale_factor, min_neighbors)
    coords = []
    for (x, y, w, h) in features:
        # cv2.putText(img, text, (x, y-4), cv2.FONT_HERSHEY_SIMPLEX,
        #             0.8, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h]
    return coords, img


def detect(img):
    color = {"red": (255, 0, 0), "blue": (0, 0, 255), "green": (0, 255, 0)}
    coords, img = draw_boundary(
        img, 1.1, 20, color["green"], "Face")
    return coords, img


def get_face_cropped(base_image):
    """
    base_image -> cv2 image
    """

    cv2image = cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB)
    img = cv2.flip(cv2image, 1)
    coords, img = detect(img)
    img = Image.fromarray(img)
    imgSnap = crop_image(img, coords, 25)
    imgSnap = imgSnap.resize((256, 256))
    imgSnap = cv2.cvtColor(np.array(imgSnap), cv2.COLOR_RGB2BGR)
    return imgSnap


def is_completely_black(image, threshold=10):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_pixel_value = int(gray_image.mean())
    return mean_pixel_value < threshold
