import cv2
import os
from PIL import Image
import numpy as np
cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
haar = os.path.join('haarcascade_frontalface_default.xml')
FACE_CASCADE = cv2.CascadeClassifier(haar)


def crop_image(im, coords, optSize=0):
    left = coords[0] + optSize
    top = coords[1] + optSize
    right = left + coords[2] - optSize*2
    bottom = top + coords[3] - optSize*2
    return im.crop((left, top, right, bottom))


def draw_boundary(img, scale_factor, min_neighbors):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = FACE_CASCADE.detectMultiScale(
        gray_img, scale_factor, min_neighbors)
    coords = []
    for (x, y, w, h) in features:
        coords = [x, y, w, h]
    return coords, img


def detect(img):
    coords, img = draw_boundary(
        img, 1.1, 20)
    return coords, img


def get_face_cropped(base_image):
    """
    base_image -> cv2 image
    """

    cv2image = cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB)
    img = cv2.flip(cv2image, 1)
    coords, img = detect(img)
    img = Image.fromarray(img)
    imgSnap = crop_image(img, coords, 20)
    imgSnap = imgSnap.resize((256, 256))
    imgSnap = cv2.cvtColor(np.array(imgSnap), cv2.COLOR_RGB2BGR)
    return imgSnap


def kaze_matchs(im1, im2):
    """
    im1 , im2 -> cv2 image object
    """
    gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # initialize the AKAZE descriptor, then detect keypoints and extract
    # local invariant descriptors from the image
    detector = cv2.AKAZE_create()
    (kps1, descs1) = detector.detectAndCompute(gray1, None)
    (kps2, descs2) = detector.detectAndCompute(gray2, None)

    # Match the features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(descs1, descs2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.9*n.distance:
            good.append([m])
    return len(good)


def is_completely_black(image, threshold=10):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_pixel_value = int(gray_image.mean())
    return mean_pixel_value < threshold


cam_port = 0
cam = cv2.VideoCapture(cam_port)

while True:
    result, image = cam.read()
    if result:
        cv2.imshow("tesss", image)
        if not is_completely_black(image):
            try:
                # cv2.imshow("tesss", get_face_cropped(image))
                cv2.imwrite("tesss.png", get_face_cropped(image))
                matches = kaze_matchs(get_face_cropped(
                    image), cv2.imread('exmp.png'))
                if matches > 15:
                    print(matches)
                    print("Face Matched")
                    break
            except Exception as e:
                print(e)
                # break
    else:
        print("No image detected. Please! try again")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
