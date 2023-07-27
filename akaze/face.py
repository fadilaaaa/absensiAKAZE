import cv2
import os
from PIL import Image
import numpy as np
cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
haar = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')
# crop_image crop image in specific coordinates
def crop_image(im, coords, optSize=0):
  left = coords[0] + optSize
  top = coords[1] + optSize
  right = left + coords[2] - optSize*2
  bottom = top + coords[3] - optSize*2
  return im.crop((left, top, right, bottom))

# draw_boundary create text to show if face is detected
def draw_boundary(img, classifier, scale_factor, min_neighbors, color, text):
  gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  features = classifier.detectMultiScale(gray_img, scale_factor, min_neighbors)
  coords = []
  for (x,y,w,h) in features:
    # cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
    cv2.putText(img, text, (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
    coords = [x,y,w,h]
  return coords, img

# detect do face detection from image
def detect(img, faceCascade):
  color = {"red":(255,0,0), "blue":(0,0,255), "green":(0,255,0)}
  coords, img = draw_boundary(img, faceCascade, 1.1, 20, color["green"], "Face")
  return coords, img

def get_face_cropped(base_image):
  """
  base_image -> cv2 image
  """
  # base_image = cv2.imread('t1.jpeg')
  face_cascade = cv2.CascadeClassifier(haar)
  cv2image = cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB)
  img = cv2.flip(cv2image, 1)
  coords, img = detect(img, face_cascade)
  img = Image.fromarray(img)
  imgSnap = crop_image(img, coords, 20)
  imgSnap = imgSnap.resize((256,256))
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

    # print("keypoints: {}, descriptors: {}".format(len(kps1), descs1.shape))
    # print("keypoints: {}, descriptors: {}".format(len(kps2), descs2.shape))

    # Match the features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(descs1,descs2, k=2)    # typo fixed

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.9*n.distance:
            good.append([m])
    return len(good)



cam_port = 0
cam = cv2.VideoCapture(cam_port)

result, image = cam.read()

if result:
    cv2.imshow("tesss", get_face_cropped(image))
    cv2.imwrite("tesss.png", image)
    print(kaze_matchs(get_face_cropped(image),get_face_cropped(cv2.imread('as.png'))))
    cv2.waitKey(0)
    cv2.destroyWindow("tesss")
else:
    print("No image detected. Please! try again")
