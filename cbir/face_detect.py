import cv2

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