from lib.matcher import features_matchs
from lib.matcher import get_featureAKAZE
from lib.face import get_face_cropped, is_completely_black
import cv2
import pandas as pd
import pickle
import numpy as np
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

cam_port = 0
cam = cv2.VideoCapture(cam_port)

NAME = 'arsa'
pathDB = f'{dir_path}/data/faceDB.csv'
db_of_descs = pd.read_csv(pathDB)
while True:
    result, image = cam.read()
    if result:
        cv2.imshow("tesss", image)
        if not is_completely_black(image):
            try:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                cropped_img = get_face_cropped(image)
                desc = get_featureAKAZE(cropped_img)
                list_of_good_matches = []
                for idx, row in db_of_descs.iterrows():
                    with open(f"{dir_path}/data/{row['desc']}", 'rb') as f:
                        desc2 = pickle.load(f)

                    matches = features_matchs(desc, desc2)
                    list_of_good_matches.append(
                        {"matches": matches, "name": row['name']})
                matches_max = max(list_of_good_matches,
                                  key=lambda d: d['matches'])
                if matches_max['matches'] > 20:
                    print(matches_max['name'])
                    break
                else:
                    print("Tidak dikenali")
            except Exception as e:
                print(e)
                # break
    else:
        print("No image detected. Please! try again")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
