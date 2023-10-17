from lib.face import get_face_cropped, is_completely_black
from lib.matcher import get_featureAKAZE
import cv2
import pandas as pd
import pickle
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

cam_port = 0
cam = cv2.VideoCapture(cam_port)

NAME = 'markus'
pathDB = f'{dir_path}/data/faceDB.csv'
df = pd.read_csv(pathDB)
# df.set_index('name', inplace=True)
while True:
    result, image = cam.read()
    if result:
        cv2.imshow("tesss", image)
        if not is_completely_black(image):
            try:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                cropped_img = get_face_cropped(image)
                cv2.imwrite(f'{dir_path}/data/image/{NAME}.png', cropped_img)
                desc = get_featureAKAZE(cropped_img)
                with open(f'{dir_path}/data/{NAME}.pkl', 'wb') as f:  # open a text file
                    pickle.dump(desc, f)
                # df.loc[len(df.index)] = [NAME, f'{NAME}.pkl']
                if NAME not in df['name'].values:
                    df.loc[len(df.index)] = [NAME, f'{NAME}.pkl']
                else:
                    df.loc[df['name'] == NAME, 'desc'] = f'{NAME}.pkl'
                df.to_csv(pathDB, index=False)
                print("wajah tersimpan")
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
