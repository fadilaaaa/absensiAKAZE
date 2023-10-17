import cv2
import pandas as pd
import pickle
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
pathDB = f'{dir_path}/../data/faceDB.csv'


detector = cv2.AKAZE_create()


def features_matchs(descs1, descs2):
    """
    descs1 , descs2 -> features
    """

    # Match the features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(descs1, descs2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.9*n.distance:
            good.append([m])
    return len(good)


def get_featureAKAZE(image):
    """
    image -> cv2 image
    """
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (_, descs) = detector.detectAndCompute(image, None)
    return descs


def who_is_this(desc):
    db_of_descs = pd.read_csv(pathDB)
    list_of_good_matches = []
    for idx, row in db_of_descs.iterrows():
        if pd.notna(row['desc']):
            with open(f"{dir_path}/../data/{row['desc']}", 'rb') as f:
                desc2 = pickle.load(f)

            matches = features_matchs(desc, desc2)
            list_of_good_matches.append(
                {"matches": matches, "name": row['name']})
    matches_max = max(list_of_good_matches, key=lambda x: x['matches'])
    if matches_max['matches'] > 10:
        return matches_max['name']
    else:
        return "Tidak dikenali"
