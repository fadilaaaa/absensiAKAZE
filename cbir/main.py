import cv2
import numpy as np
import os
import math
import pandas as pd
import pickle
from PIL import Image
from sklearn.metrics import accuracy_score
from face_detect import detect, crop_image

cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
haar_model = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')
faceCascade = cv2.CascadeClassifier(haar_model)

def glcm(image, distance=1, colorBit=256, angle=0):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    w, h = gray.shape[:2]
    # variabel co-occurance
    glcm_mat = np.zeros((colorBit, colorBit))

    # hitung glcm matrix
    for i in range(w):
        for j in range(h-1):
            x, y = gray[i][j], gray[i][j+distance]
            glcm_mat[x][y] += 1
    symm_mat = glcm_mat + glcm_mat.transpose()  # jumlahkan glcm dan transpose
    glcm_norm = symm_mat / symm_mat.sum()  # normalisasi
    return glcm_norm


def glcm_features(glcm_mat):
    asm = 0
    contrast = 0
    homogeneity = 0
    correlation = 0
    mi, mj, pi, pj = 0, 0, 0, 0

    w, h = glcm_mat.shape[:2]
    for i in range(w):
        for j in range(h):
            val = glcm_mat[i][j]
            mi += (i * val)
            mj += (j * val)

    for i in range(w):
        for j in range(h):
            val = glcm_mat[i][j]
            pi += ((i - mi)**2 * val)
            pj += ((j - mj)**2 * val)

    # print("NILAI: ", mi, mj, pi, pj)
    for i in range(w):
        for j in range(h):
            val = glcm_mat[i][j]
            asm += val**2
            contrast += (i-j)**2 * val
            homogeneity += (val/(1+abs(i-j)**2))

            correlation += ((i - mi) * (j - mj) * val)/(pi * pj)
    return asm, contrast, homogeneity, correlation

def kaze_feature(image, vector_size=32):
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  try:
    # Using KAZE, cause SIFT, ORB and other was moved to additional module
    # which is adding addtional pain during install
    alg = cv2.KAZE_create()
    # Finding image keypoints
    kps = alg.detect(image)
    # Getting first 32 of them. 
    # Number of keypoints is varies depend on image size and color pallet
    # Sorting them based on keypoint response value(bigger is better)
    kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
    # computing descriptors vector
    kps, dsc = alg.compute(image, kps)
    # Flatten all of them in one big vector - our feature vector
    dsc = dsc.flatten()
    # Making descriptor of same size
    # Descriptor vector size is 64
    needed_size = (vector_size * 64)
    if dsc.size < needed_size:
        # if we have less the 32 descriptors then just adding zeros at the
        # end of our feature vector
        dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
  except cv2.error as e:
    print('Error: ', e)
    return None
  return dsc

# update_dataset update csv of features extraction


def update_dataset(dataset_path='dataset_train', test=False):  
    classnames = []
    kaze_feats = []
    csv_tmp = ["filename,class,asm,contrast,homogeneity,correlation"]
    for image_file in os.listdir(dataset_path):
        if image_file.endswith(".jpg") or image_file.endswith(".jpeg") or image_file.endswith(".png"):
            image_path = os.path.join(dataset_path, image_file)
            print(image_file)
            
            image = cv2.imread(image_path)
            if test:
              coords, img = detect(image, faceCascade)
              img = Image.fromarray(img)
              imgSnap = crop_image(img, coords, 20)
              imgSnap = imgSnap.resize((256,256))
              image = np.asarray(imgSnap)
            
            glcm_norm = glcm(image)
            asm, contrast, homogeneity, correlation = glcm_features(glcm_norm)
            
            # kaze features
            kaze_feats.append(kaze_feature(image))

            classname = image_file.split("-")[0]
            if classname not in classnames:
              classnames.append(classname)

            features = "{},{},{},{},{},{}".format(
                image_file,
                classname,
                asm, contrast, homogeneity, correlation,
            )
            csv_tmp.append(features)
            
    if test:
      # saving all our feature vectors in pickled file
      with open("kazefeat_test.pck", 'wb') as fp:
          pickle.dump(kaze_feats, fp)

      csv_txt = "\n".join(csv_tmp)
      with open('features_test.csv', 'w') as file:
          file.write(csv_txt)
      classnames = "\n".join(classnames)
      return
    
    # saving all our feature vectors in pickled file
    with open("kazefeat.pck", 'wb') as fp:
        pickle.dump(kaze_feats, fp)

    csv_txt = "\n".join(csv_tmp)
    with open('features.csv', 'w') as file:
        file.write(csv_txt)
    classnames = "\n".join(classnames)
    with open('dataname.txt', 'w') as file:
        file.write(classnames)

# test_program do testing with accuracy for face classifier
# face_rec_option is option for face recognition algorithm
# 1: glcm only
# 2: kaze only
def test_program(dataset_path="dataset_test", face_rec_option=2):
  df = pd.read_csv("features_test.csv", header=0)
  
  # get sample data and save actual class
  df_test = df.sample(frac=1).sort_index()
  actual = np.array(df_test["class"])
  pred = []
  
  # for each data sample do face recognition
  name_files = np.array(df_test["filename"])
  for image_file in os.listdir(dataset_path):
    if image_file in name_files:
      image_path = os.path.join(dataset_path, image_file)
      image = cv2.imread(image_path)
      res = face_recognition(image, opt=face_rec_option)
      print(image_file, " ==> ", res)
      pred.append(res[0])  # get class
  
  # get accuracy
  print("Accuracy: {}".format(accuracy_score(actual, np.array(pred))))

# face_recognition predict and classify face by compare it with dataset


def face_recognition(query_image, opt=2, test=False):
    if test:
      coords, img = detect(query_image, faceCascade)
      img = Image.fromarray(img)
      imgSnap = crop_image(img, coords, 20)
      imgSnap = imgSnap.resize((256,256))
      query_image = np.asarray(imgSnap)
    glcm_norm = glcm(query_image)
    query_glcm_feat = glcm_features(glcm_norm)
    query_kaze_feat = kaze_feature(query_image)

    # Initialize a dictionary to store similarity scores for each image
    similarity_scores = {}
    csvFilename = "features.csv"
    kazeFilename = "kazefeat.pck"
    df = pd.read_csv(csvFilename, header=0)
    
    # kaze feature extractions
    with open(kazeFilename, "rb") as fp:
      kaze_feats = pickle.load(fp)
      
    for i, row in df.iterrows():
        glcm_feat = row[2:]
        if opt == 1: # glcm only
          euc_score = euclidean_distance(query_glcm_feat, glcm_feat)
        elif opt == 2: # kaze feature only
          euc_score = euclidean_distance(query_kaze_feat, kaze_feats[i])
        # elif opt==3: # glcm + kaze feature
        #   euc_score1 = euclidean_distance(query_glcm_feat, glcm_feat)
        #   euc_score2 = euclidean_distance(query_kaze_feat, kaze_feats[i])
        #   euc_score = (euc_score1 + euc_score2) /2
        else:
          print("Non-valid option")
          return None
        
        if row["class"] in similarity_scores.keys():
            similarity_scores[row["class"]].append(euc_score)
        else:
            similarity_scores[row["class"]] = [euc_score]

    for key, lst in similarity_scores.items():
        similarity_scores[key] = sum(lst) / len(lst)

    # Sort the images based on similarity scores
    sorted_images = sorted(similarity_scores.items(), key=lambda x: x[1])

    # Display the most similar images
    # print("Predict: ", sorted_images[0:2])
    return sorted_images[0]


def euclidean_distance(vecA, vecB):
    sum = 0
    for j in range(len(vecA)):
        sum += math.pow(vecA[j] - vecB[j], 2)
    return math.sqrt(sum)


def main():
    # print("updating dataset...")
    # update_dataset('dataset_train')
    # print("done")

    # query_image_path = "dataset_train/Clara-5.png"
    # query_image = cv2.imread(query_image_path)
    # print("PREDICTED: ", face_recognition(query_image))
    
    print("test started")
    update_dataset('dataset_test', test=True)
    test_program(face_rec_option=2)
    print("test done")

if __name__ == "__main__":
  main()