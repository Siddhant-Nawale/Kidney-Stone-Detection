from lightgbm.basic import _label_from_pandas
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
import lightgbm as lgb
import pandas as pd
from segmentation import *
import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)


def feature_extractor(dataset):
    image_dataset = pd.DataFrame()
    for image in range(dataset.shape[0]):
        df = pd.DataFrame()

        img = dataset[image, :]
        n = 0
        for j in [0, np.pi / 4, np.pi / 2]:
            for i in [1, 3, 5, 7, 9, 11]:
                GLCM = graycomatrix(img, [i], [j])
                GLCM_Energy = graycoprops(GLCM, 'energy')[0]
                df['Energy-' + str(n)] = GLCM_Energy
                GLCM_corr = graycoprops(GLCM, 'correlation')[0]
                df['Correlation-' + str(n)] = GLCM_corr
                GLCM_diss = graycoprops(GLCM, 'dissimilarity')[0]
                df['Dissimilarity-' + str(n)] = GLCM_diss
                GLCM_hom = graycoprops(GLCM, 'homogeneity')[0]
                df['Homogeneity-' + str(n)] = GLCM_hom
                GLCM_contr = graycoprops(GLCM, 'contrast')[0]
                df['Contrast-' + str(n)] = GLCM_contr
                entropy = shannon_entropy(img)
                df['Entropy-' + str(n)] = entropy
                n += 1
                img = img.copy()

        image_dataset = image_dataset.append(df)

    return image_dataset

SIZE = 128
addr = './image1.png'
img = cv2.imread(addr,0)
img = cv2.resize(img, (SIZE, SIZE))
image = np.array(img)

lgb_model = lgb.Booster(model_file='lgb_classifier.dat')

input_img = np.expand_dims(image, axis=0)

input_img_features=feature_extractor(input_img)

input_img_features = np.expand_dims(input_img_features, axis=0)

input_img_for_RF = np.reshape(input_img_features, (input_img.shape[0], -1))


from sklearn import preprocessing
import glob
import os
label=[]

le = preprocessing.LabelEncoder()
for directory_path in glob.glob("DataSet/Train/*"):
    fruit_label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        label.append(fruit_label)

label = np.array(label)

le.fit(label)

img_prediction = lgb_model.predict(input_img_for_RF)
img_prediction=np.argmax(img_prediction, axis=1)


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
    img_prediction = le.inverse_transform([img_prediction])


if img_prediction[0] == 'With':
    print("Kidney Stone Detected")
    img = cv2.imread(addr)
    plt.imshow(img,'gray')
    plt.title('Original')
    plt.xticks([]),plt.yticks([])
    plt.show()
    img=img-img*2
    img3 = GaborFilter(img)
    img3 = Watershed(img3, img3)
    plt.imshow(img3,'gray')
    plt.title('Marked')
    plt.xticks([]),plt.yticks([])
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows
else:
    print("No Kidney Stone Was Detected")
