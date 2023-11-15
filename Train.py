import numpy as np 
import glob
import cv2
import os
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy

print(os.listdir("DataSet\Train"))

SIZE = 128

train_images = []
train_labels = [] 
for directory_path in glob.glob("DataSet\Train\*"):
    label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        img = cv2.imread(img_path, 0)
        kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
        img1 = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
        img = cv2.resize(img, (SIZE, SIZE))
        train_images.append(img)
        train_labels.append(label)
        
train_images = np.array(train_images)
train_labels = np.array(train_labels)


test_images = []
test_labels = []
for directory_path in glob.glob("DataSet\Test\*"):
    fruit_label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        img = cv2.imread(img_path, 0)
        kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
        img1 = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
        img = cv2.resize(img, (SIZE, SIZE)) 
        test_images.append(img)
        test_labels.append(fruit_label)
        
test_images = np.array(test_images)
test_labels = np.array(test_labels)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(test_labels)
test_labels_encoded = le.transform(test_labels)
le.fit(train_labels)
train_labels_encoded = le.transform(train_labels)

x_train, y_train, x_test, y_test = train_images, train_labels_encoded, test_images, test_labels_encoded


def feature_extractor(dataset):
    image_dataset = pd.DataFrame()
    for image in range(dataset.shape[0]):
        df = pd.DataFrame()
        
        img = dataset[image, :,:]
        n=0
        for j in [0,np.pi/4,np.pi/2]:
            for i in [1,3,5,7,9,11]:
                GLCM = graycomatrix(img, [i], [j])       
                GLCM_Energy = graycoprops(GLCM, 'energy')[0]
                df['Energy-'+str(n)] = GLCM_Energy
                GLCM_corr = graycoprops(GLCM, 'correlation')[0]
                df['Correlation-'+str(n)] = GLCM_corr       
                GLCM_diss = graycoprops(GLCM, 'dissimilarity')[0]
                df['Dissimilarity-'+str(n)] = GLCM_diss       
                GLCM_hom = graycoprops(GLCM, 'homogeneity')[0]
                df['Homogeneity-'+str(n)] = GLCM_hom       
                GLCM_contr = graycoprops(GLCM, 'contrast')[0]
                df['Contrast-'+str(n)] = GLCM_contr
                entropy = shannon_entropy(img)
                df['Entropy-'+str(n)] = entropy
                n+=1
                img = img.copy()
        
        image_dataset = image_dataset.append(df)
        
    return image_dataset



####################################################################
image_features = feature_extractor(x_train)
X_for_ML =image_features

import lightgbm as lgb
d_train = lgb.Dataset(X_for_ML, label=y_train)

lgbm_params = {'learning_rate':0.05, 'boosting_type':'dart',    
              'objective':'multiclass',
              'metric': 'multi_logloss',
              'num_leaves':100,
              'max_depth':20,
              'num_class':2,
              'force_col_wise' : True
              }

lgb_model = lgb.train(lgbm_params, d_train, 100)    


lgb_model.save_model('lgb_classifier.dat', num_iteration=lgb_model.best_iteration) 
 


test_features = feature_extractor(x_test)
test_features = np.expand_dims(test_features, axis=0)
test_for_RF = np.reshape(test_features, (x_test.shape[0], -1))

test_prediction = lgb_model.predict(test_for_RF)
test_prediction = np.argmax(test_prediction, axis=1)
test_prediction = le.inverse_transform(test_prediction)

from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(test_labels, test_prediction))


import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_labels, test_prediction)

fig, ax = plt.subplots(figsize=(6,6))
sns.set(font_scale=1.6)
sns.heatmap(cm, annot=True, linewidths=.5, ax=ax)
plt.show()


