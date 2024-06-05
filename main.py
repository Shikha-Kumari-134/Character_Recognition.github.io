import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import os
import pickle

path="Data/"
files=os.listdir(path)
print(files)
classes={"01_ka":0,"02_kha":1,"03_ga":2,"04_gha":3,"05_kna":4,"06_cha":5,"07_chha":6,"08_ja":7,"09_jha":8,"10_yna":9}
import cv2
x= []
y= []
for cl in classes:
    pth = path+cl
    for img_name in os.listdir(pth):
        img = cv2.imread(pth +"/"+img_name,0)
        x.append(img)
        y.append(classes[cl])
x[0].shape
x = np.array(x)
y = np.array(y)
x.shape
x_new = x.reshape(len(x),-1)
x_train , x_test, y_train, y_test  = train_test_split(x_new,y,test_size=.20,random_state=1)
print(x_train.max())
print(x_test.max())
xtrain = x_train/255
xtest = x_test/255
print(xtrain.max())
print(xtest.max())
from sklearn.decomposition import PCA
pca= PCA(.98)
x_train= pca.fit_transform(xtrain)
x_test= pca.transform(xtest)
y_test[:10]
from sklearn.manifold import TSNE
t_sne = TSNE(n_components=3, learning_rate='auto',init='random')
x_embedded=t_sne.fit_transform(x_new)
x_embedded.shape

#Model1
model1=MLPClassifier(hidden_layer_sizes=(30, 15), max_iter=30,activation='relu',alpha=0.01)
model1.fit(x_train,y_train)

pickle.dump(model1, open("model.pkl","wb"))
pickle.dump(pca, open("pca.pkl","wb"))