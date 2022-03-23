import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt 
import os

base_dir = "E:/Kuliah/Projek/Visi Komputer/visicomputer/eye-dataset"
label = ['close_look','forward_look','left_look','right_look']
IMG_SIZE = 50
train_data = []

def create_training_data():
    for l in label :
        path = os.path.join(base_dir, l)
        class_num = label.index(l)
        for img in os.listdir(path):
            try:
                img_array = cv.imread(os.path.join(path, img) , cv.IMREAD_GRAYSCALE)
                new_array = cv.resize(img_array,(IMG_SIZE,IMG_SIZE))
                train_data.append([new_array,class_num])
            except Exception as e:
                pass
create_training_data()
X = []
y = []
for fitur, label in train_data :
    X.append(fitur)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE,IMG_SIZE,1)

print(X[1])