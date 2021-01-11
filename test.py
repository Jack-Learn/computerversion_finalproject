import os,sys 
import cv2 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model, load_model
from keras.layers import *
from keras.callbacks import *
from keras.utils import np_utils
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

def read_images_labels(path,i):
    for file in os.listdir(path):
        abs_path = os.path.abspath(os.path.join(path, file))  
        if os.path.isdir(abs_path):
            i+=1
            temp = os.path.split(abs_path)[-1]
            name.append(temp)
            read_images_labels(abs_path,i)
            amount = int(len(os.listdir(path)))
            sys.stdout.write('\r'+'>'*(i)+' '*(amount-i)+'[%s%%]'%(i*100/amount)+temp)
        else:
            if file.endswith('.jpg'):
                image=cv2.resize(cv2.imread(abs_path),(128,128))
                images.append(image)
                labels.append(i-1)
    return images, labels, name

def read_images(path):
    images=[]
    for i in range(25):
        image=cv2.resize(cv2.imread(path+str(i+1)+'.jpg'),(128,128))
        images.append(image)
    images=np.array(images,dtype=np.float32)/255
    return images
    
def transform(listdir,label,lenSIZE):
    label_str=[]
    for i in range (lenSIZE):
        temp=listdir[label[i]] 
        label_str.append(temp)
    return label_str

images = read_images('D:/GitHub/ACV/dataset/mingxang/test/')
model = load_model('./weight/0214_05-0.83.hdf5')

predict = model.predict_classes(images, verbose=1)
print(predict)
label_str=transform(np.loadtxt('name.txt',dtype='str'),predict,images.shape[0])

df = pd.DataFrame({"character":label_str})
df.index = np.arange(1, len(df) + 1)
df.index.names = ['id']
df.to_csv('./result/test.csv')