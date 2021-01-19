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

epochs=100
batch_size=8

images= []
labels= []
name= []



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
                image=cv2.resize(cv2.imread(abs_path),(256,256))
                images.append(image)
                labels.append(i-1)
    return images, labels, name




def read_main(path):
    images, labels, name = read_images_labels(path,i=0)
    images = np.array(images,dtype=np.float32)/255
    labels = np_utils.to_categorical(labels, num_classes=2)
    np.savetxt('name.txt', name, delimiter = ' ',fmt="%s")
    return images, labels

images, labels=read_main(r'D:\Doucuments\Homework\ACV\Final_Project\Dataset\train-test\train')
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2)                        


model = Sequential()
model.add(Conv2D(8, kernel_size=3, padding='valid',activation='relu',input_shape=X_train.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(64, kernel_size=3, padding='valid',activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=3, padding='valid',activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=3, padding='valid',activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=3, padding='same',activation='relu'))

model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2,activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

# datagen = ImageDataGenerator(zoom_range=0.1,width_shift_range=0.05,height_shift_range=0.05,horizontal_flip=True)
# datagen.fit(X_train)

# checkpoint
filepath="./logs/{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
mode='max')
callbacks_list = [checkpoint]

file_name=str(epochs)+'_'+str(batch_size)
history = model.fit(X_train, y_train, batch_size=batch_size,  epochs=epochs, validation_data = (X_test, y_test ), verbose = 1, callbacks=callbacks_list,)

model.save('./weight/my_modle.h5')
score = model.evaluate(X_test, y_test, verbose=1)
print(score)

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model acc')
plt.ylabel('acc')
plt.xlabel('epochs')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()



def read_images(path):
    images=[]
    for i in range(25):
        image=cv2.resize(cv2.imread(path+str(i+1)+'.jpg'),(256,256))
        images.append(image)
    images=np.array(images,dtype=np.float32)/255
    return images

def transform(listdir,label,lenSIZE):
    label_str=[]
    for i in range (lenSIZE):
        temp=listdir[label[i]] 
        label_str.append(temp)
    return label_str

images = read_images(r'D:\Doucuments\Homework\ACV\Final_Project\Dataset\train-test\test\')
model = load_model('./weight/my_modle.h5')

predict = model.predict_classes(images, verbose=1)
print(predict)
label_str=transform(np.loadtxt('name.txt',dtype='str'),predict,images.shape[0])

df = pd.DataFrame({"character":label_str})
df.index = np.arange(1, len(df) + 1)
df.index.names = ['id']
df.to_csv('./result/test.csv')