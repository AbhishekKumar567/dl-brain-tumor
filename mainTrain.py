import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
#from tensorflow.python.keras.utils import normalize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Activation,MaxPooling2D,Dropout,Flatten,Dense
from keras.utils import to_categorical

image_directory = "datasets/"
dataset=[]
label=[]
no_tumor_images = os.listdir(image_directory + 'no/')
#print(no_tumor_images)
yes_tumor_images = os.listdir(image_directory + 'yes/')
#print(yes_tumor_images)

for i,image_name in enumerate(no_tumor_images):
    if(image_name.split('.')[1]=='jpg'):
        image = cv2.imread(image_directory+'no/'+image_name)
        image = Image.fromarray(image,'RGB')
        image=image.resize((64,64))
        dataset.append(np.array(image))
        label.append(0)


for i,image_name in enumerate(yes_tumor_images):
    if(image_name.split('.')[1]=='jpg'):
        image = cv2.imread(image_directory+'yes/'+image_name)
        image = Image.fromarray(image,'RGB')
        image=image.resize((64,64))
        dataset.append(np.array(image))
        label.append(1)

# print(len(label))
#print(label)
dataset=np.array(dataset)
label=np.array(label)

#print(dataset)
#print(label)

x_train,x_test,y_train,y_test = train_test_split(dataset,label,test_size=0.2,random_state=0)

# print(x_train.shape)
# print(y_train.shape)

# print(x_test.shape)
# print(y_test.shape)

x_train = x_train/255.0 #tf.keras.utils.normalize(x_train,axis=1)
x_test = x_test/255.0

y_train = to_categorical(y_train,num_classes=2)
y_test = to_categorical(y_test,num_classes=2)



#Model Building
model = Sequential()
model.add(Conv2D(32,(3,3),input_shape=(64,64,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3),kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3),kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense((64)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=16,verbose=1,epochs=10,validation_data=(x_test,y_test),shuffle=False)

model.save('BrainTumor10EpochsCategorical.h5')
