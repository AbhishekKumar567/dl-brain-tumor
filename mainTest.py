import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model=load_model('BrainTumor10EpochsCategorical.h5')

image = cv2.imread('C:\\Users\\Abhishek Kumar\\Downloads\\archive\\pred\\pred26.jpg')

#image = cv2.resize(image, (64,64))  # Resize the image to the expected shape

img = Image.fromarray(image)

img = img.resize((64,64))

img = np.array(img)

input_img = np.expand_dims(img,axis=0)

res = model.predict(input_img)
inp = np.argmax(res,axis=1)

print(inp)
                   
