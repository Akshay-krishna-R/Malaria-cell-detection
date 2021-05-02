

from PIL import Image # We use the PIL Library to resize images
import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
data = []
labels = []

Parasitized = os.listdir("./cell_images/Parasitized/")
for parasite in Parasitized:
    try:
        image=cv2.imread("./cell_images/Parasitized/"+parasite)
        image_from_array = Image.fromarray(image, 'RGB')
        size_image = image_from_array.resize((50, 50))
        data.append(np.array(size_image))
        labels.append(0)
    except AttributeError:
        print("")

Uninfected = os.listdir("./cell_images/Uninfected/")
for uninfect in Uninfected:
    try:
        image=cv2.imread("./cell_images/Uninfected/"+uninfect)
        image_from_array = Image.fromarray(image, 'RGB')
        size_image = image_from_array.resize((50, 50))
        data.append(np.array(size_image))
        labels.append(1)
    except AttributeError:
        print("")

len(data)

type(data)

data = np.array(data)

type(data)

data.shape

labels =np.array(labels)

type(labels)

labels.shape

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(data,labels,test_size=0.3,random_state = 0)

X_train.shape

X_test.shape

y_train.shape

y_test.shape

X_train = X_train/255.0

model = models.Sequential()
model.add(layers.Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', input_shape=(50,50,3)))
model.add(layers.MaxPooling2D(pool_size=2))
model.add(layers.Conv2D(filters=32,kernel_size=2,padding='same',activation='relu'))
model.add(layers.MaxPooling2D(pool_size=2))
model.add(layers.Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
model.add(layers.MaxPooling2D(pool_size=2))
model.add(layers.Dropout(0.2))
model.add(layers.Flatten())
model.add(layers.Dense(32,kernel_initializer="he_uniform",activation="relu"))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(64,kernel_initializer="he_uniform",activation="relu"))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(128,kernel_initializer="he_uniform",activation="relu"))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(256,kernel_initializer="he_uniform",activation="relu"))
model.add(layers.Dropout(0.2))




model.add(layers.Dense(1,kernel_initializer='glorot_uniform',activation="sigmoid"))#2 represent output layer neurons 
model.summary()

model.compile(optimizer="adam",
              loss="binary_crossentropy", 
             metrics=["accuracy"])

history = model.fit(X_train,y_train, epochs=20, validation_data=(X_test,y_test))

model.save("Malaria_cell_classification.h5")

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('/content/C38P3thinF_original_IMG_20150621_112116_cell_205.png',target_size = (50,50))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
result = model.predict(test_image)

result

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('/content/C3thin_original_IMG_20150608_163047_cell_145.png',target_size = (50,50))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
result = model.predict(test_image)
result
