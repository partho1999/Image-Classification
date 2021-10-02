import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow import keras
from keras.models import Model, load_model
import glob
import cv2
import os


# IMG_SIZE = 224
# size = (IMG_SIZE, IMG_SIZE)
img_width, img_height = 224, 224

inputs = layers.Input(shape=(img_width, img_height, 3))


# Using model without transfer learning

outputs = EfficientNetB0(include_top=True, weights=None)(inputs)


#Model Fiting
model = tf.keras.Model(inputs, outputs)


model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"] )

# checkpoint_filepath = 'E:\codes\EfficentNetB0\Image-Classification-Using-EfficientNets\my_model'
#
# model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_filepath,
#     save_weights_only=True,
#     monitor='accuracy',
#     mode='max',
#     save_best_only=True)

model.summary()

#Resizing images is optional, CNNs are ok with large images
SIZE_X = 224 #Resize images (height  = X, width = Y)
SIZE_Y = 224

#Capture training image info as a list
train_images = []

for directory_path in glob.glob("E:\codes\EfficentNetB0\Image-Classification-Using-EfficientNets\dataset\Brain Tumor"):
    for img_path in glob.glob(os.path.join(directory_path, "*.tif")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (SIZE_Y, SIZE_X))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        train_images.append(img)
        #train_labels.append(label)
#Convert list to array for machine learning processing
train_images = np.array(train_images)
X_train = train_images

features=model.predict(X_train)
print(features)