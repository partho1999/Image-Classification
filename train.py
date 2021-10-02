import pathlib

import numpy as np
import tensorflow as tf

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow import keras
from keras.models import Model, load_model
import h5py



dataset_path = os.listdir('dataset')

# print (dataset_path)  #what kinds of classes are in this dataset
#
# print("Types of classes labels found: ", len(dataset_path))

class_labels = []

for item in dataset_path:
 # Get all the file names
 all_classes = os.listdir('dataset' + '/' +item)
 #print(all_classes)

 # Add them to the list
 for room in all_classes:
    class_labels.append((item, str('dataset_path' + '/' +item) + '/' + room))
    #print(class_labels[:5])

df = pd.DataFrame(data=class_labels, columns=['Labels', 'image'])
#print(df.head())
#print(df.tail())

# Let's check how many samples for each category are present
#print("Total number of images in the dataset: ", len(df))

label_count = df['Labels'].value_counts()
#print(label_count)


path = 'dataset/'
#dataset_path = os.listdir('dataset')

im_size = 224

images = []
labels = []

for i in dataset_path:
    data_path = path + str(i)
    filenames = [i for i in os.listdir(data_path)]

    for f in filenames:
        img = cv2.imread(data_path + '/' + f)
        img = cv2.resize(img, (im_size, im_size))
        images.append(img)
        labels.append(i)

#This model takes input images of shape (224, 224, 3), and the input data should range [0, 255].

images = np.array(images)

images = images.astype('float32') / 255.0
images.shape

#preprocessing and Encoding
y=df['Labels'].values
#print(y)

y_labelencoder = LabelEncoder ()
y = y_labelencoder.fit_transform (y)
#print (y)

y=y.reshape(-1,1)

#ColumnTransformer
ct = ColumnTransformer([('my_ohe', OneHotEncoder(), [0])], remainder='passthrough')
Y = ct.fit_transform(y) #.toarray()
# print(Y[:5])
# print(Y[35:])

#Shuffle
#train_test_split


images, Y = shuffle(images, Y, random_state=1)


train_x, test_x, train_y, test_y = train_test_split(images, Y, test_size=0.05, random_state=415)

#inpect the shape of the training and testing.
# print(train_x.shape)
# print(train_y.shape)
# print(test_x.shape)
# print(test_y.shape)


#model
NUM_CLASSES = 2
IMG_SIZE = 224
size = (IMG_SIZE, IMG_SIZE)


inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))


# Using model without transfer learning

outputs = EfficientNetB0(include_top=True, weights=None, classes=NUM_CLASSES)(inputs)


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

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

model.fit(train_x, train_y, epochs=20, callbacks=[cp_callback], verbose=2)

model.save('n_model.h5')


#model.models.save_model("my_model.h5")
#model.save("my_model")
# tf.keras.models.save_model(
#     model,r"E:\codes\EfficentNetB0\Image-Classification-Using-EfficientNets\my_model", overwrite=True, include_optimizer=True, save_format='h5py',
#     signatures=None, options=None, save_traces=True
# )

# serialize model to JSON

# new_model =tf.keras.models.load_model('my_model.h5')
#
# loss, acc = new_model.evaluate(test_x, test_y)
# print("accuracy:{:5.2f}%".format(100*acc))



