from matplotlib.pyplot import imread
from matplotlib.pyplot import imshow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow import keras
from keras.models import Model, load_model
import tensorflow as tf
import cv2
import numpy as np

loaded_model = load_model('E:\\codes\\EfficentNetB0\\Image-Classification-Using-EfficientNets\\n_model.h5')


img_path = 'unseen_imagenet.jfif'

img = image.load_img(img_path, target_size=(224, 224))
#x = img.img_to_array(img)
x = np.array(img)

#img = cv2.imread(img_path)
#img = cv2.resize(img, (224, 224))

x = np.expand_dims(img, axis=0)
x = preprocess_input(x)

print('Input image shape:', x.shape)

my_image = imread(img_path)
#cv2.imshow(my_image)


preds= loaded_model.predict(x)
print(preds)