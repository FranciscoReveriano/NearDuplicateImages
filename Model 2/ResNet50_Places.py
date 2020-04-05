from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.applications.resnet50 import ResNet50, preprocess_input
import numpy
import os

IMG_HEIGHT = 224
IMG_WIDTH = 224
encoder_model = ResNet50(input_shape=(IMG_HEIGHT,IMG_WIDTH,3), weights='imagenet', include_top=False, pooling='avg')
model_json = encoder_model.to_json()
with open("ResNet50_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
encoder_model.save_weights("ResNet50_model.h5")
print("Saved model to disk")