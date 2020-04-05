# Import Necessary Libraries
import glob
import os
import Function_Library
from itertools import groupby
from pathlib import Path
import numpy as np
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from keras.layers import Dense, Reshape
from keras.models import Model
from tqdm import tqdm
import tensorflow as tf

# Shut down Tensorflow Warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Basic Parameters
IMG_HEIGHT, IMG_WIDTH = (224, 224)
img_folder = Path("./NearDuplicateDataset/").expanduser()

# Base Network for Feature Extraction
encoder_model = ResNet50(input_shape=(IMG_HEIGHT,IMG_WIDTH,3), weights='imagenet', include_top=False, pooling='avg')
np.prod(encoder_model.output.shape.as_list()[1:])

# Split Data
image_filenames = glob.glob(str(img_folder / '*.jpg'))
train_filenames = []
for key, items in groupby(sorted(image_filenames), lambda f: Path(f).parts[-2]):
    if key == 'BACKGROUND_Google':
        continue
    train_filenames += items
print("train images", len(train_filenames))
size = len(train_filenames)

# Train DataGenerator
train_datagen = image.ImageDataGenerator()
train_generator = train_datagen.flow_from_directory(directory='NearDuplicateDataset',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=32,
        class_mode=None,
        shuffle=True)

batch_encoded = encoder_model.predict_generator(train_generator, 1)
batch_encoded = batch_encoded.T
encoded_imgs = batch_encoded.reshape(2048, -1)

# Extract Features for all images in the database
#encoded_imgs = Function_Library.load_encode_images(encoder_model, train_filenames).T

# KNN MODEL
def build_knn(model, output_size):
    # Flatten feature vector
    flat_dim_size = np.prod(model.output_shape[1:])
    x = Reshape(target_shape=(flat_dim_size,),
                name='features_flat')(model.output)
    # Dot product between feature vector and reference vectors
    
    x = Dense(units=output_size,
              activation='linear',
              name='dense_1',
              use_bias=False)(x)
    classifier = Model(inputs=[model.input], outputs=x)
    return classifier

# Create Joined Model
joined_model = build_knn(encoder_model, encoded_imgs.shape[1])

# Normalize Encodings
def normalize_ecnodings(encodings):
    ref_norms = np.linalg.norm(encoded_imgs, axis=0)
    return encodings / ref_norms
encoded_imgs_normalized = normalize_ecnodings(encoded_imgs)

# Set Weights to Extracted Features
temp_weights = joined_model.get_weights()
temp_weights[-1] = encoded_imgs_normalized
joined_model.set_weights(temp_weights)



for i in range(len(train_filenames)):
    # Predict
    # example_filename = test_filenames[0[
    example_filename = train_filenames[i]
    print(example_filename)
    # Function_Library.view_image(example_filename)
    example_img = image.load_img(example_filename, target_size=(IMG_WIDTH, IMG_HEIGHT))
    example_img = image.img_to_array(example_img)
    example_img = np.expand_dims(example_img, axis=0)
    example_img = preprocess_input(example_img)
    prediction = joined_model.predict([example_img]).reshape(-1)
    #print("Prediction Original: ", prediction)
    img = ''
    for index in prediction.argsort()[-2:][::-1]:
        img = train_filenames[index]
        if img == example_filename:
            continue
        # Function_Library.view_image(img)
        print(train_filenames[index])
        example_img = image.load_img(train_filenames[index], target_size=(IMG_WIDTH, IMG_HEIGHT))
        example_img = image.img_to_array(example_img)
        example_img = np.expand_dims(example_img, axis=0)
        example_img = preprocess_input(example_img)
        prediction2 = joined_model.predict([example_img]).reshape(-1)
        #print("Prediction of KNN: ", prediction2)
        dist= np.linalg.norm(prediction-prediction2)
        print("Distance: ", dist)
        #if dist > 15:
        #    continue
        # Display Both Images
        Function_Library.view_both_images(example_filename, img)