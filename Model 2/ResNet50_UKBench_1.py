
# Import Libraries
import glob
from itertools import groupby
from pathlib import Path
import numpy as np
import tensorflow.keras
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense, Reshape
from tensorflow.keras.models import Model, model_from_json
import tensorflow as tf
import time
import re


# ### Check For GPU
device_name = tf.test.gpu_device_name()
print(device_name)
if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))
config = tf.ConfigProto()
sess = tf.Session(config=config)
tensorflow.keras.backend.set_session(sess)

# Encode Images
TotalStartTime = time.time()
IMG_HEIGHT, IMG_WIDTH = (224, 224)
img_folder = Path("./UKBench").expanduser()

# ### Load Model
''' Load Model Directly From Network Repository'''
# encoder_model = ResNet50(input_shape=(IMG_HEIGHT,IMG_WIDTH,3), weights='imagenet', include_top=False, pooling='avg')

''' Load Model Directly From Saved JSON File'''
json_file = open('ResNet50_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
encoder_model = model_from_json(loaded_model_json)
# load weights into new model
encoder_model.load_weights("ResNet50_model.h5")
print("Loaded model from disk")
np.prod(encoder_model.output.shape.as_list()[1:])
len(encoder_model.get_weights())

# ### Encode Images
# Read File Pathnames
image_filenames = glob.glob(str(img_folder / '*.jpg'))
train_filenames = []
for key, items in groupby(sorted(image_filenames), lambda f: Path(f).parts[-2]):
    if key == 'BACKGROUND_Google':
        continue
    train_filenames += items

# Get the Length of the Images
print("train images", len(train_filenames))
database_size = len(train_filenames)

# Encode the Images
StartEncodeTime = time.time()                                                         # Get Starting Time for the Encoding
with tf.device('/GPU:0'):
    encoded_dim = np.prod(encoder_model.output.shape[1:]).value
    encoded = np.zeros((database_size, encoded_dim))
    StartDatabaseTrain = time.time()
    count = 0
    for filename in train_filenames:
        # Train Image For Features
        img = image.load_img(filename, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = encoder_model.predict(x)
        features_flat = features.reshape(1, -1)
        encoded[count] = features_flat
        if count % 1000 == 0:
            print("Count: ", count)
        count = count + 1
    encoded_imgs = encoded.T
    EndDatabaseTime = time.time()
    EndEncodeTime = time.time()                                                     # Get the Final Time of the Encoding
    print("All Images Have Been Encoded")


# ## Build KNN Model
# Build KNN Model
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
# Normalize Images
encoded_imgs_normalized = normalize_ecnodings(encoded_imgs)
# ### Set Weights to Extracted Features
temp_weights = joined_model.get_weights()
temp_weights[-1] = encoded_imgs_normalized
joined_model.set_weights(temp_weights)

# ### Create Dictionary Array With Correct Labels
''' This Part of the Program Creates An Array with Subarrays that hold the correct labels by index version'''
with tf.device('/GPU:0'):
    dictionary = []
    for i in range(0, len(train_filenames), 4):
        subdictionary = []
        for j in range(0, 4):
            label = re.sub(r'\D', "", train_filenames[i + j])
            subdictionary.append(label)
        # print(subdictionary)
        for k in range(0, 4):
            dictionary.append(subdictionary)

# ### Identify Nearest Image Pairs
CheckStartTime = time.time()                                                # Check Start Time For Testing Images
with tf.device('/GPU:0'):
    count = 0
    subCount = 0
    PairsIdentified = 0
    Distance_List = []
    Covariance_List = []
    Correlation_List = []
    subPair = []
    for i in range(len(train_filenames)):
        if count == 3:
            subPair.append(subCount)
            count = 0
            subCount = 0
        # Get Dictionary Array For This Image
        correctList = dictionary[i]
        # Predict
        example_filename = train_filenames[i]
        label1 = re.sub(r'\D', "", example_filename)
        # Load Test Image into Nearest Neighbor Map
        example_img = image.load_img(example_filename, target_size=(IMG_WIDTH, IMG_HEIGHT))
        example_img = image.img_to_array(example_img)
        example_img = np.expand_dims(example_img, axis=0)
        example_img = preprocess_input(example_img)
        prediction = joined_model.predict([example_img]).reshape(-1)
        # Show Nearest Neighbors to Image
        img = ''
        for index in prediction.argsort()[-2:][::-1]:
            # Declare SubList
            pairList = []
            # Make Sure Nearest Neighbor Is Not the Same Image
            img = train_filenames[index]
            if img == example_filename:
                continue
            # Get Label of Image
            label2 = train_filenames[index]
            label2 = re.sub(r'\D', "", label2)
            example_img = image.load_img(train_filenames[index], target_size=(IMG_WIDTH, IMG_HEIGHT))
            example_img = image.img_to_array(example_img)
            example_img = np.expand_dims(example_img, axis=0)
            example_img = preprocess_input(example_img)
            prediction2 = joined_model.predict([example_img]).reshape(-1)
            # Find Distance of Numpy Array in Map
            dist = np.linalg.norm(prediction - prediction2)
            # Check if Nearest Neighbor Map is Correct
            if label2 in correctList:
                PairsIdentified = PairsIdentified + 1
                subCount = subCount + 1
        count = count + 1

# Time Commands
CheckEndTime = time.time()
TotalEndTime = time.time()                                                  # Get the Final Running Time To Calculate Total Time
AverageAccuracy = np.mean(subPair)

print("------------------------ Results ---------------------------")
print("Number of Near Identical Images Identified is: ", PairsIdentified)
print("Accuracy: ", PairsIdentified / 10200)
print("Pair Accuracy: ", AverageAccuracy)
print("------------------------ Time ---------------------------")
print("Total Program Running Time: ", TotalEndTime - TotalStartTime)
print("Total Time to Encode Images: ", StartEncodeTime - EndEncodeTime)
print("Total time to Check Images: ", CheckEndTime - CheckStartTime)


