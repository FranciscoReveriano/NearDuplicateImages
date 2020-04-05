# Import Necessary Libraries
import numpy as np
from tqdm import tqdm
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

'''Load and Encodes the Images'''
def load_encode_images(encoder, filenames):
    batch_size = 16
    encoded_dim = np.prod(encoder.output.shape[1:]).value
    file_count = len(filenames)
    encoded = np.zeros((file_count, encoded_dim))
    for start_index in tqdm(list(range(0, file_count, batch_size))):
        end_index = min(start_index + batch_size, file_count)
        batch_filenames = filenames[start_index:end_index]

        batch_images = load_images(batch_filenames)
        batch_encoded = encoder.predict(batch_images)
        batch_encoded_flat = batch_encoded.reshape(len(batch_images), -1)
        encoded[start_index:end_index, :] = batch_encoded_flat

    return encoded

'''Load The Necessary Images'''
def load_images(filenames):
    IMG_HEIGHT, IMG_WIDTH = (224, 224)
    images = np.zeros((len(filenames), IMG_HEIGHT, IMG_WIDTH, 3))
    for i, filename in enumerate(filenames):
        img = image.load_img(filename, target_size=(IMG_HEIGHT,IMG_WIDTH))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        images[i, :, :, :] = img_array
    return images

def view_image(filename):
    img = mpimg.imread(filename)
    plt.imshow(img)
    plt.show()
    return

def view_both_images(filename1, filename2):
    fig = plt.figure()
    a = fig.add_subplot(1,2,1)
    img1 = mpimg.imread(filename1)
    img2 = mpimg.imread(filename2)
    plt.imshow(img1)
    a.set_title("Original")
    a = fig.add_subplot(1,2,2)
    plt.imshow(img2)
    a.set_title("KNN Image")
    plt.show()
    return

