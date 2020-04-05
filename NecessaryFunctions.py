# Necessary Libraries
import matplotlib.pyplot as plt
import mxnet as mx
import glob
import re

'''This Function is designed to read the images from a folder and return them as an array'''
def reading_multiple_images():
    images = []                                     # Initalize Array To Store the Images
    labels = []
    path = 'NearDuplicateDataset\*.*'               # Path of the folder were the images are stored
    for file in glob.glob(path):                    # Function to go through all the images in the folder
        label = re.sub(r'\D',"",file)               # Regex Function to just get the Number Label of the Image
        #print(file)                                # Print File Name with extension
        #print(label)                               # Print Just the Number Label of the image
        img = mx.image.imread(file)                 # Read the Images using MxNet
        images.append(img)                          # Append the Images Into the Array
        labels.append(label)
    return images, labels                                   # Return the Array with all the images stored

''' This Function is designed to read the images from a folder and return them in a two dimensional array'''
def read_label_multiple_images():
    data = []                                           # Master Array that is returned
    path = 'NearDuplicateDataset\*.*'                   # Path of the folder were the images are stored
    for file in glob.glob(path):                        # For Loop to go through all the images in the folder
        image = []                                      # Sub-Array used for hold labe, img info
        label = re.sub(r'\D', "", file)                 # Regex Function to just get the Number Label of the Image
        img = mx.image.imread(file)                     # Read the Images using MxNet
        image.append(label)                             # Append Label to SubArray
        image.append(img)                               # Append Image Information into Subarray
        data.append(image)                              # Append subarray onto main array
    return data                                         # Eeturn Main Array with all the image info


'''Function Counts How Many Images Are in the folder'''
def count_images_in_folder():
    path = 'NearDuplicateDataset\*.*'               # Path of the folder were the images are stored
    count = 0                                       # Set Counter
    for file in glob.glob(path):                    # Function to go through all the images in the folder
        count = count + 1                           # Increment Counter
    return count                                    # Return the Array with all the images stored

'''This Function allows you to view all the images that were in the folder database'''
'''Function calls the reading_multiple_images()'''
def view_images():
    Images = reading_multiple_images()[0]             # Calls the Reading Multiple Function and stores the array
    for image in Images:                            # For loop to go through the Images in the array
        plt.imshow(image.asnumpy())                 # Convert the image into a numpy array to show
        plt.show()                                  # Shows the image
    return                                          # Void Function

'''Function is used to View Single Image'''
def view_image(image):                                                  # Used to View A single Image
    plt.imshow(image.asnumpy())                                         # Convert to Numpy Array
    plt.show()                                                          # View Image
    return

'''Function shows all the images in the Dataset Array'''
def view_images_array(dataset):                                         # View all images in dataset
    for img in dataset:                                                 # Loop to go through the whole array
        plt.imshow(img[1].asnumpy())                                    # View Image
        plt.show()                                                      # Show Image
    return

''' Separate Dataset Into Labels '''
def return_label(data=[]):
    label_dataset = []                                                  # Create New Array that will just hold the labels
    for label in data:                                                  # Split the Array
        label_dataset.append(label[0])                                  # Append Only labels to the Dataset
    return label_dataset                                                # Return the labels in an array

'''Seperate Dataset into the Images'''
def return_images(data):
    image_dataset = []
    for image in data:
        image_dataset.append(image[1])
    return image_dataset

'''Seperate Dataset into the Features'''
def return_features(data):
    features_dataset = []
    for feature in data:
        features_dataset.append(data[2])
    return features_dataset