'''Runs the Main Program'''

# Import Necessary Libraries
import mxnet as mx
import NecessaryFunctions
import Batch_PretrainedModel
import Simple_NearestK_Algorithm
import numpy as np
from sklearn.neighbors import NearestNeighbors, KDTree
from numpy import array

# See How Many Images Are in the folder
numImages = NecessaryFunctions.count_images_in_folder()
print("Number of Images are: ", numImages)

# Get the Images from Folder Database
# Format (labels, image numpy)
first_dataset = NecessaryFunctions.read_label_multiple_images()

# Get Features of all the images
# Format (labels, image numpy, features)
second_dataset = Batch_PretrainedModel.batch_ResNet18(first_dataset)

# Run Nearest Neighbor
Features = []
Labels = []
for image in second_dataset:
    Labels.append(image[0])
    Features.append(image[1].asnumpy())
Features = array(Features)
print(type(Features))


