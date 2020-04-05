# Import Necessary Libraries
import mxnet as mx
import NecessaryFunctions
import ResNet18

'''Batch Training of the Resnet18 Model'''
def batch_ResNet18(dataset):
    new_daset = []                                                          # Dataset to Hold the Label, image, features
    for image in dataset:
        images = []                                                         # Dataset to Hold the Sub Image Array
        label = image[0]                                                    # Hold the Label Information
        img = image[1]                                                      # Hold the image MX Info
        feature = ResNet18.run_image_ResNet18(img)                          # Send the Image to Get The Features
        images.append(label)                                                # Append Label to New Subarray
        images.append(img)                                                  # Append Image Data to New Subarray
        images.append(feature.asnumpy())                                              # Append Feature Data to New Subarray
        new_daset.append(images)
    return new_daset
