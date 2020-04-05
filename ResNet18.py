# Import Libraries
import mxnet as mx
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple


path='http://data.mxnet.io/models/imagenet/'
[mx.test_utils.download(path+'resnet/18-layers/resnet-18-0000.params'),
 mx.test_utils.download(path+'resnet/18-layers/resnet-18-symbol.json'),
 mx.test_utils.download(path+'synset.txt')]

# Set the context on CPU, switch to GPU if there is one available
ctx = mx.cpu()

# Load the Downloaded Model
sym, arg_params, aux_params = mx.model.load_checkpoint('resnet-18', 0)
mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))], label_shapes=mod._label_shapes)
mod.set_params(arg_params, aux_params, allow_missing=True)
with open('synset.txt','r') as f:
    labels = [l.rstrip() for l in f]
# Define Helper Function
Batch = namedtuple('Batch', ['data'])


'''Image Converts the Image into the Necessary Requirements'''
def get_image(img):
    if img is None:
        return None
    # Convert into format (batch, RGB, width, height)
    img = mx.image.imresize(img, 224, 224)                        # Resize
    img = img.transpose((2, 0, 1))                                  # Channel First
    img = img.expand_dims(axis=0)                                   # Batchify
    img = img.astype('float32')                                     # for GPU context
    return img


''' Get the Internal Layers of the Model'''
def get_internal_ResNet18():
    all_layers = sym.get_internals()                                # Get Internal Layer of the Model
    print(all_layers.list_outputs()[-10:])                          # Print the Last Ten Layers of the Model
    return

# An often used Layer for Feature Extraction is the one before the last fully connected layer.
# For ResNet, and also Inception, it is the flattened layer with name flatten0 which reshapes the 4-d Convolutional Layer
'''Function Provides the Features of every image passed through it'''
'''Function is dependent on the get_image local function'''
def run_image_ResNet18(img):
    all_layers = sym.get_internals()                            # Get Internal Layer of the Model
    fe_sym = all_layers['flatten0_output']
    fe_mod = mx.mod.Module(symbol=fe_sym, context=ctx, label_names=None)
    fe_mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))])
    fe_mod.set_params(arg_params, aux_params)
    # Obtain Forward to Obtain the Features
    img = get_image(img)
    fe_mod.forward(Batch([img]))
    features = fe_mod.get_outputs()[0]
    #print('Shape', features.shape)
    #print(features.asnumpy())
    assert features.shape == (1,512)
    return features
