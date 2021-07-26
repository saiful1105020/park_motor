'''Deep Dreaming in Keras.

Run the script with:
```
python deep_dream.py path_to_your_base_image.jpg prefix_for_results
```
e.g.:
```
python deep_dream.py img/mypic.jpg results/dream
```
'''
from __future__ import print_function
import json
import argparse
from os import system
import numpy as np
import scipy.io
import h5py
import sys

from keras.models import load_model
from keras import backend as K
import tensorflow as tf
import mat73
tf.compat.v1.disable_eager_execution()

#%%
parser = argparse.ArgumentParser()
# data I/O
parser.add_argument('-p', '--nb_sub', type=int, default=1,
                    help='nb_sub')
parser.add_argument('-t', '--nb_task', type=int, default=1,
                    help='nb_task')
parser.add_argument('-s', '--nb_split', type=int, default=1,
                    help='nb_split')                                                  
parser.add_argument('-v', '--video', type=str, default="",
                    help='video file')
parser.add_argument('-o', '--out_dir', type=str, default="",
                    help='out_dir')                                                
args = parser.parse_args()
#print('input args:\n', json.dumps(vars(args), indent=4, separators=(',', ':'))) # pretty print args

#%%
# These are the names of the layers
# for which we try to maximize activation,
# as well as their weight in the final loss
# we try to maximize.
# You can tweak these setting to obtain new visual effects.
layer_name = 'dense_2'

data_dir = args.video

model_dir = 'E:/Saiful/park_motor/DeepMAG/model/model15.h5'


#%%
def deprocess_image(x):
    # Util function to convert a tensor into a valid image.
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, x.shape[2], x.shape[3]))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((x.shape[1], x.shape[2], 3))
    x /= 2.
    x += 0.5
    x *= 255.
    x = np.clip(x, 0, 255).astype('uint8')
    return x

#%%
K.set_learning_phase(0)

# Build the InceptionV3 network with our placeholder.
# The model will be loaded with pre-trained ImageNet weights.
#model = load_model(data_dir + '/data/center36x36difference_raw_motion_final_ind/T1T7B2/split0/model47.h5')
model = load_model(model_dir)
dream = model.input
#print('Model loaded.')

# Get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers])

#%%
# Define the loss.
loss = K.variable(0.)
# Add the L2 norm of the features of a layer to the loss.
assert layer_name in layer_dict.keys(), 'Layer ' + layer_name + ' not found in model.'
x = layer_dict[layer_name].output
# We avoid border artifacts by only involving non-border pixels in the loss.
loss = K.square(x) # K.sum(K.square(x))

# Compute the gradients of the dream wrt the loss.
grads = K.gradients(loss, dream)[0]
grads = grads * K.sign(grads * dream)
#grads = grads * K.clip(K.sign(grads * dream), 0, np.inf)
# Normalize gradients.
grads /= K.maximum(K.mean(K.abs(grads)), K.epsilon())

# Set up function to retrieve the value
# of the loss and gradients given an input image.
outputs = [loss, grads]
fetch_loss_and_grads = K.function([dream], outputs)

def eval_loss_and_grads(x):
    outs = fetch_loss_and_grads([x])
    loss_value = outs[0]
    grad_values = outs[1]
    return loss_value, grad_values

def gradient_ascent(x, iterations, step, max_loss=None):
    for i in range(iterations):
        loss_value, grad_values = eval_loss_and_grads(x)
        if max_loss is not None and loss_value > max_loss:
            break
        #print('..Loss value at', i, ':', loss_value)
        x += step * grad_values
#        print('Max grad', max(grad_values.flatten()))
    return x

def save_img(img, fname):
    pil_img = deprocess_image(np.copy(img))
    scipy.misc.imsave(fname, pil_img)

#%%
f1 = h5py.File(data_dir)
#f1 = mat73.loadmat(data_dir)
#sys.exit(0)
dXsub = np.transpose(np.array(f1["dXsub"]))
sdNew = np.array(f1["sdNew"])

# Playing with these hyperparameters will also allow you to achieve new effects
step = 0.2/2/sdNew[0,0]  # Gradient ascent step size 0.0001
iterations = 20  # Number of ascent steps per scale
#max_loss = 100000.

#%%
Xnew = np.zeros((len(dXsub), 123, 123, 4),dtype=np.float32)
for i in range(len(dXsub)):
    img = dXsub[[i],:,:,:]
    img_out = gradient_ascent(img,
                          iterations=iterations,
                          step=step)
    Xnew[i,:,:,:] = img_out              
#    save_img(img_out, fname='result.png')
#scipy.io.savemat('/media/cvx/Windows7_OS/P'+str(args.nb_sub)+'T'+str(args.nb_task)+'VideoB2_phase.mat', mdict={'Xnew': Xnew})
#print(args.video[0:-4]+'Mag.mat')
scipy.io.savemat(args.video[0:-4]+'Mag.mat', mdict={'Xnew': Xnew})