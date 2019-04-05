# -*- coding: utf-8 -*-

# --------------------------
# ---       Imports      ---
# --------------------------
from __future__ import print_function
import os
import sys
import numpy as np
import tensorflow as tf
import keras
from keras import optimizers
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.utils import plot_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ------------------------------------
# ---     Preparation for Vispa    ---
# ------------------------------------
# limit GPU memory usage to allow 2 jobs on each GPU
# Please do not change
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.45)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
K.set_session(sess)

# Define output directory
try:
    CONDOR_ID = os.environ['CONDOR_ID']
except:
    sys.exit('Error: Run this script with "pygpu %file"')

condor_dir = os.path.abspath('results-NN-%s' % CONDOR_ID)  # folder for training results
os.makedirs(condor_dir)

# ---------------------------
# ---     Prepare Data    ---
# ---------------------------

# variable definitions
batch_size = 128
num_classes = 10
epochs = 50
lr = 0.1
mom = 0.5
temp=1

num_valid = 10000

# input image dimensions
img_rows, img_cols = 28, 28

# --------------------------------------
# ---      Load MNIST from VISPA     ---
# --------------------------------------
# load the data, split between train, validation, and test sets
(x_, y_), (x_test, y_test) = mnist.load_data(path='/home/public/deeplearning/data/mnist.npz')

x_train, y_train = x_[num_valid:], y_[num_valid:]
x_valid, y_valid = x_[:num_valid], y_[:num_valid]

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_valid = x_valid.reshape(x_valid.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_valid = x_valid.astype('float32')
x_test = x_test.astype('float32')

# normalize input features
x_train /= 255
x_valid /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_valid.shape[0], 'validation samples')
print(x_test.shape[0], 'test samples')
print('training with a batch size of ', batch_size, ' for ', epochs, ' epochs with temperature ', temp)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_valid = keras.utils.to_categorical(y_valid, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# -------------------------
# ---     Functions     ---
# -------------------------

# define custom activation function
def softmaxTemp(x, axis=-1, temp=temp):
    """"Softmax activation function.
    # Arguments
        x: Input tensor.
        axis: Integer, axis along which the softmax normalization is applied.
        temp: Temperature used
    # Returns
        Tensor, output of softmax transformation.
    # Raises
        ValueError: In case `dim(x) == 1`.
    """
    #epsilon = K.constant(1e-6)
    epsilon = 1e-4
    ndim = K.ndim(x)
    if ndim == 1:
        raise ValueError('Cannot apply softmax to a tensor that is 1D')
    else:
        e = K.exp(x / temp)
        s = K.sum(e, axis=axis, keepdims=True)
        res = e / s
        res = K.clip(res,epsilon,1.0-epsilon)
        return res
        
        
get_custom_objects().update({'custom_activation': Activation(softmaxTemp)})


# Define neural network
def create_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, kernel_size=(3, 3), padding = 'same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), padding = 'same', activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), padding = 'same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation(softmaxTemp))
    
    sgd = optimizers.SGD(lr=lr, momentum=mom, nesterov=False)
    
    model.compile(loss=keras.losses.categorical_crossentropy,
    optimizer=sgd,
    metrics=['accuracy'])

    return model
    
# Define test neural network
def create_model_test(input_shape):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, kernel_size=(3, 3), padding = 'same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), padding = 'same', activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), padding = 'same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    
    sgd = optimizers.SGD(lr=lr, momentum=mom, nesterov=False)
    
    model.compile(loss=keras.losses.categorical_crossentropy,
    optimizer=sgd,
    metrics=['accuracy'])

    return model   
    
def predict_with_logits(model, input):
    for i, layer in enumerate(model.layers):
        output = input
        input = layer (input)
    return input, output

# Callback to calculate metrics during training
class CalculateMetrics(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        train_loss, train_acc = model.evaluate(x_train, y_train, verbose=0)
        logs['train_loss'] = train_loss
        logs['train_acc']  = train_acc

get_metrics = CalculateMetrics()

# ---------------------
# ---     Model     ---
# ---------------------

# define callbacks for training
model_chk_path = os.path.join(condor_dir, 'model_weights.h5')
save_best = ModelCheckpoint(model_chk_path, monitor='val_acc',
                      save_best_only=True, save_weights_only=True)

# perform training
model = create_model(input_shape)
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
          verbose=0, validation_data=(x_valid, y_valid),
          callbacks=[save_best, get_metrics])
# load best model
model.load_weights(model_chk_path)
model.save(os.path.join(condor_dir, 'model.h5'))

# ------------------------
# ---     Plotting     ---
# ------------------------
model_pic_path = os.path.join(condor_dir, 'model.png')
plot_model(model, to_file=model_pic_path)
# plot training history
figure_training = plt.figure()
ax_training = figure_training.add_subplot(111)
for key in ['train_acc', 'val_acc']:
    ax_training.plot(history.history[key], label=key)
handles, labels = ax_training.get_legend_handles_labels()
ax_training.legend(handles, labels, loc='upper right')

figure_training.savefig(os.path.join(condor_dir, 'accuracies1.png'))

# ------------------------
# ---     Evaluate     ---
# ------------------------

# evaluate performance
model_test = create_model_test(input_shape)
model_test.load_weights(model_chk_path)
score = model_test.evaluate(x_test, y_test, verbose=0)
print('Test 1 loss:', score[0])
print('Test 1 accuracy:', score[1])

plot_model(model_test, to_file=os.path.join(condor_dir, 'model_test.png'))
model_test.save(os.path.join(condor_dir, 'model_test.h5'))

y_test1 = model_test.predict(x_test)
y_test1 = np.argmax(y_test1, axis =1)

# ---------------------------------
# ---     Distilled Network     ---
# ---------------------------------

# prepare soft labels
y_train2 = model.predict(x_train)

# define callbacks for training
model_chk_path2 = os.path.join(condor_dir, 'model_weights2.h5')
save_best2 = ModelCheckpoint(model_chk_path2, monitor='val_acc',
                      save_best_only=True, save_weights_only=True)  

# perform training for distillation network
model2 = create_model(input_shape)
history2 = model2.fit(x_train, y_train2, batch_size=batch_size, epochs=epochs,
          verbose=0, validation_data=(x_valid, y_valid),
          callbacks=[save_best2, get_metrics])
# load best model
model2.load_weights(model_chk_path2)
model2.save(os.path.join(condor_dir, 'model2.h5'))

# ------------------------
# ---     Plotting     ---
# ------------------------
model_pic_path2 = os.path.join(condor_dir, 'model2.png')
plot_model(model2, to_file=model_pic_path2)
# plot training history
figure_training2 = plt.figure()
ax_training2 = figure_training2.add_subplot(111)
for key in ['train_acc', 'val_acc']:
    ax_training2.plot(history2.history[key], label=key)
handles2, labels2 = ax_training2.get_legend_handles_labels()
ax_training2.legend(handles2, labels2, loc='upper right')

figure_training2.savefig(os.path.join(condor_dir, 'accuracies2.png'))

# ------------------------
# ---     Evaluate     ---
# ------------------------

# evaluate performance
model_test2 = create_model_test(input_shape)
model_test2.load_weights(model_chk_path2)
score2 = model_test2.evaluate(x_test, y_test, verbose=0)
print('Test 1 loss:', score2[0])
print('Test 1 accuracy:', score2[1])

plot_model(model_test2, to_file=os.path.join(condor_dir, 'model_test2.png'))
model_test2.save(os.path.join(condor_dir, 'model_test2.h5'))

y_test2 = model_test2.predict(x_test)
y_test2 = np.argmax(y_test2, axis=1)

y_test_org = np.argmax(y_test, axis=1)


print('Falsch 1: ')
ind2 = np.where(y_test_org!=y_test1)
print(ind2[0])
print(len(ind2[0]))

print('Falsch 2: ')
ind3 = np.where(y_test_org!=y_test2)
print(ind3[0])
print(len(ind3[0]))

print('Ungleich: ')
ind4 = np.where(y_test2!=y_test1)
print(ind4[0])
print(len(ind4[0]))