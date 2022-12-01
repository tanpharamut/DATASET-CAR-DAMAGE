##Import library
# import os
# os.environ["SM_FRAMEWORK"] = "tf.keras" #before the import
# from segmentation_models import PSPNet
import PIL
from keras import models
from keras import layers
from tensorflow.keras import optimizers
import os
import glob
import shutil
import sys
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from tensorflow.keras import callbacks
import pandas as pd
from keras.utils import generic_utils
import tensorflow as tf
import argparse
#os.SM_FRAMEWORK=tf.keras
my_parser = argparse.ArgumentParser()
my_parser.add_argument('--fold', type=int, default='', help='training Number of fold(1-8)')

args = my_parser.parse_args()

### '''' Seting '''' <---- ''''
## 🚗 Train Fold(1-8)
fold = args.fold
trainfold = f'fold-{fold}'
print(f'Trainning Data Set Fold-{fold}')
print(f'-'*100)

## 🚗 Train Fold2-8 
## set tensorflow environ
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] ='false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
## set gpu
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"
#tf_device='/gpu:1'

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

#batch_size = 128
batch_size = 64
epochs = 200

#Train
database = pd.read_csv('/media/SSD/Data_photogram_Pcar/Dataset_3DCar_OS_solu3.csv') #แก้ data เปลี่ยนตาม fold
trainframe = database[database['Fold'] != trainfold].reset_index(drop=True)
#base_dir = '/media/SSD/Data_photogram_Pcar/8-Fold/'
print(f'Train Data Shape [ {trainframe.shape} ]')
#test
testframe = database[database['Fold'] == trainfold].reset_index(drop=True) ###*** เปลี่ยนตาม fold
print(f'test Data Shape [ {testframe.shape} ]')
print('-'*100)

## Create Model
from efficientnet.keras import EfficientNetB5 as Net
from efficientnet.keras import center_crop_and_resize, preprocess_input

conv_base = Net(weights='imagenet')
height = width = conv_base.input_shape[1]
input_shape = (height, width, 3)

# loading pretrained conv base model
conv_base = Net(weights='imagenet', include_top=False, input_shape=input_shape)

# create new model with a new classification layer
x = conv_base.output  
global_average_layer = layers.GlobalAveragePooling2D(name = 'head_pooling')(x)
dropout_layer_1 = layers.Dropout(0.50,name = 'head_dropout')(global_average_layer)
prediction_layer = layers.Dense(15, activation='softmax',name = 'prediction_layer')(dropout_layer_1)  ###Dense 15 class

model = models.Model(inputs= conv_base.input, outputs=prediction_layer) 
model.summary()

#showing before&after freezing
print('This is the number of trainable layers '
      'before freezing the conv base:', len(model.trainable_weights))
#conv_base.trainable = False  # freeze เพื่อรักษา convolutional base's weight
for layer in conv_base.layers:
    layer.trainable = False
print('This is the number of trainable layers '
      'after freezing the conv base:', len(model.trainable_weights))  #freez แล้วจะเหลือ max pool and dense
model.summary()

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# train_datagen = ImageDataGenerator(
#       rescale=1./255,
#       validation_split=0.25,
#       rotation_range=40,
#       width_shift_range=0.2,
#       height_shift_range=0.2,
#       shear_range=0.2,
#       zoom_range=0.2,
#       horizontal_flip=True,
#       fill_mode='nearest')

train_datagen = ImageDataGenerator(
      rescale=1./255,
      validation_split=0.25,
      fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
        dataframe = trainframe,
        directory = None ,
        x_col = 'img_path',
        y_col = 'p',
        subset="training",
        target_size = (height, width),
        batch_size=batch_size,
        color_mode= 'rgb',
        shuffle=True,
        class_mode='categorical')

valid_generator = train_datagen.flow_from_dataframe(
        dataframe = trainframe,
        directory = None ,
        x_col = 'img_path',
        y_col = 'p',
        subset="validation",
        target_size = (height, width),
        batch_size=batch_size,
        color_mode= 'rgb',
        shuffle=True,
        class_mode='categorical')

# test_generator = test_datagen.flow_from_dataframe(
#         dataframe = testframe,
#         directory = None,
#         x_col = 'img_path',
#         y_col = 'p',
#         target_size = (height, width),
#         batch_size=batch_size,
#         shuffle=False,
#         color_mode= 'rgb',
#         class_mode='categorical')

## Set TensorBoard 
root_logdir = f'/media/SSD/Data_photogram_Pcar/EffnetModel/R1/N{trainfold}/Mylogs_tensor/'  ##เปลี่ยน path 
if not os.path.exists(root_logdir) :
    os.makedirs(root_logdir)
    
def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d_%H_%M_%S")
    return os.path.join(root_logdir,run_id)
run_logdir = get_run_logdir()

tensorboard_cb = callbacks.TensorBoard(log_dir = run_logdir)


def avoid_error(gen):
    while True:
        try:
            data, labels = next(gen)
            yield data, labels
        except:
            pass
        
        
#Training model    
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(learning_rate=2e-5),
              metrics=['accuracy'])


checkpoint_filepath = f'/media/SSD/Data_photogram_Pcar/EffnetModel/R1/N{trainfold}/checkpoint/' ### เปลี่ยน path
if not os.path.exists(checkpoint_filepath) :
        os.makedirs(checkpoint_filepath)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                                filepath=checkpoint_filepath, save_freq='epoch', ave_weights_only=False)

## Fit model 
model.fit(train_generator,
    epochs=epochs,
    validation_data=valid_generator,
    callbacks = [tensorboard_cb, model_checkpoint_callback])

# STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
# STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
# #STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

# model.fit_generator(generator= avoid_error(train_generator),
#                     steps_per_epoch=STEP_SIZE_TRAIN,
#                     validation_data=avoid_error(valid_generator),
#                     validation_steps=STEP_SIZE_VALID,
#                     epochs=epochs,
#                     callbacks = [tensorboard_cb, model_checkpoint_callback]
# )


Pth_model_save = f'/media/SSD/Data_photogram_Pcar/EffnetModel/R1/N{trainfold}/models/'  ##เปลี่ยน path 
if not os.path.exists(Pth_model_save) :
    os.makedirs(Pth_model_save)

modelName = f'Effnet_R1_3DCAR15p_Nfold{fold}.h5'
#save model        
model.save(f'{Pth_model_save}{modelName}') 
print(f'Save Model as [ {Pth_model_save}{modelName} ]')
print('*'*100)
