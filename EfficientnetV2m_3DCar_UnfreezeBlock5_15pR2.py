import os
import tensorflow as tf
from efficientnet_v2 import *
import glob
import shutil
import sys
import numpy as np
from skimage.io import imread
from tensorflow.keras import callbacks
import pandas as pd
from keras.utils import generic_utils
from efficientnet_v2 import get_preprocessing_layer
from keras import layers
from keras import models
from tensorflow.keras import optimizers
import argparse

my_parser = argparse.ArgumentParser()
my_parser.add_argument('--fold', type=int, default='', help='training Number of fold(1-8)')

args = my_parser.parse_args()

### '''' Seting '''' <---- ''''
## 🚗 Train Fold(1-8)
fold = args.fold
trainfold = f'fold-{fold}'
print(f'Trainning Data Set Fold-{fold}')
print(f'-'*100)

## set tensorflow environ
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] ='false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

## set gpu
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="1"
#tf_device='/gpu:1'
#tf_device='/gpu:0'

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

#Setting
BATCH_SIZE = 4
TARGET_SIZE = (480, 480)  # M variant expects images in shape (480, 480)
epochs = 200

#Train
#database = pd.read_csv('/media/SSD/Data_photogram_Pcar/Dataset_3DCar.csv') #แก้ data เปลี่ยนตาม fold
database = pd.read_csv('/home/kannika/code/Dataset_3DCar_train.csv')
trainframe = database[database['Fold'] != trainfold].reset_index(drop=True)
#base_dir = '/media/SSD/Data_photogram_Pcar/8-Fold/'
print(f'Train Data Shape [ {trainframe.shape} ]')
#test
# testframe = database[database['Fold'] == trainfold].reset_index(drop=True) ###*** เปลี่ยนตาม fold
# print(f'test Data Shape [ {testframe.shape} ]')
print('-'*100)

#load model
from tensorflow.keras.models import load_model

model_dir = f'/media/tohn/SSD/ModelEfficientV2/p15_3Dcar/R1/N{trainfold}/models/EffnetV2m_R1_3DCAR15p_Nfold{fold}.h5'
model = load_model(model_dir)
height = width = model.input_shape[1]


## Create Data Loader
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
      rescale=1./255,
      validation_split=0.25,
      fill_mode='nearest')

train_generator = train_datagen.flow_from_dataframe(
        dataframe = trainframe,
        directory = None ,
        x_col = 'img_path',
        y_col = 'p',
        subset="training",
        target_size = (height, width),
        batch_size=BATCH_SIZE,
        color_mode= 'rgb',
        class_mode='categorical')

valid_generator = train_datagen.flow_from_dataframe(
        dataframe = trainframe,
        directory = None ,
        x_col = 'img_path',
        y_col = 'p',
        subset="validation",
        target_size = (height, width),
        batch_size=BATCH_SIZE,
        color_mode= 'rgb',
        class_mode='categorical')


## Set TensorBoard 
root_logdir = f'/media/tohn/SSD/ModelEfficientV2/p15_3Dcar/R2/N{trainfold}/Mylogs_tensor/'    ##เปลี่ยน path 
if not os.path.exists(root_logdir) :
    os.makedirs(root_logdir)

def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d_%H_%M_%S")
    return os.path.join(root_logdir,run_id)
run_logdir = get_run_logdir()

tensorboard_cb = callbacks.TensorBoard(log_dir=run_logdir)

#Unfreez
print('This is the number of trainable layers '
          'before freezing the conv base:', len(model.trainable_weights))
model.trainable = True
set_trainable = False
for layer in model.layers:
    if layer.name.startswith('block5'):
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
print('This is the number of trainable layers '
      'after freezing the conv base:', len(model.trainable_weights))

model.summary()


#Training model    
model.compile(
    optimizer= optimizers.Adam(learning_rate=0.000001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

checkpoint_filepath = f'/media/tohn/SSD/ModelEfficientV2/p15_3Dcar/R2/N{trainfold}/checkpoint/' ### เปลี่ยน path
if not os.path.exists(checkpoint_filepath) :
        os.makedirs(checkpoint_filepath)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                                filepath=checkpoint_filepath, save_freq='epoch', ave_weights_only=False)


## Fit model 
model.fit(train_generator,
    epochs=epochs,
    validation_data=valid_generator,
    callbacks = [tensorboard_cb, model_checkpoint_callback])

##Save model as TFLiteConverter
modelName = f'EffnetV2m_R2_3DCAR15p_Nfold{fold}'

##Save model as TFLiteConverter
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()

Pth_model_save = f'/media/tohn/SSD/ModelEfficientV2/p15_3Dcar/R2/N{trainfold}/models/'  ##เปลี่ยน path 
if not os.path.exists(Pth_model_save) :
    os.makedirs(Pth_model_save)
    
# Save model as .tflite
# with open(f"{Pth_model_save}{modelName}.tflite", "wb") as file:
#       file.write(tflite_model)
# Save model as .h5        
model.save(f'{Pth_model_save}{modelName}.h5') 
print(f'Save Model as [ {Pth_model_save}{modelName}.h5 ]')
print('*'*120)
