import json 
import pandas as pd
import numpy as np
from pathlib import Path
import PIL
from PIL import Image
import cv2 
from tqdm import tqdm
import os


Table_path_ = pd.read_csv('/media/SSD/Data_photogram_Pcar/Dataset_3DCar.csv')
print(Table_path_.shape)
img_path = Table_path_['img_path'].tolist()
print(f'img len --->> {len(img_path)}')

sZ_ = []
for im in img_path:
    image = cv2.imread(im)
    sZ = os.path.getsize(im)
    sZ_.append(sZ)
    
df = pd.DataFrame(list(zip(img_path, sZ_)),
               columns =['img_path', 'image size'])
df.to_save('/media/SSD/Data_photogram_Pcar/sZ_3DCar.csv')
print('save file at ---->> [ /media/SSD/Data_photogram_Pcar/sZ_3DCar.csv ]')
