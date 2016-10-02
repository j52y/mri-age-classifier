import os
import csv
import numpy as np
import nibabel as nib

data_num = 111
meta_file = './data/meta.csv'
img_dir = './data/img/'
incorrect_indices = [0, 11, 27]

data_dirs = [('%03d/' % (i+1)) for i in range(data_num) if i not in incorrect_indices]

def load_train_data():
  return data_imgs(), data_ages()

def data_imgs():
  return [__data_img(d) for d in data_dirs]
  

# return iso.nii and voxel.nii file
def __data_img(num):
  path = img_dir + num
  files = os.listdir(path) 
  files = [f for f in files if f.endswith('voxel.nii')]
  return nib.load(path + files[0])


def data_ages():
  f = open(meta_file, encoding='utf-8')
  reader = csv.reader(f)
  header = next(reader, None)
  age_index = header.index('age')
  ages = [int(row[age_index]) for i, row in enumerate(reader) if i not in incorrect_indices]
  return ages

