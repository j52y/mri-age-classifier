import os
import csv
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import zoom

data_num = 111
meta_file = './data/meta.csv'
img_dir = './data/img/'
incorrect_indices = [0, 11, 27, 47]

data_dirs = [('%03d' % (i+1)) for i in range(data_num) if i not in incorrect_indices]

def load_train_data():
  return data_imgs(), data_ages()

def data_imgs():
  return [__data_img(d) for d in data_dirs]
  

# return iso.nii and voxel.nii file
def __data_img(num):
  path = img_dir + num + '/'
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


def image_crop(x, shape=[]):
  s = x.shape
  c0, c1, c2 = shape[0], shape[1], shape[2]
  l0, l1, l2 = round((s[0]-c0)/2), round((s[1]-c1)/2), round((s[2]-c2)/2)
  return x[l0:l0+c0, l1:l1+c1, l2:l2+c2]


def save_sample(crop=[], z=1):
  for d in data_dirs:
    prefix = ''
    img = __data_img(d).get_data()

    if len(crop) == 3:
      img = image_crop(img, crop)
      prefix += 'c'

    if zoom != 1:
      img = zoom(img, 1/z)
      prefix += 'z'
    
    for i in range(3):
      x = np.max(img, axis=i)
      plt.imsave('data/sample/' + prefix + d + '_' + str(i), x)
