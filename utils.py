import os
import csv
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import zoom

data_num = 111
meta_file = './data/meta.csv'
img_dir = './data/img/'
pre_dir = './data/pre/'

incorrect_indices = [0, 11, 27, 47]
validation_indices = [31, 21, 2, 65, 100, 3, 30, 84] # ages [48, 55, 58, 63, 66, 71, 77, 40]
train_indices = [ i for i in range(data_num) 
                    if i not in incorrect_indices and i not in validation_indices]

def load_train_data():
  return load_data(train_indices)


def load_validation_data():
  return load_data(validation_indices)


def load_data(indices):
  return data_imgs(indices), data_ages(indices)


def data_imgs(indices):
  return [__data_img(i) for i in indices]
#  return [__preprocessed_img(i) for i in indices]
  

# return iso.nii and voxel.nii file
def __data_img(i):
  num = '%03d' % (i+1)
  path = img_dir + num + '/'
  files = os.listdir(path) 
  files = [f for f in files if f.endswith('voxel.nii')]
  return nib.load(path + files[0])


def __preprocessed_img(i):
  path = pre_dir + str(i) + '.nii.gz'
  return nib.load(path)


def data_ages(indices):
  f = open(meta_file, encoding='utf-8')
  reader = csv.reader(f)
  header = next(reader, None)
  age_index = header.index('age')
  ages = [int(row[age_index]) for i, row in enumerate(reader) if i in indices]
  return ages


def crop_image(x, shape=[]):
  s = x.shape
  c0, c1, c2 = shape[0], shape[1], shape[2]
  l0, l1, l2 = round((s[0]-c0)/2), round((s[1]-c1)/2), round((s[2]-c2)/2)
  return x[l0:l0+c0, l1:l1+c1, l2:l2+c2]


def normalize_image(x):
  minimum = np.min(x)
  maximum = np.max(x)
  return (x - minimum) / (maximum - minimum)


def save_sample(indices, preprocess=False, crop=[], z=1):
  for d in indices:
    prefix = ''
    img = __data_img(d).get_data()

    if preprocess:
      img = preprocess(img)
      prefix += 'p'

    if len(crop) == 3:
      img = image_crop(img, crop)
      prefix += 'c'

    if zoom != 1:
      img = zoom(img, 1/z)
      prefix += 'z'
    
    for i in range(3):
      x = np.max(img, axis=i)
      plt.imsave('data/sample/' + prefix + d + '_' + str(i), x)


def save_preprocessed():
  indices = train_indices + validation_indices
  indices = sorted(indices)[80:]
  print(indices)
  for i in indices:
    print(i)
    img = __data_img(i)
    p = nib.Nifti1Image(preprocess(img.get_data()), np.eye(4))
    nib.save(p, './data/pre/' + str(i) + '.nii.gz') 


def preprocess(img):
  p = np.max(img, axis=0)
  avg = np.average(p[100:130, :30])
  avg2= np.average(p[360:420, :60])
  avg = max(avg, avg2)
  avg = avg * 1.15
  img[img<avg] = 0

  d1, d2, d3 = img.shape
  w = 2
  for k in range(d1-w):
    print(d1, k)
    for j in range(d2-w):
      for h in range(d3-w):
        m = np.max(img[k:k+w, j:j+w, h:h+w])
        if (m < avg*1.15):
          img[k:k+w, j:j+w, h:h+w] = 0

  return img
