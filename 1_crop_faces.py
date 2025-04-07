import numpy as np
import cv2
from glob import glob
import matplotlib.pyplot as plt

HARR_CLASSIFIER = 'HaarCascadeModel/haarcascade_frontalface_default.xml'
REDUCE_FACTOR = 1
SCALE_FACTOR = 1.3
MIN_NEIGHBOURS = 3

female_path = glob('./DATASETS/female/*.png') + glob('./DATASETS/female/*.jpg')
male_path = glob('./DATASETS/male/*.png') + glob('./DATASETS/male/*.jpg')

print(f'Male images: {len(male_path)}, Female images: {len(female_path)}')

# Cropping all female images
haar = cv2.CascadeClassifier(HARR_CLASSIFIER)
for i in range(int(len(female_path)/REDUCE_FACTOR)):
    try:
        img = cv2.imread(female_path[i])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        
        faces = haar.detectMultiScale(gray, SCALE_FACTOR, MIN_NEIGHBOURS)
        for x, y, w, h in faces:
            roi = img[y:y+h, x:x+w]
            cv2.imwrite(f'crop_data/female/female_{i}.jpg',roi)
            print(f'female_{i} done')
    except:
        print(f'Processing female_{i} failed')


# Cropping all male images
haar = cv2.CascadeClassifier(HARR_CLASSIFIER)
for i in range(int(len(male_path)/REDUCE_FACTOR)):
    try:
        img = cv2.imread(male_path[i])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        
        faces = haar.detectMultiScale(gray, SCALE_FACTOR, MIN_NEIGHBOURS)
        for x, y, w, h in faces:
            roi = img[y:y+h, x:x+w]
            cv2.imwrite(f'crop_data/male/male_{i}.jpg',roi)
            print(f'male_{i} done')
    except:
        print(f'Processing male_{i} failed')