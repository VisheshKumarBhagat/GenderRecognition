import pandas as pd
import cv2
from glob import glob
import pickle

import warnings
warnings.filterwarnings('ignore')

MAX_IMAGES = 5000
female_path = glob('./crop_data/female/*.jpg')[:MAX_IMAGES]
male_path = glob('./crop_data/male/*.jpg')[:MAX_IMAGES]

df_female = pd.DataFrame(female_path, columns=['filepath'])
df_female['gender'] = 'female'

df_male = pd.DataFrame(male_path, columns=['filepath'])
df_male['gender'] = 'male'

df = pd.concat((df_female, df_male), axis=0)
print(df.shape)

def get_image_size(path):   
    img = cv2.imread(path)
    return img.shape[0]

df['dimension'] = df['filepath'].apply(get_image_size)
print(df.shape)

df_filter = df.query('dimension > 120')

def structuring(path):
    try:
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.shape[0]
        if size>=169:
            gray_resize = cv2.resize(gray,(169,169),cv2.INTER_AREA)
        else:
            gray_resize = cv2.resize(gray,(169,169),cv2.INTER_CUBIC)
        
        flatten_image = gray_resize.flatten()
        return flatten_image

    except:
        return None
    
df_filter['data'] = df_filter['filepath'].apply(structuring)

data = df_filter['data'].apply(pd.Series)
data.columns = [f'pixel_{i}' for i in data.columns]

# Normalize the data
data = data/255.0
data['gender'] = df_filter['gender']

# Drop missing values
data.isnull().sum().sum()
data.dropna(inplace=True)

pickle.dump(data,open('./data/data_images_169_169.pickle',mode='wb'))
print(data.head())