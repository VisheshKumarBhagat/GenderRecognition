import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import pickle

from sklearn.decomposition import PCA

data = pickle.load(open('data/data_images_150_150.pickle', 'rb'))
# print(data)

X = data.drop('gender', axis=1)
print(X)

mean_face = pd.Series(X.mean(axis=0))
print(mean_face.values.reshape((150, 150)))
plt.imshow(mean_face.values.reshape((150, 150)),cmap='gray')
plt.axis('off')
plt.show()

# Transformed data
X_t = X-mean_face
# pca = PCA(n_components=None, whiten=True, svd_solver='auto')
# pca.fit(X_t)

# exp_var_df = pd.DataFrame()
# exp_var_df['explained_var'] = pca.explained_variance_ratio_
# exp_var_df['cum_explained_var'] = exp_var_df['explained_var'].cumsum()
# exp_var_df['principal_components'] = np.arange(1,len(exp_var_df)+1)

# exp_var_df.set_index('principal_components',inplace=True)

pca_50 = PCA(n_components=50,whiten=True,svd_solver='auto')
pca_data = pca_50.fit_transform(X_t)

# saving data
y = data['gender'].values # independent variables
np.savez('./data/data_pca_50_target',pca_data,y)
# saving the model
pca_dict = {'pca':pca_50,'mean_face':mean_face}
pickle.dump(pca_dict,open('model/pca_dict.pickle','wb'))