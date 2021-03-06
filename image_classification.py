# -*- coding: utf-8 -*-
"""Image_Classification.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1j-pbdo7li5u6sZtrh6qqvAe39GwmCTn8
"""

# Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.datasets import fetch_lfw_people

faces = fetch_lfw_people(min_faces_per_person=50)
len(faces.data), faces.target_names

fig, axes = plt.subplots(1, 11, figsize=(16, 3))
fig.subplots_adjust(hspace=0.1, wspace=0.1)
first_img = [np.argmax(faces.target == x) for x in list(range(len(faces.target_names)))]
for i, ax in enumerate(axes.flat):
    idx = first_img[i]
    ax.imshow(faces.data[idx].reshape(62, 47), cmap='gray')
    if i % 2:
        ax.set_title(faces.target_names[i], fontsize=10)
    else:
        ax.set_title(faces.target_names[i], fontsize=10, y=-0.2)
    ax.axis('off')
    
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(faces.data, faces.target, test_size=0.2)

from sklearn.model_selection import KFold, GridSearchCV
model = make_pipeline(PCA(n_components=128, svd_solver='randomized'), SVC())
param_grid = [
    {'svc__kernel': ['linear'], 'svc__C': [1, 10, 100]},
    {'svc__kernel': ['rbf'], 'svc__C': [1, 10, 100], 'svc__gamma': [0.1, 1.0, 10.0]}
]
grid = GridSearchCV(model, param_grid, cv=KFold(n_splits=5, shuffle=True))
grid.fit(xtrain, ytrain)
grid.best_score_, grid.best_estimator_

from sklearn.metrics import classification_report
y_hat = grid.best_estimator_.predict(xtest)
print(classification_report(ytest, y_hat, target_names=faces.target_names))

from sklearn.metrics import confusion_matrix
mat = confusion_matrix(ytest, y_hat)
fig, ax = plt.subplots(figsize=(12, 12))
cax = ax.matshow(mat, cmap='summer')
ticks = np.arange(0,len(faces.target_names))
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(faces.target_names, rotation=45, ha='right')
ax.set_yticklabels(faces.target_names, rotation=45, ha='right')
ax.set_ylabel('true label')
ax.set_xlabel('predicted label')
ax.xaxis.set_ticks_position('bottom')

for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        ax.text(j, i, mat[i, j], ha='center', va='center')
        
        fig, axes = plt.subplots(2, 6, figsize=(16, 5))
fig.subplots_adjust(hspace=0.3, wspace=0.3)
names = faces.target_names
for i, ax in enumerate(axes.flat):
    ax.imshow(xtest[i].reshape(62, 47), cmap='gray')
    ax.set_title('true: %s\npredicted: %s' % (names[ytest[i]], names[y_hat[i]]), fontsize=10)
    ax.axis('off')