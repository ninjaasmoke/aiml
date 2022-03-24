#!/usr/bin/env python
# coding: utf-8

# In[3]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

names = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width', 'Class']
dataset = pd.read_csv("8dataset.csv", names=names)

X = dataset.iloc[:,0:-1] 
label = {'Iris-setosa': 0,'Iris-versicolor': 1, 'Iris-virginica': 2} 
y = [label[c] for c in dataset.iloc[:, -1]]

def drawGraph(labels, title, num):
	plt.subplot(2, 2, num)
	plt.scatter(X.Petal_Length, X.Petal_Width, c=labels)
	plt.title(title)
	plt.xlabel('Petal Length')
	plt.ylabel('Petal Width')

plt.figure(figsize=(10,10))
drawGraph(y, "Real Clusters", 1)

model = KMeans(3)
model.fit(X)

drawGraph(model.labels_, "K Means Clusters", 2)

gm = GaussianMixture(3).fit(X)
labels = gm.predict(X)
drawGraph(labels, "Gaussian Cluster", 3)


# In[ ]:




