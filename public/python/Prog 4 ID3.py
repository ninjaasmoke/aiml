#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd

data = pd.read_csv('play.csv')

def entropy(target):
	val,counts = np.unique(target,return_counts = True)
	ent = 0
	for i in range(len(val)):
		c = counts[i]/sum(counts)
		ent += -c*np.log2(c)
	return ent

def infoGain(data,features,target):
	te = entropy(data[target])
	val,counts = np.unique(data[features],return_counts = True)
	eg = 0
	for i in range(len(val)):
		c = counts[i]/sum(counts)
		eg += c*entropy(data[data[features] == val[i]][target])
	InfoGain = te-eg
	return InfoGain

def id3(data, features, target, pnode):
	
	t = np.unique(data[target])
		
	if len(t) == 1:
		return t[0]

	if len(features) == 0:
		return pnode

	pnode = t[np.argmax(t[1])]
	
	IG = [infoGain(data,f,target) for f in features]
	index = np.argmax(IG)
	
	col = features[index]
	tree = {col:{}}
	
	features = [f for f in features if f!=col]
	
	for val in np.unique(data[col]):
		sub_data = data[data[col]==val].dropna()
		subtree = id3(sub_data,features,target,pnode)
		tree[col][val] = subtree
	return tree

testData = data.sample(frac = 0.1)
data.drop(testData.index,inplace = True)

target = 'play'
features = data.columns[data.columns!=target]

tree = id3(data,features,target,None)
print (tree, end='\n\n')

test = testData.to_dict('records')[0]
print(test, '=>', id3(test,features,target,None))


# In[ ]:





# In[ ]:





# In[ ]:




