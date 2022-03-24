#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd

data = pd.read_csv('prg1.csv')
			
concepts = np.array(data.iloc[:,0:-1])
targets = np.array(data.iloc[:,-1])
			
def learn(concepts, target):
	s = concepts[0].copy()
	g = [["?" for i in range(len(s))] for i in range(len(s))]
				
	print(f'Specific: {s}\nGeneral: {g}\n\n')
				
	for i, h in enumerate(concepts):
		
		for x in range(len(s)):
			if target[i] == "yes":
				if h[x] != s[x]:
					s[x] = '?'
					g[x][x] = '?'
			else:
				if h[x] != s[x]:
					g[x][x] = s[x]
				else:
					g[x][x] = '?'
					
	G = []
	for val in g:
		if val != ['?', '?', '?', '?', '?', '?']:
			G.append(val)
	return s, G

s, g = learn(concepts, targets)

print(f'Specific: {s}\nGeneral: {g}')


# In[ ]:




