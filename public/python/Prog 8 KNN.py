#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets

iris=datasets.load_iris()
print("Iris Data set loaded...")

X, x, Y, y = train_test_split(iris.data,iris.target, test_size = 0.1, random_state=50)

model = KNeighborsClassifier(3).fit(X, Y)

pred=model.predict(x)

print("Results of Classification using KNN")

for r in range(0,len(x)):
    print("Sample:", x[r], " Actual:", y[r], " Predicted:",pred[r])
    
print("Accuracy :" , model.score(x,y));


# In[ ]:




