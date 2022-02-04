export const progs: {[id:string] : {[id:string]: string}} = {
    "6" : {
        "id" : "6",
        "name" : "Naive Bayes Classifier",
        "code": 
`import csv
import pandas as pd
import numpy as np

from sklearn.naive_bayes import GaussianNB
data = pd.read_csv('prima_indian_diabetes.csv')

x = np.array(data.iloc[:,0:-1])
y = np.array(data.iloc[:,-1])

print(data.head())

#Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training sets
model.fit(x,y)

predicted= model.predict([[6,149,78,35,0,34,0.625,54]])
print("Predicted Value:", predicted)`
    },
    "7": {
        "id": "7",
        "name": "K Means Clustering",
        "code":
`from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('8-kmeansdata.csv')

f1 =data['Distance_Feature']
f2=data['Speeding_Feature']

X =np.array(list(zip(f1,f2)))
plt.scatter(f1,f2,color='black')
plt.show()

kmeans = KMeans(3).fit(X)
labels = kmeans.predict(X)

plt.scatter(f1,f2,c=labels)
plt.show()

gm = GaussianMixture(3).fit(X)
labels = gm.predict(X)
plt.scatter(f1,f2,c=labels)

plt.show()`
    },
    "8": {
        "id": "8",
        "name": "K Nearest Neighbors",
        "code": 
`from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets

iris=datasets.load_iris()
print("Iris Data set loaded...")
x_train, x_test, y_train, y_test = train_test_split(iris.data,iris.target)
classifier = KNeighborsClassifier(3).fit(x_train, y_train)
y_pred=classifier.predict(x_test)

print("Results of Classification using K-nn with K=1 ")
for r in range(0,len(x_test)):
print(" Sample:", str(x_test[r]), " Actual-label:", str(y_test[r]), " Predictedlabel:",str)
print("Classification Accuracy :" , classifier.score(x_test,y_test));


### different file
### this code below is for visualization of the iris data set

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets

iris=datasets.load_iris()
print("Iris Data set loaded...")
print(iris.data.shape)
print(iris.feature_name
print(iris.target_names)
print(iris.data)
print(iris.target)

x_train, x_test, y_train, y_test = train_test_split(iris.data,iris.target)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

classifier = KNeighborsClassifier(3).fit(x_train, y_train)
y_pred=classifier.predict(x_test)
print(len(x_test))

for r in range(0,len(x_test)):
print(" Sample:", str(x_test[r]), " Actual-label:", str(y_test[r]), " Predictedlabel:",str
print("Classification Accuracy :" , classifier.score(x_test,y_test));

x_test = [[3,4,5,2],[5,4,5,2]]
y_pred=classifier.predict(x_test)
print(y_pred)`
    },
}; 