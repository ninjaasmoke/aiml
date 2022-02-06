export const progs: { [id: string]: { [id: string]: string } } = {
	"6": {
		"id": "6",
		"name": "Naive Bayes Classifier",
		"code":
			`import pandas as pd
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
			`import matplotlib.pyplot as plt
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
drawGraph(labels, "Gaussian Cluster", 3)`
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

X, x, Y, y = train_test_split(iris.data,iris.target, test_size = 0.1, random_state=50)

model = KNeighborsClassifier(3).fit(X, Y)

pred=model.predict(x)

print("Results of Classification using KNN")

for r in range(0,len(x)):
    print("Sample:", x[r], " Actual:", y[r], " Predicted:",pred[r])
    
print("Accuracy :" , model.score(x,y));`
	}
}; 