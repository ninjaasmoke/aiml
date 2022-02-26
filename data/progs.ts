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
print("Predicted Value:", predicted)`,
		"output": 
`
Sample: [  1.    153.     82.     42.    485.     40.6     0.687  23.   ]  Actual:  0  Predicted:  1
Sample: [  3.   111.    58.    31.    44.    29.5    0.43  22.  ]  Actual:  0  Predicted:  0
Sample: [  3.    173.     82.     48.    465.     38.4     2.137  25.   ]  Actual:  1  Predicted:  1
Sample: [  0.    134.     58.     20.    291.     26.4     0.352  21.   ]  Actual:  0  Predicted:  0
Sample: [3.00e+00 1.58e+02 6.40e+01 1.30e+01 3.87e+02 3.12e+01 2.95e-01 2.40e+01]  Actual:  0  Predicted:  1
Sample: [  3.  142.   80.   15.    0.   32.4   0.2  63. ]  Actual:  0  Predicted:  0
Sample: [  5.    111.     72.     28.      0.     23.9     0.407  27.   ]  Actual:  0  Predicted:  0
Sample: [ 2.    84.    50.    23.    76.    30.4    0.968 21.   ]  Actual:  0  Predicted:  0
Accuracy:  0.75
`,
	},
	"7": {
		"id": "7",
		"name": "K Means Clustering",
		"code":
			`import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

names = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width', 'Class']
dataset = pd.read_csv("8dataset.csv", names=names)

X = dataset.iloc[:,0:-1] 
label = {'Iris-setosa': 0,'Iris-versicolor': 1, 'Iris-virginica': 2} 
y = [label[c] for c in dataset.iloc[:, -1]]

def drawGraph(labels, title):
	plt.scatter(X.Petal_Length, X.Petal_Width, c=labels)
	plt.title(title)
	plt.xlabel('Petal Length')
	plt.ylabel('Petal Width')
	plt.show()

drawGraph(y, "Real Clusters")

model = KMeans(3)
model.fit(X)

drawGraph(model.labels_, "K Means Clusters")

gm = GaussianMixture(3).fit(X)
labels = gm.predict(X)
drawGraph(labels, "Gaussian Cluster")`
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