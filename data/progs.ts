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
    "6alt": {
        "id": "6 alt",
        "name": "Naive Bayes Alt",
        "code":
`import csv
import random
import math


def loadcsv(filename):
	lines = csv.reader(open(filename, "r"));
	dataset = list(lines)
	for i in range(len(dataset)):
       #converting strings into numbers for processing
		dataset[i] = [float(x) for x in dataset[i]]
	return dataset


def splitdataset(dataset, splitratio):
    #67% training size
	trainsize = int(len(dataset) * splitratio);
	trainset = []
	copy = list(dataset);    
	while len(trainset) < trainsize:
        #generate indices for the dataset list randomly to pick ele for training data
		index = random.randrange(len(copy));       
		trainset.append(copy.pop(index))    
	return [trainset, copy]


def separatebyclass(dataset):
	separated = {} #dictionary of classes 1 and 0 
    #creates a dictionary of classes 1 and 0 where the values are 
    #the instances belonging to each class
	for i in range(len(dataset)):
		vector = dataset[i]
		if (vector[-1] not in separated):
			separated[vector[-1]] = []
		separated[vector[-1]].append(vector)
	return separated

def mean(numbers):
	return sum(numbers)/float(len(numbers))

def stdev(numbers):
	avg = mean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
	return math.sqrt(variance)

def summarize(dataset): #creates a dictionary of classes
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)];
	del summaries[-1] #excluding labels +ve or -ve
	return summaries

def summarizebyclass(dataset):
	separated = separatebyclass(dataset); 
    #print(separated)
	summaries = {}
	for classvalue, instances in separated.items(): 
        #for key,value in dic.items()
        #summaries is a dic of tuples(mean,std) for each class value        
		summaries[classvalue] = summarize(instances) #summarize is used to cal to mean and std
	return summaries

def calculateprobability(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

def calculateclassprobabilities(summaries, inputvector):
	probabilities = {} # probabilities contains the all prob of all class of test data
	for classvalue, classsummaries in summaries.items():#class and attribute information as mean and sd
		probabilities[classvalue] = 1
		for i in range(len(classsummaries)):
			mean, stdev = classsummaries[i] #take mean and sd of every attribute for class 0 and 1 seperaely
			x = inputvector[i] #testvector's first attribute
			probabilities[classvalue] *= calculateprobability(x, mean, stdev);#use normal dist
	return probabilities

def predict(summaries, inputvector): #training and test data is passed
	probabilities = calculateclassprobabilities(summaries, inputvector)
	bestLabel, bestProb = None, -1
	for classvalue, probability in probabilities.items():#assigns that class which has he highest prob
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classvalue
	return bestLabel

def getpredictions(summaries, testset):
	predictions = []
	for i in range(len(testset)):
		result = predict(summaries, testset[i])
		predictions.append(result)
	return predictions

def getaccuracy(testset, predictions):
	correct = 0
	for i in range(len(testset)):
		if testset[i][-1] == predictions[i]:
			correct += 1
	return (correct/float(len(testset))) * 100.0


def main():
	filename = 'naivedata.csv'
	splitratio = 0.67
	dataset = loadcsv(filename);
     
	trainingset, testset = splitdataset(dataset, splitratio)
	print('Split {0} rows into train={1} and test={2} rows'.format(len(dataset), len(trainingset), len(testset)))
	# prepare model
	summaries = summarizebyclass(trainingset);    
	#print(summaries)
    # test model
	predictions = getpredictions(summaries, testset) #find the predictions of test data with the training data
	accuracy = getaccuracy(testset, predictions)
	print('Accuracy of the classifier is : {0}%'.format(accuracy))


main()`
    }
}; 