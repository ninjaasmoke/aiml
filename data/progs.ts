export const progs: { [id: string]: { [id: string]: string } } = {
	"1": {
		"id": "1",
		"repo": "https://github.com/ninjaasmoke/aiml-progs/tree/main/A%20Star",
		"name": "A Star Search",
		"code":
			`class Graph:

		def __init__(self, adjacency_list):
			self.adjacency_list = adjacency_list
	
		def get_neighbors(self, v):
			return self.adjacency_list[v]
		
		def h(self, n):
			H = {
				'A': 1,
				'B': 1,
				'C': 1,
				'D': 1
			}
			return H[n]
	
		def a_star_algorithm(self, start_node, stop_node):
			open_list = set([start_node])
			closed_list = set([])
	
			g = {}
			g[start_node] = 0
	
			parents = {}
			parents[start_node] = start_node
	
			while len(open_list) > 0:
				n = None
	
				for v in open_list:
					if n == None or g[v] + self.h(v) < g[n] + self.h(n):
						n = v;
	
				if n == None:
					print('Path does not exist!')
					return None
	
				if n == stop_node:
					recon = []
	
					while parents[n] != n:
						recon.append(n)
						n = parents[n]
	
					recon.append(start_node)
					recon.reverse()
	
					print(f'Path found: {recon}')
					return recon
	
				for (m, weight) in self.get_neighbors(n):
					if m not in open_list and m not in closed_list:
						open_list.add(m)
						parents[m] = n
						g[m] = g[n] + weight

					else:
						if g[m] > g[n] + weight:
							g[m] = g[n] + weight
							parents[m] = n
	
							if m in closed_list:
								closed_list.remove(m)
								open_list.add(m)

				open_list.remove(n)
				closed_list.add(n)
	
			print('Path does not exist!')
			return None

adjacency_list = {
	'A': [('B', 1), ('C', 3), ('D', 7)],
	'B': [('D', 5)],
	'C': [('D', 12)]
}
graph1 = Graph(adjacency_list)
graph1.a_star_algorithm('A', 'D')`,
		"output": "Path found: ['A', 'B', 'D']",
	},
	"2": {
		"id": "2",
		"repo": "https://github.com/ninjaasmoke/aiml-progs/tree/main/AO%20star",
		"name": "AO Star Search",
		"code":
			`class Graph:
	def __init__(self, graph, heuristicNodeList, startNode):
		self.graph = graph
		self.H=heuristicNodeList
		self.start=startNode
		self.parent={}
		self.status={}
		self.solutionGraph={}
		
	def applyAOStar(self): 
		self.aoStar(self.start, False)

	def getNeighbors(self, v):
		return self.graph.get(v,'')

	def getStatus(self,v):
		return self.status.get(v,0)

	def setStatus(self,v, val): 
		self.status[v]=val

	def getHeuristicNodeValue(self, n):
		return self.H.get(n,0) 

	def setHeuristicNodeValue(self, n, value):
		self.H[n]=value 

	def printSolution(self):
		print("FOR GRAPH SOLUTION, TRAVERSE THE GRAPH FROM THE START NODE:",self.start)
		print("------------------------------------------------------------")
		print(self.solutionGraph)
		print("------------------------------------------------------------")

	def computeMinimumCostChildNodes(self, v): 
		minimumCost=0
		costToChildNodeListDict={}
		costToChildNodeListDict[minimumCost]=[]
		flag=True
		for nodeInfoTupleList in self.getNeighbors(v): 
			cost=0
			nodeList=[]
			for c, weight in nodeInfoTupleList:
				cost=cost+self.getHeuristicNodeValue(c)+weight
				nodeList.append(c)
			if flag==True: 
				minimumCost=cost
				costToChildNodeListDict[minimumCost]=nodeList 
				flag=False
			else: 
				if minimumCost>cost:
					minimumCost=cost
					costToChildNodeListDict[minimumCost]=nodeList 
		return minimumCost, costToChildNodeListDict[minimumCost] 

	def aoStar(self, v, backTracking): 
		print("HEURISTIC VALUES :", self.H)
		print("SOLUTION GRAPH :", self.solutionGraph)
		print("PROCESSING NODE :", v)
		print("-----------------------------------------------------------------------------------------")
		if self.getStatus(v) >= 0: 
			minimumCost, childNodeList = self.computeMinimumCostChildNodes(v)
#             print(minimumCost, childNodeList)
			self.setHeuristicNodeValue(v, minimumCost)
			self.setStatus(v,len(childNodeList))
			solved=True # check the Minimum Cost nodes of v are solved
			for childNode in childNodeList:
				self.parent[childNode]=v
				if self.getStatus(childNode)!=-1:
					solved=solved & False
			if solved==True: 
				self.setStatus(v,-1)
				self.solutionGraph[v]=childNodeList 
			if v!=self.start:
				self.aoStar(self.parent[v], True) 
			if backTracking==False: # check the current call is not for backtracking 
				for childNode in childNodeList:
					self.setStatus(childNode,0) 
					self.aoStar(childNode, False)  


h1 = {'A': 1, 'B': 6, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 5, 'H': 7, 'I': 7, 'J': 1}
graph1 = {
	'A': [[('B', 1), ('C', 1)], [('D', 1)]],
	'B': [[('G', 1)], [('H', 1)]],
	'C': [[('J', 1)]],
	'D': [[('E', 1), ('F', 1)]],
	'G': [[('I', 1)]]
}

G1= Graph(graph1, h1, 'A')
G1.applyAOStar()
G1.printSolution()`,
		"output": `HEURISTIC VALUES : {'A': 1, 'B': 6, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 5, 'H': 7, 'I': 7, 'J': 1}
SOLUTION GRAPH : {}
PROCESSING NODE : A
-----------------------------------------------------------------------------------------
HEURISTIC VALUES : {'A': 10, 'B': 6, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 5, 'H': 7, 'I': 7, 'J': 1}
SOLUTION GRAPH : {}
PROCESSING NODE : B
-----------------------------------------------------------------------------------------
HEURISTIC VALUES : {'A': 10, 'B': 6, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 5, 'H': 7, 'I': 7, 'J': 1}
SOLUTION GRAPH : {}
PROCESSING NODE : A
-----------------------------------------------------------------------------------------
HEURISTIC VALUES : {'A': 10, 'B': 6, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 5, 'H': 7, 'I': 7, 'J': 1}
SOLUTION GRAPH : {}
PROCESSING NODE : G
-----------------------------------------------------------------------------------------
HEURISTIC VALUES : {'A': 10, 'B': 6, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 8, 'H': 7, 'I': 7, 'J': 1}
SOLUTION GRAPH : {}
PROCESSING NODE : B
-----------------------------------------------------------------------------------------
HEURISTIC VALUES : {'A': 10, 'B': 8, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 8, 'H': 7, 'I': 7, 'J': 1}
SOLUTION GRAPH : {}
PROCESSING NODE : A
-----------------------------------------------------------------------------------------
HEURISTIC VALUES : {'A': 12, 'B': 8, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 8, 'H': 7, 'I': 7, 'J': 1}
SOLUTION GRAPH : {}
PROCESSING NODE : I
-----------------------------------------------------------------------------------------
HEURISTIC VALUES : {'A': 12, 'B': 8, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 8, 'H': 7, 'I': 0, 'J': 1}
SOLUTION GRAPH : {'I': []}
PROCESSING NODE : G
-----------------------------------------------------------------------------------------
HEURISTIC VALUES : {'A': 12, 'B': 8, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 1, 'H': 7, 'I': 0, 'J': 1}
SOLUTION GRAPH : {'I': [], 'G': ['I']}
PROCESSING NODE : B
-----------------------------------------------------------------------------------------
HEURISTIC VALUES : {'A': 12, 'B': 2, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 1, 'H': 7, 'I': 0, 'J': 1}
SOLUTION GRAPH : {'I': [], 'G': ['I'], 'B': ['G']}
PROCESSING NODE : A
-----------------------------------------------------------------------------------------
HEURISTIC VALUES : {'A': 6, 'B': 2, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 1, 'H': 7, 'I': 0, 'J': 1}
SOLUTION GRAPH : {'I': [], 'G': ['I'], 'B': ['G']}
PROCESSING NODE : C
-----------------------------------------------------------------------------------------
HEURISTIC VALUES : {'A': 6, 'B': 2, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 1, 'H': 7, 'I': 0, 'J': 1}
SOLUTION GRAPH : {'I': [], 'G': ['I'], 'B': ['G']}
PROCESSING NODE : A
-----------------------------------------------------------------------------------------
HEURISTIC VALUES : {'A': 6, 'B': 2, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 1, 'H': 7, 'I': 0, 'J': 1}
SOLUTION GRAPH : {'I': [], 'G': ['I'], 'B': ['G']}
PROCESSING NODE : J
-----------------------------------------------------------------------------------------
HEURISTIC VALUES : {'A': 6, 'B': 2, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 1, 'H': 7, 'I': 0, 'J': 0}
SOLUTION GRAPH : {'I': [], 'G': ['I'], 'B': ['G'], 'J': []}
PROCESSING NODE : C
-----------------------------------------------------------------------------------------
HEURISTIC VALUES : {'A': 6, 'B': 2, 'C': 1, 'D': 12, 'E': 2, 'F': 1, 'G': 1, 'H': 7, 'I': 0, 'J': 0}
SOLUTION GRAPH : {'I': [], 'G': ['I'], 'B': ['G'], 'J': [], 'C': ['J']}
PROCESSING NODE : A
-----------------------------------------------------------------------------------------
FOR GRAPH SOLUTION, TRAVERSE THE GRAPH FROM THE START NODE: A
------------------------------------------------------------
{'I': [], 'G': ['I'], 'B': ['G'], 'J': [], 'C': ['J'], 'A': ['B', 'C']}
------------------------------------------------------------`,
	},
	"3": {
		"id": "3",
		"repo": "https://github.com/ninjaasmoke/aiml-progs/tree/main/Candidate",
		"name": "Candidate Elimination",
		"code":
			`import numpy as np
import pandas as pd

data = pd.read_csv('candi.csv')
			
concepts = np.array(data.iloc[:,0:-1])
targets = np.array(data.iloc[:,-1])
			
def learn(concepts, target):
	s = concepts[0].copy()
	g = [["?" for i in range(len(s))] for i in range(len(s))]
				
	print(f'Specific: {s}\\nGeneral: {g}\\n\\n')
				
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

print(f'Specific: {s}\\nGeneral: {g}')`,
		"output": `Specific: ['sunny' 'warm' 'normal' 'strong' 'warm' 'same']
General: [['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], 
['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], 
['?', '?', '?', '?', '?', '?']]


Specific: ['sunny' 'warm' '?' 'strong' '?' '?']
General: [['sunny', '?', '?', '?', '?', '?'], ['?', 'warm', '?', '?', '?', '?']]`,
	},
	"4": {
		"id": "4",
		"repo": "https://github.com/ninjaasmoke/aiml-progs/tree/main/ID3",
		"name": "ID3",
		"code":
			`import numpy as np
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
print (tree, end='\\n\\n')

test = testData.to_dict('records')[0]
print(test, '=>', id3(test,features,target,None))`,
		"output": `{'outlook': {'Overcast': 'Yes', 'Rain': {'wind': {'Strong': 'No', 'Weak': 'Yes'}}, 
'Sunny': {'temp': {'Cool': 'Yes', 'Hot': 'No', 'Mild': 'No'}}}}

{'outlook': 'Sunny', 'temp': 'Mild', 'humidity': 'Normal', 'wind': 'Strong', 'play': 'Yes'} => Yes`,
	},
	"5": {
		"id": "5",
		"repo": "https://github.com/ninjaasmoke/aiml-progs/tree/main/Backpropagation",
		"name": "Backpropagation",
		"code":
			`import numpy as np

X = np.array([[2,9], [3,6], [4,8]])
y = np.array([[92], [86], [84]])

X = X/np.amax(X, axis=0)
y = y/100

def sigmoid(x):
	return 1/(1 + np.exp(-x))

def sigmoid_grad(x):
	return x*(1 - x)

epoch = 1000
eta = 0.1

i_n = 2
h_n = 3
o_n = 1

wh = np.random.uniform(size=(i_n, h_n))
bh = np.random.uniform(size=(1, h_n))

wout = np.random.uniform(size=(h_n, o_n))
bout = np.random.uniform(size=(1,o_n))

for i in range(epoch):
	h_ip = np.dot(X, wh)
	h_act = sigmoid(h_ip)
	
	o_ip = np.dot(h_act, wout) + bout
	output = sigmoid(o_ip)
	
	Eo = y - output
	outgrad = sigmoid_grad(output)
	d_output = Eo* outgrad
	
	Eh = np.dot(d_output, wout.T)
	hiddengrad = sigmoid_grad(h_act)
	d_hidden = Eh* hiddengrad
	
	wout += np.dot(h_act.T, d_output) *eta
	wh += np.dot(X.T,d_hidden) *eta

print("Normal: ", X)
print("\\nActual: ", y)
print("\\nPred: ", p)`,
		"output": `Normal:  [0.5 1. ]
[0.75       0.66666667]
[1.         0.88888889]

Actual: 
[0.92]
[0.86]
[0.84]

Pred: 
[0.8721367]
[0.86917497]
[0.87797585]`
	},
	"6": {
		"id": "6",
		"repo": "https://github.com/ninjaasmoke/aiml-progs/tree/main/Naive%20Bayes",
		"name": "Naive Bayes Classifier",
		"code":
			`import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB

data = pd.read_csv('naivedata.csv')

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
		"repo": "https://github.com/ninjaasmoke/aiml-progs/tree/main/K%20means",
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
		"repo": "https://github.com/ninjaasmoke/aiml-progs/tree/main/KNN",
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
	},
	"9": {
		"id": "9",
		"repo": "https://github.com/ninjaasmoke/aiml-progs/tree/main/LocallyWeightedRegression",
		"name": "Locally Weighted Regression",
		"code": `from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def localWeigh(point, X, ymat, k):
	m, n = np.shape(X)
	weights = np.mat(np.eye(m))
	for i in range(m):
		diff = point - X[i]
		weights[i,i] = np.exp(diff*diff.T/(-2.0*k**2))
	W = (X.T * (weights*X)).I * (X.T*(weights*ymat.T))
	return W

def localWeightReg(X,ymat,k):
	m,n = np.shape(X)
	ypred = np.zeros(m)
	for i in range(m):
		ypred[i] = X[i] * localWeigh(X[i],X,ymat,k)
	return ypred

def plott(X,pred):
	sortIndex = X[:,1].argsort(0)
	xsort = X[sortIndex][:,0][:,1]
	ysort = pred[sortIndex]
	plt.scatter(x,y,color='green')
	plt.plot(xsort,ysort,color="red",linewidth=5)
	plt.xlabel('Total bill')
	plt.ylabel('Tips')
	plt.show()
	return

data = pd.read_csv('data10.csv')
x=data['total_bill']
y = data['tip']
xmat = np.mat(x)
ymat = np.mat(y)
size = np.shape(xmat)[1]
ones = np.mat(np.ones(size))
X=np.hstack((ones.T,xmat.T))
pred = localWeightReg(X,ymat,3)
plott(X,pred)`,
		"output": ``,
	},
}; 