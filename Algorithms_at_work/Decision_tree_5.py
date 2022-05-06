import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

#df = pd.read_csv('final_data_cardio_disease.csv')

data = np.genfromtxt('final_data_cardio_disease_2.csv', delimiter=',')

data = np.delete(data, 0, 0)
data = np.delete(data, 0, 1)
data = np.delete(data, 0 ,1)
data = np.delete(data, 0 ,1)
data = np.delete(data, 0, 1)

        
X = data[:,:11]
Y = data[:,-1]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


decision_tree = DecisionTreeClassifier(ccp_alpha=0.001, class_weight=None, criterion="entropy", max_depth=5,
                                       max_features=9, min_samples_leaf=1, min_samples_split=2, 
                                       random_state=2, splitter='best')

# param_dict = {"criterion" : ["gini", "entropy"],
#               "max_depth": range(1,10)}

# grid = GridSearchCV(decision_tree, param_grid = param_dict, cv=10, verbose=1, n_jobs=1)

# grid.fit(x_train, y_train)
#-----------------------------------------------------------------------------
#Learning Curve
samples = []

for i in range(10):
    samples.append(94 * (i + 1))
    
accuracy = []


for i in range(10):
    
    x_sample = x_train[0:samples[i], :]
    y_sample = y_train[0:samples[i]]
    decision_tree.fit(x_sample, y_sample)
    y_predicted = decision_tree.predict(x_test)
    accuracy.append(accuracy_score(y_test, y_predicted))
    
plt.plot(samples, accuracy)
plt.grid()
plt.show()

#-----------------------------------------------------------------------------
#validation curve

maxDepth = []
for i in range(10):
    maxDepth.append(i)

minLeaf = np.arange(0.1,0.5,0.1)
max_features = []
for i in range(12):
    max_features.append(i)
score = []

ccp_alpha = []
for i in range(10):
    ccp_alpha.append(i**-i)
    
for i in range(len(ccp_alpha)):
    decision_tree = DecisionTreeClassifier(ccp_alpha=ccp_alpha[i], class_weight=None, criterion="entropy", max_depth=5,
                                       max_features=3, min_samples_leaf=1, min_samples_split=2, 
                                       random_state=2, splitter='best')
    score.append(cross_val_score(decision_tree, X, Y, cv=10, verbose=1, n_jobs=1).mean())

plt.plot(ccp_alpha, score, color='green', marker='+', markersize=5, linestyle='--', label='Validation Accuracy')
plt.title('Validation Curve')
plt.xlabel('ccp_alpha')
plt.ylabel('Model accuracy')
plt.grid()
plt.legend(loc='lower right')
plt.show()