import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
#df = pd.read_csv('final_data_cardio_disease.csv')

data = np.genfromtxt('final_data_cardio_disease.csv', delimiter=',')

data = np.delete(data, 0, 0)
data = np.delete(data, 0, 1)

X = data[:,:11]
Y = data[:,-1]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


knn = KNeighborsClassifier(n_neighbors=40, p=1, weights=('uniform'), algorithm = 'ball_tree')

param_dict = {'n_neighbors': range(1,10),
              'weights': ['uniform', 'distance'],
              'p': [1, 2],
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
              }

# grid = GridSearchCV(knn, param_grid = param_dict, cv=10, verbose=1, n_jobs=1)

# grid.fit(x_train, y_train)

# print(grid.best_params_)
#n_neighbors=8, p=2, weights=('uniform'), algorithm = 'ball_tree'
#n_neighbors=40, p=1, weights=('uniform'), algorithm = 'ball_tree'
#-----------------------------------------------------------------------------
#Learning Curve
train_sizes, train_scores, test_scores = learning_curve(estimator=knn, X=X, y=Y,
                                                        cv=10, train_sizes=np.linspace(0.1, 1.0, 10),
                                                      n_jobs=1)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training Accuracy')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean, color='green', marker='+', markersize=5, linestyle='--', label='Validation Accuracy')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
plt.title('Learning Curve')
plt.xlabel('Training Data Size')
plt.ylabel('Model accuracy')
plt.grid()
plt.legend(loc='lower right')
plt.show()

# #-----------------------------------------------------------------------------
#validation curve

# score = []

# n_neighbors = []
# for i in range(100):
#     n_neighbors.append(i)
    
# p = [1,2]
# weights = ['uniform', 'distance']
# algorithm = ['ball_tree', 'brute', 'kd_tree']

# for i in range(len(n_neighbors)):
#     knn = KNeighborsClassifier(n_neighbors=n_neighbors[i], p=2, weights='uniform', algorithm='ball_tree')
#     score.append(cross_val_score(knn, X, Y, cv=10, verbose=1, n_jobs=1).mean())

# plt.plot(n_neighbors, score, color='green', marker='+', markersize=5, linestyle='--', label='Validation Accuracy')
# plt.title('Validation Curve')
# plt.xlabel('n_neighbors')
# plt.ylabel('Model accuracy')
# plt.grid()
# plt.legend(loc='lower right')
# plt.show()