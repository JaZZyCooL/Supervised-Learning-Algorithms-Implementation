import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import learning_curve
from sklearn.tree import DecisionTreeClassifier

data = np.genfromtxt('final_data_water_quality.csv', delimiter=',')

data = np.delete(data, 0, 0)

X = data[:,:9]
Y = data[:,-1]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

decision_tree = DecisionTreeClassifier(ccp_alpha=0.0, class_weight="balanced", criterion="gini", max_depth=15,
                                       max_features=9, random_state=42, splitter='best', min_samples_leaf=38)

ada = AdaBoostClassifier(base_estimator=(decision_tree), learning_rate=0.1, n_estimators=4, random_state=(42))

# param_dict = {'n_estimators':range(1,10),
#               'learning_rate' : np.arange(0.1,1.0, 0.1)
#               }

# grid = GridSearchCV(ada, param_grid = param_dict, cv=10, verbose=1, n_jobs=1)

# grid.fit(X, Y)

# print(grid.best_params_)
# base_estimator=(decision_tree - max_depth=10), learning_rate=0.1, n_estimators=1, random_state=(42) - learning curve 1
#ada = AdaBoostClassifier(base_estimator=(decision_tree), learning_rate=0.1, n_estimators=1, random_state=(42))

#-----------------------------------------------------------------------------
# Learning Curve
train_sizes, train_scores, test_scores = learning_curve(estimator=ada, X=X, y=Y,
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
    
# learning_rate = np.arange(0.1,1.1,0.1)

# n_estimators = range(1,10,1)
# score = []
# for i in range(len(n_estimators)):
#     ada = AdaBoostClassifier(base_estimator=(decision_tree), learning_rate=0.1, n_estimators=n_estimators[i], random_state=(42))
#     score.append(cross_val_score(ada, X, Y, cv=10, verbose=1, n_jobs=1).mean())

# plt.plot(n_estimators, score, color='green', marker='+', markersize=5, linestyle='--', label='Validation Accuracy')
# plt.title('Validation Curve')
# plt.xlabel('n_estimators')
# plt.ylabel('Model accuracy')
# plt.grid()
# plt.legend(loc='lower right')
# plt.show()
