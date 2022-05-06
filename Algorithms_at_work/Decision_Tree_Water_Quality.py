import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_score
#df = pd.read_csv('final_data_cardio_disease.csv')

data = np.genfromtxt('final_data_water_quality.csv', delimiter=',')

data = np.delete(data, 0, 0)

X = data[:,:9]
Y = data[:,-1]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


decision_tree = DecisionTreeClassifier(ccp_alpha=0.0, class_weight="balanced", criterion="gini", max_depth=15,
                                       max_features=9, random_state=42, splitter='best', min_samples_leaf=38)

# param_dict = {"criterion" : ["gini", "entropy"],
#               "max_depth": range(1,100),
#               "min_samples_leaf" : np.arange(0.1,0.5,0.1),
#               "max_features" : range(1,10),
#               }

# grid = GridSearchCV(decision_tree, param_grid = param_dict, cv=10, verbose=1, n_jobs=1)

# grid.fit(x_train, y_train)

# print(grid.best_params_)

#ccp_alpha=0.0, class_weight="balanced", criterion="gini", max_depth=9
#max_features=9, random_state=42, splitter='best', min_samples_leaf=1
#------------------------------------------------------------------------------
#LEARNING CURVE
train_sizes, train_scores, test_scores = learning_curve(estimator=decision_tree, X=X, y=Y,
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
#Validation Curve
# maxDepth = []
# for i in range(100):
#     maxDepth.append(i)


# minLeaf = np.arange(0.1,0.5,0.1)

# max_features = []

# for i in range(10):
#     max_features.append(i)

# min_samples_leaf = []
# for i in range(100):
#     min_samples_leaf.append(i)
    
# score = []
# score2 = []

# for i in range(len(max_features)):
#     decision_tree = DecisionTreeClassifier(criterion="gini", class_weight="balanced", max_depth = 15, max_features=max_features[i], min_samples_leaf=38)
#     score.append(cross_val_score(decision_tree, X, Y, cv=10, verbose=1, n_jobs=1).mean())

# for i in range(len(max_features)):
#     decision_tree2 = DecisionTreeClassifier(ccp_alpha=0.0, class_weight="random", criterion="gini", max_depth=15,
#                                         max_features=max_features[i], min_samples_leaf=38,
#                                         random_state=42, splitter='best')
#     score2.append(cross_val_score(decision_tree, X, Y, cv=10, verbose=1, n_jobs=1).mean()) 
    
# plt.plot(max_features, score, color='green', marker='+', markersize=5, linestyle='--', label='Balanced Accuracy')
# plt.plot(max_features, score2, color='Orange', marker='+', markersize=5, linestyle='--', label='Random Accuracy')
# plt.title('Validation Curve')
# plt.xlabel('max_features')
# plt.ylabel('Model accuracy')
# plt.grid()
# plt.legend(loc='lower right')
# plt.show()
    

