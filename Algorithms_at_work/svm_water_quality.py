import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
#df = pd.read_csv('final_data_cardio_disease.csv')

data = np.genfromtxt('final_data_water_quality.csv', delimiter=',')

data = np.delete(data, 0, 0)

X = data[:,:9]
Y = data[:,-1]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


svc = SVC(gamma='auto', probability=(False), C=1.0, kernel='rbf')

# # param_dict = {'C':np.arange(0.1,1.0,0.1),
# #               'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
# #               'gamma': ['scale', 'auto'],
# #               }

# # grid = GridSearchCV(svc, param_grid = param_dict, cv=10, verbose=1, n_jobs=1)

# # grid.fit(x_train, y_train)

# # print(grid.best_params_)

#-----------------------------------------------------------------------------
#Learning Curve
# train_sizes, train_scores, test_scores = learning_curve(estimator=svc, X=X, y=Y,
#                                                         cv=10, train_sizes=np.linspace(0.1, 1.0, 10),
#                                                       n_jobs=1)

# train_mean = np.mean(train_scores, axis=1)
# train_std = np.std(train_scores, axis=1)
# test_mean = np.mean(test_scores, axis=1)
# test_std = np.std(test_scores, axis=1)

# plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training Accuracy')
# plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
# plt.plot(train_sizes, test_mean, color='green', marker='+', markersize=5, linestyle='--', label='Validation Accuracy')
# plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
# plt.title('Learning Curve')
# plt.xlabel('Training Data Size')
# plt.ylabel('Model accuracy')
# plt.grid()
# plt.legend(loc='lower right')
# plt.show()

# #-----------------------------------------------------------------------------
#validation curve

score = []

C = np.arange(0.1,1.0,0.1)
    
kernel=['sigmoid', 'rbf', 'poly']

gamma=['auto', 'scale']

for i in range(len(kernel)):
    svc = SVC(gamma='auto', probability=(False), C=0.1, kernel=kernel[i], degree=9)
    score.append(cross_val_score(svc, X, Y, cv=10, verbose=1, n_jobs=1).mean())

plt.plot(kernel, score, color='green', marker='+', markersize=5, linestyle='--', label='Validation Accuracy')
plt.title('Validation Curve')
plt.xlabel('Kernel')
plt.ylabel('Model accuracy')
plt.grid()
plt.legend(loc='lower right')
plt.show()
