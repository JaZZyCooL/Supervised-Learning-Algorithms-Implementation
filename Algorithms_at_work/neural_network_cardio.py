import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier

data = np.genfromtxt('final_data_cardio_disease.csv', delimiter=',')

data = np.delete(data, 0, 0)
data = np.delete(data, 0, 1)

X = data[:,:11]
Y = data[:,-1]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

MLP = MLPClassifier(activation='logistic', alpha=0.0, hidden_layer_sizes=(9), max_iter=1000,random_state=42)

alpha = []
for i in range(9):
    alpha.append(10**-i)

print(alpha)

hidden_layers = []
for i in range(9):
    hidden_layers.append(((3),(i+1)),)

print(hidden_layers)

# param_dict = {"alpha":alpha,
#               "hidden_layer_sizes":hidden_layers,
#               "activation": ['relu', 'logistic', 'identity', 'tanh'],
#               "max_iter": [1000,5000,1000]
#               }

# grid = GridSearchCV(MLP, param_grid = param_dict, cv=10, verbose=1, n_jobs=1)

# grid.fit(x_train, y_train)

# print(grid.best_params_)
#activation='relu', alpha=0.01, hidden_layer_sizes=(10), max_iter=4000,random_state=42

#------------------------------------------------------------------------------
#LEARNING CURVE
# train_sizes, train_scores, test_scores = learning_curve(estimator=MLP, X=X, y=Y,
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
#-----------------------------------------------------------------------------
# #Validation Curve
# activation = ['relu', 'logistic', 'identity', 'tanh']
# score = []

# for i in range(len(hidden_layers)):
#     MLP = MLPClassifier(activation='logistic', alpha=0.0, hidden_layer_sizes= 3, max_iter=4000, random_state=42)
#     score.append(cross_val_score(MLP, X, Y, cv=10, verbose=1, n_jobs=1).mean())

# plt.plot(hidden_layers, score, color='green', marker='+', markersize=5, linestyle='--', label='Validation Accuracy')
# plt.title('Validation Curve')
# plt.xlabel('hidden_layers (2)')
# plt.ylabel('Model accuracy')
# plt.grid()
# plt.legend(loc='lower right')
# plt.show()

#------------------------------------------------------------------------------
#Iterative graph

# mlp = MLPClassifier(activation='logistic', alpha=0.0, hidden_layer_sizes=(9), max_iter=10000,random_state=42)

# """ Home-made mini-batch learning
#     -> not to be used in out-of-core setting!
# """
# N_TRAIN_SAMPLES = x_train.shape[0]
# N_EPOCHS = 25
# N_BATCH = 128
# N_CLASSES = np.unique(y_train)

# scores_train = []
# scores_test = []

# # EPOCH
# epoch = 0
# while epoch < N_EPOCHS:
#     print('epoch: ', epoch)
#     # SHUFFLING
#     random_perm = np.random.permutation(x_train.shape[0])
#     mini_batch_index = 0
#     while True:
#         # MINI-BATCH
#         indices = random_perm[mini_batch_index:mini_batch_index + N_BATCH]
#         mlp.partial_fit(x_train[indices], y_train[indices], classes=N_CLASSES)
#         mini_batch_index += N_BATCH

#         if mini_batch_index >= N_TRAIN_SAMPLES:
#             break

#     # SCORE TRAIN
#     scores_train.append(mlp.score(x_train, y_train))

#     # SCORE TEST
#     scores_test.append(mlp.score(x_test, y_test))

#     epoch += 1

# """ Plot """
# fig, ax = plt.subplots(2, sharex=True, sharey=True)
# ax[0].plot(scores_train)
# ax[0].set_title('Train')
# ax[1].plot(scores_test)
# ax[1].set_title('Test')
# fig.suptitle("Accuracy over epochs", fontsize=14)
# plt.show()