import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

#df = pd.read_csv('final_data_cardio_disease.csv')

data = np.genfromtxt('final_data_water_quality.csv', delimiter=',')

data = np.delete(data, 0, 0)

X = data[:,:9]
Y = data[:,-1]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)