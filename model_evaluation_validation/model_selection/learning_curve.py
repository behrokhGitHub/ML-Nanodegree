#!/usr/bin/python
# Import, read, and split data
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

from sklearn.model_selection import learning_curve

import learning_curve_utils as utils

data = pd.read_csv('learning_curve_data.csv')
X = np.array(data[['x1', 'x2']])
y = np.array(data['y'])

# Fix random seed
np.random.seed(55)

# TODO: Uncomment one of the three classifiers, and hit "Test Run"
# to see the learning curve. Use these to answer the quiz below.

### Logistic Regression
estimator = LogisticRegression()

X2, y2 = utils.randomize(X, y)
num_trainings = 100
utils.draw_learning_curves(X2, y2, estimator, num_trainings)

### Decision Tree
#estimator = GradientBoostingClassifier()

### Support Vector Machine
#estimator = SVC(kernel='rbf', gamma=1000)
