#!/usr/bin/python

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

import grid_search_utils as utils

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, make_scorer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

X, y = utils.load_pts('data.csv')
plt.show()

#Fixing a random seed
random.seed(42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model (with default hyperparameters)
clf = DecisionTreeClassifier(random_state=42)

# Fit the model
clf.fit(X_train, y_train)

# Make predictions
train_predictions = clf.predict(X_train)
test_predictions = clf.predict(X_test)

utils.plot_model(X, y, clf)

print('The Training F1 Score is', f1_score(train_predictions, y_train))
print('The Testing F1 Score is', f1_score(test_predictions, y_test))


clf = DecisionTreeClassifier(random_state=42)

# TODO: Create the parameters list you wish to tune.
parameters = {'max_depth': [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20], 'min_samples_split' : [ 2, 4, 6, 8, 10, 12, 14, 16, 18, 20], 'min_samples_leaf':[2, 5, 10] }

# TODO: Make an fbeta_score scoring object.
scorer = make_scorer(f1_score)

# TODO: Perform grid search on the classifier using 'scorer' as the scoring method.
grid_obj = GridSearchCV( clf, parameters, scoring = scorer )

# TODO: Fit the grid search object to the training data and find the optimal parameters.
grid_fit = grid_obj.fit( X, y )

# TODO: Get the estimator.
best_clf = grid_fit.best_estimator_

# Fit the new model.
best_clf.fit(X_train, y_train)

# Make predictions using the new model.
best_train_predictions = best_clf.predict(X_train)
best_test_predictions = best_clf.predict(X_test)

# Calculate the f1_score of the new model.
print('The training F1 Score is', f1_score(best_train_predictions, y_train))
print('The testing F1 Score is', f1_score(best_test_predictions, y_test))

# Plot the new model.
utils.plot_model(X, y, best_clf)

# Let's also explore what parameters ended up being used in the new model.
best_clf
