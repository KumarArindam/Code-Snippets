import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import logging

from multiprocessing import cpu_count

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('mnist_rf.log')
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

mnist = fetch_openml('mnist_784')
X = mnist.data
y = mnist.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

# param_grid = {
#     'n_estimators': [10],
#     'max_depth': [5],
#     'min_samples_split': [10],
#     'min_samples_leaf': [1],
#     'max_features': ['sqrt']
# }

rf = RandomForestClassifier(random_state=42)

n_jobs = cpu_count()
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=n_jobs, verbose=10)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_score = grid_search.best_score_
logging.info(f'Best hyperparameters: {best_params}')
logging.info(f'Best score: {best_score}')

## Use the best parameters

rf = RandomForestClassifier(random_state=42, **grid_search.best_params_)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
logging.debug('Test accuracy: %.2f', accuracy)