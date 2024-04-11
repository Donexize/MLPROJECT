# packages

import importlib_resources
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import svd
from itertools import combinations
import xlrd
from matplotlib.patches import Ellipse
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import zscore

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from dtuimldmtools import rlr_validate
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_predict, KFold
from sklearn.neural_network import MLPRegressor
from dtuimldmtools import draw_neural_net, train_neural_net
import torch

import os
os.chdir(os.path.abspath(''))


filename = 'Raisin_Dataset.xls'
doc = xlrd.open_workbook(filename).sheet_by_index(0)


attributeNames = doc.row_values(0, 0, 7)

classLabels = doc.col_values(7, 1, 901)
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames, range(len(classNames))))

# Extract vector y, convert to NumPy array
y = np.asarray([classDict[value] for value in classLabels])

# Preallocate memory, then extract excel data to matrix X
X = np.empty((900, 7))
for i, col_id in enumerate(range(0, 7)):
    X[:, i] = np.asarray(doc.col_values(col_id, 1, 901))

# Compute values of N, M and C
N = len(y)
M = len(attributeNames)
C = len(classNames)

print(f'matrix size: {N} by {M}')
print(attributeNames)
print(classNames)


# eccentricity and first class
selected_feature = attributeNames[3]
y = X[:,3]
X = np.delete(X, 3, axis=1)
attributeNames.pop(3)

print('selected: ', selected_feature)

# regularization
mu = np.mean(X, 0)
sigma = np.std(X, 0)
X = (X - mu) / sigma

mu2 = np.mean(y)
sigma2 = np.std(y)
y = (y - mu2) / sigma2

# Add offset attribute
X = np.concatenate((np.ones((X.shape[0], 1)), X), 1)
attributeNames = ["Offset"] + attributeNames
M = M

print(f'matrix size: {N} by {M}')
print(attributeNames)

# using rlr_validate

# parameters
#lambdas = np.power(10.0, range(-5, 9))
lambdas = np.logspace(-3, 2, 50)
cross_validation = 10
XtX = X.T @ X
Xty = X.T @ y

(   opt_val_err,
    opt_lambda,
    mean_w_vs_lambda,
    train_err_vs_lambda,
    test_err_vs_lambda,
    ) = rlr_validate(X, y, lambdas, cross_validation)

lambdaI = opt_lambda * np.eye(M)
lambdaI[0, 0] = 0  # Do no regularize the bias term
w_rlr= np.linalg.solve(XtX + lambdaI, Xty).squeeze()
print(f'optimal lambda = {opt_lambda} with error = {opt_val_err}')

print("Weights:")
for m in range(M):
    print(f'\t{attributeNames[m].ljust(20)}: {w_rlr[m]:.5f}')

plt.figure()
plt.title("Optimal lambda: 1e{0}".format(np.log10(opt_lambda)))
plt.loglog(lambdas, train_err_vs_lambda.T, "b.-", label='train error')
plt.loglog(lambdas, test_err_vs_lambda.T, "r.-", label='test error')
plt.xlabel("Regularization factor")
plt.ylabel("Squared error (crossvalidation)")
plt.legend()
plt.grid()


plt.show()
