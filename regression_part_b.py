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

print('original')
print(f'matrix size: {N} by {M}')
print(attributeNames)
print(classNames)


# eccentricity and first class
selected_feature = attributeNames[3]
y = X[:,3]
X = np.delete(X, 3, axis=1)
attributeNames.pop(3)

print('selected: ', selected_feature)

# no regularization

# Add offset attribute
X = np.concatenate((np.ones((X.shape[0], 1)), X), 1)
attributeNames = ["Offset"] + attributeNames
M = M

print(f'matrix size: {N} by {M}')
print(attributeNames)

def compute_error_rate(y_true, y_pred):
    return np.sum(y_true - y_pred)**2 / len(y_true)


# modified very slow

folds=10

cv_model = model_selection.KFold(n_splits=folds, shuffle=True)
outer_results = []
lambdas = np.logspace(-3, 2, 50)
nodes_range = range(1, 12)
n_replicates = 1  # number of networks trained in each k-fold
max_iter = 1000


# GridSearchCV for Linear Regression and KNN at the same time
for i, (train_index, test_index) in enumerate(cv_model.split(X, y), start=1):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # linear regression
    (   opt_err_reg,
        opt_lambda,
        mean_w_vs_lambda,
        train_err_vs_lambda,
        test_err_vs_lambda,
        ) = rlr_validate(X_train, y_train, lambdas, folds)


    ## ANN
    ANN_error = []

    for n in nodes_range:
        print(f'outer fold {i}/{folds}')
        print(f'node iteration {n}/{len(nodes_range)}')

        model = lambda: torch.nn.Sequential(
            torch.nn.Linear(M, n),  # M features to n_hidden_units
            torch.nn.Tanh(),  # 1st transfer function,
            torch.nn.Linear(n, 1),  # n_hidden_units to 1 output neuron
            )
        loss_fn = torch.nn.MSELoss()  # notice how this is now a mean-squared-error loss

        error = []
        for k, (train_index_in, test_index_in) in enumerate(cv_model.split(X_train, y_train)):
            print(f'Crossvalidation fold: {k}/{folds}')

            X_train_in = torch.Tensor(X[train_index_in, :])
            y_train_in = torch.Tensor(y[train_index_in])
            X_test_in = torch.Tensor(X[test_index_in, :])
            y_test_in = torch.Tensor(y[test_index_in])

            net, final_loss, learning_curve = train_neural_net(
                model,
                loss_fn,
                X=X_train_in,
                y=y_train_in,
                n_replicates=n_replicates,
                max_iter=max_iter,
            )

            y_test_est = net(X_test_in)
            se = (y_test_est.float() - y_test_in.float()) ** 2  # squared error
            mse = (sum(se).type(torch.float) / len(y_test_in)).data.numpy()  # mean
            error.append(mse)  # store error rate for current CV fold

        ANN_error.append(np.mean(error))
    min_error_idx = np.argmin(ANN_error)
    min_error_ANN = ANN_error[min_error_idx]
    optimal_nodes = nodes_range[min_error_idx]

    #Evaluation Baseline
    y_pred_baseline = np.ones(len(y_test))*np.mean(y_train)
    error_rate_baseline = compute_error_rate(y_test, y_pred_baseline)
    
    #Results
    outer_results.append({
        'Outer_Fold': i,
        'ANN_Param': optimal_nodes,
        'ANN_Error': min_error_ANN * 100,
        'LogReg_lam_Param': opt_lambda,
        'LogReg_Error': opt_err_reg * 100,
        'Baseline_Error': error_rate_baseline * 100
    })

#Display
results_panda = pd.DataFrame(outer_results)
print(results_panda)


# LogReg vs ANN
t_lr_ann, p_lr_ann = stats.ttest_rel(results_panda['LogReg_Error'], results_panda['ANN_Error'])
cin_lr_ann = stats.t.interval(0.91, len(results_panda['LogReg_Error'])-1, loc=np.mean(results_panda['LogReg_Error']-results_panda['ANN_Error']), scale=stats.sem(results_panda['LogReg_Error']-results_panda['ANN_Error']))
print(f"LR - ANN: p = {p_lr_ann}, CI = {cin_lr_ann}")
# LogReg vs Baseline
t_lr_baseline, p_lr_baseline = stats.ttest_rel(results_panda['LogReg_Error'], results_panda['Baseline_Error'])
cin_lr_baseline = stats.t.interval(0.91, len(results_panda['LogReg_Error'])-1, loc=np.mean(results_panda['LogReg_Error']-results_panda['Baseline_Error']), scale=stats.sem(results_panda['LogReg_Error']-results_panda['Baseline_Error']))
print(f"LR - Baseline: p = {p_lr_baseline}, CI = {cin_lr_baseline}")
# ANN vs Baseline
t_ann_baseline, p_ann_baseline = stats.ttest_rel(results_panda['ANN_Error'], results_panda['Baseline_Error'])
cin_ann_baseline = stats.t.interval(0.91, len(results_panda['ANN_Error'])-1, loc=np.mean(results_panda['ANN_Error']-results_panda['Baseline_Error']), scale=stats.sem(results_panda['ANN_Error']-results_panda['Baseline_Error']))
print(f"ANN - Baseline: p = {p_ann_baseline}, CI = {cin_ann_baseline}")