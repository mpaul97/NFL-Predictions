import pandas as pd
import numpy as np
from numpy import array, average
import random
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression
from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectFpr

import pickle

from pprint import pprint

data = pd.read_csv("blue6CER_All-13-WonSpread.csv")

data1 = pd.read_csv("black6_All-13-WonSpread.csv")
names = data1['name']
oppAbbr = data1['opp_abbr']

iterations = 10
k = int(len(data.columns)*0.8)

for col in data.columns:
	if "Unnamed" in col:
		data.drop(columns=col, inplace=True)

for col in data1.columns:
	if "Unnamed" in col:
		data1.drop(columns=col, inplace=True)

col_drops = ['name', 'opp_abbr', 'wy']

data.drop(col_drops, axis=1, inplace=True)
data1.drop(col_drops, axis=1, inplace=True)

######### NOT LOOP ###################

# features = data.drop('won', 1)
# label = data['won']

# data1.drop(columns='points', inplace=True)

# ####################### VarianceThreshold ####################

# # sel = VarianceThreshold(threshold=(.8 + (1 - .8)))

# # sel.fit(X, y)

# ####################### KBest ####################

# sel = SelectKBest(f_regression, k=k)

# sel.fit(features, label)

# # # ######################################################

# # # ####################### Recursive ####################

# # # # sel = RFECV(RandomForestClassifier(), scoring='accuracy')

# # # # sel.fit(x_train, y_train)

# # # ######################################################

# cols = sel.get_support(indices=True)
# new_cols = features.iloc[:, cols]
# features = new_cols

# new_cols = data1.iloc[:, cols]
# x_test1 = new_cols

# x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=0.05)

# scaler = preprocessing.StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.fit_transform(x_test)

# ############## LogisitcRegression ###############

# log_model = LogisticRegression(max_iter=131, verbose=2, random_state=42)
# log_model.fit(x_train, y_train)
# y_pred_log = log_model.predict(x_test)
# print(metrics.accuracy_score(y_test, y_pred_log))

###################################

# x_test1 = scaler.fit_transform(x_test1)
# y_pred_log1 = log_model.predict(x_test1)
# # acc = metrics.accuracy_score(y_test1, y_pred_log1)

# df = pd.DataFrame()

# # df['acc'] = acc
# df['name'] = names
# df['opp'] = oppAbbr
# df['won'] = list(y_pred_log1)

# df.to_csv("yellow7Won_Log1.csv")

############# GridSearchCV ###############

# # Create the parameter grid based on the results of grid search 
# # Penalty type 
# penalty = ['l1', 'l2']
# # Solver type 
# solver = ['lbfgs', 'liblinear']
# # Maximum number of iterations                 #80         120
# max_iter = [int(x) for x in np.linspace(start = 80, stop = 120, num = 5)]
# # Multi class 
# multi_class = ['auto']
# # Verbosity 
# verbose = [1]
# # l1 ratio 
# l1_ratio = [0, 0.8, 0.9, 1]
# # C
# C = [0.5, 1.0, 1.5]

# # Create the param grid
# param_grid = {'penalty': penalty, 'solver': solver, 'max_iter':max_iter, 
#     'multi_class':multi_class, 'verbose':verbose, 'l1_ratio':l1_ratio, 
#     'C':C
# }
# # pprint(param_grid)

# # Instantiate the grid search model with 2-fold cross-validation
# log_grid_search = GridSearchCV(estimator = LogisticRegression(random_state=42), param_grid = param_grid, cv = 2, n_jobs = -1)

# # Fit the grid search to the data
# log_grid_search.fit(x_train, y_train)
# best_log_grid = log_grid_search.best_estimator_
# best_log_grid.fit(x_train, y_train)
# y_pred_best_log = best_log_grid.predict(x_test)
# print(metrics.accuracy_score(y_test, y_pred_best_log))

# x_test1 = scaler.fit_transform(x_test1)
# y_pred_best_log1 = best_log_grid.predict(x_test1)
# # acc = metrics.accuracy_score(y_test1, y_pred_rfr_random1)

# df = pd.DataFrame()

# # df['acc'] = acc
# df['name'] = names
# df['opp'] = oppAbbr
# df['won'] = list(y_pred_best_log1)

# df.to_csv("WonProj_Grid.csv")

####################################

############# RandomSearchCV ###############

# # Number of trees in random forest
# n_estimators = [int(x) for x in np.linspace(start = 1, stop = 2000, num = 11)]
# # Number of features to consider at every split
# max_features = ['auto', 'sqrt', 'log2']
# # Maximum number of levels in tree       #  200        48
# max_depth = [int(x) for x in np.linspace(1, 100, num = 24)]
# max_depth.append(None)
# # Minimum number of samples required to split a node
# min_samples_split = [2, 5, 10]
# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 2, 4]
# # Method of selecting samples for training each tree
# bootstrap = [True, False]
# # Create the random grid
# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}
# # pprint(random_grid)

# # Use the random grid to search for best hyperparameters
# # First create the base model to tune
# rfc = RandomForestClassifier(random_state=42)
# # Random search of parameters, using 2-fold cross validation, 
# # search across 100 different combinations, and use all available cores
# rfc_random = RandomizedSearchCV(estimator = rfc, param_distributions = random_grid, n_iter = 100, cv = 2, verbose=2, n_jobs = -1)
# # Fit the random search model
# rfc_random.fit(x_train, y_train)
# y_pred_rfr_random = rfc_random.predict(x_test)
# print(metrics.accuracy_score(y_test, y_pred_rfr_random))

# ####################################
# # x_test1 = data1.drop("won", 1)
# # y_test1 = data1['won']

# x_test1 = scaler.fit_transform(x_test1)
# y_pred_rfr_random1 = rfc_random.predict(x_test1)
# acc = metrics.accuracy_score(y_test1, y_pred_rfr_random1)
# print(acc)

# df = pd.DataFrame()

# # df['acc'] = acc
# df['name'] = names
# df['opp'] = oppAbbr
# df['won'] = list(y_pred_rfr_random1)

# df.to_csv("white6R-13-Won_Forest.csv")

############## KNearestNeighbor ###############

# model = KNeighborsClassifier(n_neighbors=1000)

# model.fit(x_train, y_train)
# knn = model.predict(x_test)
# print(metrics.accuracy_score(y_test, knn))

########### LOOP ####################

df = pd.DataFrame()

df['name'] = names
df['opp'] = oppAbbr

predList = []
accList = []

for i in range(0, iterations):

	features = data.drop('won', 1)
	label = data['won']

	sel = SelectKBest(f_regression, k=k)

	sel.fit(features, label)

	cols = sel.get_support(indices=True)
	new_cols = features.iloc[:, cols]
	features = new_cols

	new_cols = data1.iloc[:, cols]
	x_test1 = new_cols

	###########################

	x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=0.05)

	scaler = preprocessing.StandardScaler()
	x_train = scaler.fit_transform(x_train)
	x_test = scaler.fit_transform(x_test)

	# LOG ####################################
	
	mi = random.randint(20, 100)
	rs = random.randint(21, 63)

	log_model = LogisticRegression(max_iter=mi, verbose=2, random_state=rs)
	log_model.fit(x_train, y_train)
	y_pred_log = log_model.predict(x_test)
	acc = metrics.accuracy_score(y_test, y_pred_log)
	accList.append(acc)

	x_test1 = scaler.fit_transform(x_test1)
	y_pred_log1 = log_model.predict(x_test1)

	predList.append(list(y_pred_log1))

	# FOREST ##################################

	# # Number of trees in random forest
	# n_estimators = [int(x) for x in np.linspace(start = 1, stop = 2000, num = 11)]
	# # Number of features to consider at every split
	# max_features = ['auto', 'sqrt', 'log2']
	# # Maximum number of levels in tree       #  200        48
	# max_depth = [int(x) for x in np.linspace(1, 100, num = 24)]
	# max_depth.append(None)
	# # Minimum number of samples required to split a node
	# min_samples_split = [2, 5, 10]
	# # Minimum number of samples required at each leaf node
	# min_samples_leaf = [1, 2, 4]
	# # Method of selecting samples for training each tree
	# bootstrap = [True, False]
	# # Create the random grid
	# random_grid = {'n_estimators': n_estimators,
	#                'max_features': max_features,
	#                'max_depth': max_depth,
	#                'min_samples_split': min_samples_split,
	#                'min_samples_leaf': min_samples_leaf,
	#                'bootstrap': bootstrap}
	# # pprint(random_grid)

	# # Use the random grid to search for best hyperparameters
	# # First create the base model to tune
	# rfc = RandomForestClassifier(random_state=42)
	# # Random search of parameters, using 2-fold cross validation, 
	# # search across 100 different combinations, and use all available cores
	# rfc_random = RandomizedSearchCV(estimator = rfc, param_distributions = random_grid, n_iter = 100, cv = 2, verbose=2, n_jobs = -1)
	# # Fit the random search model
	# rfc_random.fit(x_train, y_train)
	# y_pred_rfr_random = rfc_random.predict(x_test)
	# acc = metrics.accuracy_score(y_test, y_pred_rfr_random)
	# accList.append(acc)

	####################################
	# x_test1 = data1.drop("won", 1)
	# y_test1 = data1['won']

	# x_test1 = scaler.fit_transform(x_test1)
	# y_pred_rfr_random1 = rfc_random.predict(x_test1)

	# predList.append(list(y_pred_rfr_random1))

	# end for in range

print(sum(accList)/len(accList))

predArr = array(predList)
colAvg = average(predArr, axis=0)

df['won'] = colAvg

df.to_csv((str(iterations) + "_wonProj_Log-WonSpread.csv"), index=False)
