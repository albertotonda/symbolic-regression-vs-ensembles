# -*- coding: utf-8 -*-
"""

@author: Alberto
"""

import numpy as np
import openml
import os
import pandas as pd
import pickle
import random
import sys

from pysr import PySRRegressor

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor

if __name__ == "__main__" :
    
    # hard-coded variables
    random_seed = 42
    use_predefined_splits = True
    results_folder = "results_20260504/" # I am assuming that the working directory is the root of the repository
    results_file_name = "openml_ctr23_statistics.csv"
    regressor_classes = [PySRRegressor, RandomForestRegressor, XGBRegressor]
    #regressor_classes = [RandomForestRegressor, XGBRegressor] # faster, for debugging
    metrics = {'R2': r2_score, 'MSE': mean_squared_error, 'RMSE': root_mean_squared_error}

    # check if the results folder exists, if not create it
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # hyperparameters for the regressors
    hyperparameter_values = {
        'RandomForestRegressor': {'n_estimators' : 1000, 'random_state' : random_seed, 'n_jobs' : -1},
        'XGBRegressor': {'n_estimators' : 1000, 'random_state' : random_seed, 'n_jobs' : -1},
        'PySRRegressor': {'niterations' : 100, 'population_size' : 27,
                          'binary_operators' : ["+", "-", "*", "/"],
                          'unary_operators' : ["sin", "cos", "tan", "log", "exp"], 
                          'temp_equation_file' : True,
                          'random_state' : random_seed, 'procs' : None, 'parallelism' : 'multiprocessing'}
    }
    
    # load CTR23 regression benchmark suite
    suite = openml.study.get_suite(353)
    
    # get all task_ids, so that if there is a crash we can easily restart
    # by skipping the first results
    task_ids = [t for t in suite.tasks]

    # this is for DEBUGGING purposes, to run only a few tasks;
    # comment the line below for full runs
    #task_ids = [361244, 361618, 361619, 361269, 361261, 361243]
    
    # prepare data structure to store information
    statistics_dictionary = {'task_id' : [], 'dataset_name' : [], 'target_name': [], 'n_samples' : [],
                             'n_features' : [], 'missing_data' : [], 'categorical_features' : [],}

    # check if the results file already exists, if so, load it
    # if it exists, we can skip the tasks that are already in the file
    if os.path.exists(os.path.join(results_folder, results_file_name)) :
        df_statistics = pd.read_csv(os.path.join(results_folder, results_file_name))
        # convert the df_statistics dataframe into the statistics_dictionary
        statistics_dictionary = df_statistics.to_dict(orient='list')
        # now we can skip the tasks that are already in the filec
        task_ids = [t for t in task_ids if t not in df_statistics['task_id'].values]
        print("Found existing results file, skipping %d tasks." % (len(df_statistics)))
    else :
        print("No existing results file found, starting from scratch.")

    for metric in metrics.keys() :
        for regressor_class in regressor_classes :
            # create two separate entries for PySR, one for the default equation
            # and one for the best equation identified with the validation set
            if regressor_class == PySRRegressor :
                statistics_dictionary[metric + '_' + regressor_class.__name__ + '_default'] = []
                statistics_dictionary[metric + '_' + regressor_class.__name__ + '_validation'] = []
            else :
                statistics_dictionary[metric + '_' + regressor_class.__name__] = []
    
    for task_id in task_ids :
        
        print("Now working on task %d..." % task_id)
        task = openml.tasks.get_task(task_id, download_splits=True,
                                     download_data=True, download_qualities=True,
                                     download_features_meta_data=True)
        
        # the 'task' object above contains a lot of useful information,
        # like the name of the target variable and the id of the dataset
        df_X, df_y = task.get_X_and_y('dataframe')
        
        # check if there is any missing value
        # here below there is a sum().sum() because it is adding up missing values
        # in rows AND THEN in columns
        missing_data = df_X.isnull().sum().sum() + df_y.isnull().sum()
        
        if missing_data > 0 :
            # we actually have to go with a task/dataset-specific correction, I think,
            # as there are only two datasets with missing values
            if task_id == 361268 : # dataset fps_benchmark
                # this task has several columns with A LOT of missing data,
                # so we are just going to drop them
                df_X.dropna(axis=1, inplace=True)
            elif task_id == 361616 : # dataset Moneyball
                # again, a few columns with 800/1200 missing values, get dropped
                df_X.dropna(axis=1, inplace=True)
        
        # check if there are any categorical columns
        df_categorical = df_X.select_dtypes(include=['category', 'object'])
        categorical_features = df_categorical.shape[1]
        
        # convert categorical columns to numerical values
        for c in df_categorical.columns :
            df_X[c] = df_X[c].astype('category') # double-check that it is treated as a categorical column
            df_X[c] = df_X[c].cat.codes # replace values with category codes (automatically computed)
        
        X = df_X.values
        y = df_y.values
        
        # let's also get the name of the dataset
        dataset = task.get_dataset()
        print("Task %d is applied to data set \"%s\" (id=%d)" % (task_id, dataset.name, dataset.dataset_id))
        
        for regressor_class in regressor_classes :
            # mean performance of regressor 
            regressor_name = regressor_class.__name__

            # tweak the dictionary with the metrics if the regressor is PySRRegressor
            if regressor_class == PySRRegressor :
                # we explicitly initialize the dictionaries like this to avoid issues with pointers,
                # which already created some issues in the past
                metric_values = {regressor_name + '_validation' : {metric : [] for metric in metrics.keys()}, 
                                regressor_name + '_default' : {metric : [] for metric in metrics.keys()}}
            else :
                metric_values = {regressor_name : {metric : [] for metric in metrics.keys()}}

            for fold in range(0, 10) :
                print("Evaluating \"%s\" performance on fold %d..." % (regressor_name, fold))
                
                # each regressor has its own hyperparameters, read from the dictionary
                regressor = regressor_class(**hyperparameter_values[regressor_name])
                
                if use_predefined_splits :
                    # get splits for N-fold cross-validation
                    # NOTE: this ignores repetitions, for a few data sets the evaluation
                    # is something like 10x(10-fold cross-validation), and here we are only
                    # performing one
                    train_index, test_index = task.get_train_test_split_indices(fold=fold)
                else :
                    # otherwise, we go for a nice 50/50 split, just like the funny
                    # guys that use conformal predictors; we need to instantiate
                    # the object managing the cross-validation with a different random
                    # seed at each iteration, to avoid issues
                    cv_random_seed = random.randint(0, 10000)
                    kf = KFold(n_splits=2, shuffle=True, random_state=cv_random_seed)
                    folds = [(train_index, test_index) for train_index, test_index in kf.split(X, y)]
                    train_index, test_index = folds[0]
                
                # get training and test data for the current fold
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                X_val, y_val = None, None # most of the regressors do not use a validation set, but PySR will

                # if we are working with PySRRegressor, we need to set the validation data, 
                # as it is used to select the best model during training
                if isinstance(regressor, PySRRegressor) :
                    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=random_seed)
                
                # normalization (it should not impact performance at all, let's see)
                scaler_X = StandardScaler()
                scaler_y = StandardScaler()
                
                X_train = scaler_X.fit_transform(X_train)
                X_test = scaler_X.transform(X_test)
                y_train = scaler_y.fit_transform(y_train.reshape(-1,1)).ravel()
                y_test = scaler_y.transform(y_test.reshape(-1,1)).ravel()
                
                if X_val is not None and y_val is not None :
                    X_val = scaler_X.transform(X_val)
                    y_val = scaler_y.transform(y_val.reshape(-1,1)).ravel()

                # now, let's split the code here into two branches: one for PySRRegressor, which employs
                # the validation set to select the best model on the complexity/error Pareto front, and
                # one for the other regressors, which might employ the validation set to perform
                # hyperparameter tuning
                if isinstance(regressor, PySRRegressor) :
                
                    # train the regressor
                    regressor.fit(X_train, y_train)
                
                    # get predictions for the test set, using the default model selected by PySRRegressor
                    y_test_pred_default = regressor.predict(X_test)
                    
                    # now, we iterate over all equations in the Pareto front, to obtain two different pieces of
                    # information: the equation with the best R2 value on the validation set, and the index of
                    # the equation selected by PySRRegressor by default
                    default_equation_index = -1
                    best_equation_index = -1
                    best_r2 = -np.inf
                    
                    n_equations = regressor.equations_.shape[0]
                    
                    for i in range(n_equations) :
                        r2_value = r2_score(y_val, regressor.predict(X_val, i))
                        if r2_value > best_r2 :
                            best_r2 = r2_value
                            best_equation_index = i

                        if regressor.latex(i) == regressor.latex() :
                            default_equation_index = i
                    
                    y_test_pred_best = regressor.predict(X_test, best_equation_index)
                    print("Best equation index: %d, R2: %.2f" % (best_equation_index, best_r2))

                    # compute r2 metrics
                    r2_test_best = r2_score(y_test, y_test_pred_best)
                    r2_test_default = r2_score(y_test, y_test_pred_default)
                    
                    # also save equations to the results folder
                    regressor.equations_.to_csv(
                        os.path.join(results_folder, "pysr_equations_task_%d_fold_%d.csv" % (task_id, fold)), index=False
                        )
                    
                    # also save the best equation as a text file, for easier readability
                    best_equation = regressor.latex(best_equation_index)
                    with open(os.path.join(results_folder, "pysr_best_equation_task_%d_fold_%d.txt" % (task_id, fold)), 'w') as fp :
                        fp.write(best_equation)
                    
                    # also save the equation that PySR selects by default, for comparison
                    default_equation = regressor.latex()
                    with open(os.path.join(results_folder, "pysr_default_equation_task_%d_fold_%d.txt" % (task_id, fold)), 'w') as fp :
                        fp.write(default_equation)

                    # also save the indexes of the best and default equations, for easier comparison
                    dict_equation_indexes = {
                        'equation_index' : [best_equation_index, default_equation_index],
                        'equation_type' : ['best_validation', 'default'],
                        'equation' : [regressor.sympy(best_equation_index), regressor.sympy(default_equation_index)],
                        'r2_test' : [r2_test_best, r2_test_default]
                    }
                    df_equation_indexes = pd.DataFrame.from_dict(dict_equation_indexes)
                    df_equation_indexes.to_csv(os.path.join(results_folder, "pysr_equation_indexes_task_%d_fold_%d.csv" % (task_id, fold)), index=False)

                    # and now, save metrics for both the best and default equations
                    for regressor_type, pred in [("_validation", y_test_pred_best), ("_default", y_test_pred_default)] :
                        print("type:", regressor_type, "pred:", pred)
                        for metric_name, metric_function in metrics.items() :
                            metric_value = metric_function(y_test, pred)
                            metric_values[regressor_name + regressor_type][metric_name].append(metric_value)   

                    # rename the y_test_pred for later use
                    y_test_pred = y_test_pred_best # this is the one we will save in the CSV file, for consistency with the other regressors                 

                    # debug
                    print("metric_values:", metric_values)

                else :
                    # TODO hyperparameter tuning
                    # train the regressor
                    regressor.fit(X_train, y_train)
                    
                    # get predictions for the test set
                    y_test_pred = regressor.predict(X_test)

                    # store partial results for the current fold
                    for metric_name, metric_function in metrics.items() :
                        metric_value = metric_function(y_test, y_test_pred)
                        metric_values[regressor_name][metric_name].append(metric_value)

            # now, at this point we have everything related to the current fold;
            # so, let's be over-cautious and save the regressor to a file
            regressor_file_name = os.path.join(results_folder, 
                                                "%s_task_%d_fold_%d.pkl" % (regressor_name, task_id, fold))
            with open(regressor_file_name, 'wb') as f :
                pickle.dump(regressor, f)
            
            # let's also save the predictions in the y_pred array as a CSV file
            y_pred_file_name = os.path.join(results_folder, 
                                            "%s_task_%d_fold_%d.csv" % (regressor_name, task_id, fold))
            y_pred_dictionary = {'test_index' : test_index, task.target_name + "_pred" : y_test_pred, task.target_name + "_true" : y_test}
            pd.DataFrame.from_dict(y_pred_dictionary).to_csv(y_pred_file_name, index=False)

            # update the statistics dictionary with the mean and std of the metric values
            for regressor_stat_name in metric_values.keys() :
                for metric_name, metric_values_list in metric_values[regressor_stat_name].items() :
                    regressor_metric = np.array(metric_values_list)
                    statistics_dictionary[metric_name + '_' + regressor_stat_name].append(
                        "%.4f +/- %.4f" % (np.mean(regressor_metric), np.std(regressor_metric))
                    )
                #print("Mean %s for %s: %.2f, std: %.2f" % (metric_name, regressor_name, np.mean(regressor_metric), np.std(regressor_metric)))
            
        # what I am interested in knowing:
        # number of samples
        # number of features
        # name of the target
        # missing data?
        # categorical variables?
        # mean performance of random forest?
        statistics_dictionary['task_id'].append(task_id)
        statistics_dictionary['dataset_name'].append(dataset.name)
        statistics_dictionary['target_name'].append(task.target_name)
        statistics_dictionary['n_samples'].append('{:,}'.format(df_X.shape[0]))
        statistics_dictionary['n_features'].append('{:,}'.format(df_X.shape[1]))
        statistics_dictionary['missing_data'].append('{:,}'.format(missing_data))
        statistics_dictionary['categorical_features'].append(categorical_features)
        
        # check if the results folder exists, if not create it
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

        df_statistics = pd.DataFrame.from_dict(statistics_dictionary)
        df_statistics.to_csv(os.path.join(results_folder, results_file_name), index=False)
        
        #sys.exit(0)