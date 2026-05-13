"""
Code refactoring to run the full experiment on the OpenML CTR23 benchmark suite.

Improvements:
- hyperparameter tuning using Optuna for the tree-based models
- better logging
- resuming from incomplete task results

TODO:
- maybe it's better to actually save the results after each fold (!), with an extra column in the statistics
for the fold id! Then, we can compute more complex statistics after the experiment is complete
"""

import logging
import numpy as np
import openml
import os
import pandas as pd
import pickle
import random
import sys
import time

from pysr import PySRRegressor

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor

# local imports
from common import initialize_logging, close_logging

def get_sorted_tasks_ids() :
    """
    Sort all the tasks by number of samples (number of samples ~ to speed),
    then if a results file already exists, remove the tasks that are already completed.
    """
    # get the list of task IDs to process
    suite = openml.study.get_suite(353) # CTR23 benchmark suite
    df_tasks = openml.tasks.list_tasks(task_id=suite.tasks, output_format='dataframe')
    sorted_tasks = df_tasks[['tid', 'NumberOfInstances']].sort_values(by='NumberOfInstances', ascending=True)
    print(sorted_tasks) # for debugging
    task_ids = sorted_tasks['tid'].tolist()

    return task_ids

def prepare_data_structures(results_folder, results_file_name, metrics) :
    """
    Prepare the data structures to store the experiment results, eventually reading
    the existing results file if it exists.
    """
    # prepare data structure to store information
    statistics_dictionary = {'task_id' : [], 'dataset_name' : [], 'target_name': [], 'n_samples' : [],
                             'n_features' : [], 'missing_data' : [], 'categorical_features' : [], 
                             'fold_id' : [], 'regressor_name' : [], 'time_on_fold' : []}
    
    for metric in list(metrics.keys()) :
        statistics_dictionary[metric] = []
            
    df_statistics = pd.DataFrame.from_dict(statistics_dictionary)

    # if a CSV results file already exists, overwrite the empty DataFrame
    if os.path.exists(os.path.join(results_folder, results_file_name)) :
        df_statistics = pd.read_csv(os.path.join(results_folder, results_file_name))

    return df_statistics

def get_task_clean_data_and_name(task_id) :
    """
    Get the data and the name of the dataset and target for a given task ID.
    """
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
        
    return X, y, dataset, task, missing_data, categorical_features

def perform_pysr_extra_steps(task_id, fold_id, regressor, X_val, y_val, X_test, y_test, metrics, 
                             fold_statistics, results_folder, logger) :
    """
    Perform extra steps specific to PySR, such as finding the best equation on the validation set
    """
    # search for the best equation on the validation set
    logger.info("- Searching for the best equation on the validation set...")
    n_equations = regressor.equations_.shape[0]
    default_equation_index = -1
    best_equation_index = -1
    best_r2 = -np.inf
    
    for i in range(n_equations) :
        r2_value = r2_score(y_val, regressor.predict(X_val, i))
        if r2_value > best_r2 :
            best_r2 = r2_value
            best_equation_index = i

        if regressor.latex(i) == regressor.latex() :
            default_equation_index = i
    
    logger.info("- Best equation index: %d, R2 on validation set: %.4f" % (best_equation_index, best_r2))
    
    y_test_pred_default = regressor.predict(X_test, default_equation_index)
    y_test_pred_best = regressor.predict(X_test, best_equation_index)

    # update the fold_statistics dictionary with the metrics for the default and best equations
    for key, val in fold_statistics.items() :
        fold_statistics[key] = val * 2 # duplicate the value, to have one row for the default equation and one row for the best equation
    
    fold_statistics['regressor_name'] = [fold_statistics['regressor_name'][0] + "_default", fold_statistics['regressor_name'][0] + "_validation"]
    for metric_name, metric in metrics.items() :
        for y_pred in [y_test_pred_default, y_test_pred_best] :
            if metric_name not in fold_statistics :
                fold_statistics[metric_name] = []
            fold_statistics[metric_name].append(metric(y_test, y_pred))

    # save predictions of the default and best equation on the test set, as a CSV file
    df_pred_default = pd.DataFrame({'y_test' : y_test, 'y_test_pred' : y_test_pred_default})
    df_pred_default.to_csv(os.path.join(results_folder, "PySRRegressor_default_task_%d_fold_%d.csv" % (task_id, fold_id)), index=False)
    df_pred_best = pd.DataFrame({'y_test' : y_test, 'y_test_pred' : y_test_pred_best})
    df_pred_best.to_csv(os.path.join(results_folder, "PySRRegressor_validation_task_%d_fold_%d.csv" % (task_id, fold_id)), index=False)

    # save LaTeX representation of all equations
    with open(os.path.join(results_folder, "equation_default_task_%d_fold_%d.tex" % (task_id, fold_id)), "w") as fp :
        fp.write(regressor.latex(default_equation_index))
    with open(os.path.join(results_folder, "equation_best_task_%d_fold_%d.tex" % (task_id, fold_id)), "w") as fp :
        fp.write(regressor.latex(best_equation_index))

    return fold_statistics

if __name__ == "__main__" :

    # hard-coded variables
    results_folder = "results_20260513/" # I am assuming that the working directory is the root of the repository
    results_file_name = "openml_ctr23_statistics.csv"
    
    random_seed = 42 # random seed
    val_set_ratio = 0.2 # percentage of the training set to use as validation
    perform_hyperparameter_tuning = False # whether to perform hyperparameter tuning for the tree-based models

    regressor_classes = [PySRRegressor, RandomForestRegressor, XGBRegressor]
    #regressor_classes = [RandomForestRegressor, XGBRegressor] # faster, for debugging
    metrics = {'R2': r2_score, 'MSE': mean_squared_error, 'RMSE': root_mean_squared_error}

    # these are the default hyperparameters for the regressors
    default_hyperparameters = {
        'RandomForestRegressor': {'random_state' : random_seed, 'n_jobs' : -1},
        'XGBRegressor': {'random_state' : random_seed, 'n_jobs' : -1},
        'PySRRegressor': {
            'temp_equation_file' : True,
            'random_state' : random_seed, 
            'procs' : None, 
            'parallelism' : 'multiprocessing'
            }
    }

    # another dictionary of values, to perform hyperparameter tuning with Optuna
    tuning_hyperparameters = {
        'RandomForestRegressor': {
            'n_estimators' : [100, 200, 300], 
            'max_depth' : [None, 10, 20]
            },
        'XGBRegressor': {
            'n_estimators' : [100, 200, 300], 
            'max_depth' : [None, 10, 20]
            },
        'PySRRegressor': {
            'niterations' : 1000, 
            'population_size' : 100,
            'binary_operators' : ["+", "-", "*", "/"],
            'unary_operators' : ["sin", "cos", "log", "exp"], 
            'temp_equation_file' : True,
            'random_state' : random_seed, 
            'procs' : None, 
            'parallelism' : 'multiprocessing'
            }
    }

    # create output folder if it does not exist
    if not os.path.isdir(results_folder):
        os.mkdir(results_folder)
    
    # save configuration for reproducibility
    # TODO convert the dictionaries to JSON files
    
    # start logger
    logger = initialize_logging(results_folder, "full_experiment_openml_ctr23.log")
    logger.info("Starting full experiment on OpenML CTR23 benchmark suite...")

    # get the list of task IDs to process, excluding the ones that are already completed
    task_ids  = get_sorted_tasks_ids()
    logger.info("Found " + str(len(task_ids)) + " tasks to process: " + str(task_ids))

    # prepare data structures, eventually reading the existing results file if it exists
    logger.info("Preparing data structures to store the results...")
    df_statistics = prepare_data_structures(results_folder, results_file_name, metrics)

    # start iterating over tasks
    for task_id in task_ids :
        # get the clean data (no missing values, categorical features converted to numerical)
        # and several other information about the dataset
        X, y, dataset, task, missing_data, categorical_features = get_task_clean_data_and_name(task_id)
        logger.info("Processing task " + str(task_id) + " with dataset \"" + dataset.name 
                    + "\", %d samples, %d features" % (X.shape[0], X.shape[1]))
        
        # now iterate over regressors
        for regressor_class in regressor_classes :
            regressor_name = regressor_class.__name__
            logger.info("Testing regressor \"%s\" on task %d (dataset \"%s\")" %
                        (regressor_name, task_id, dataset.name))

            # and over the folds
            for fold_id in range(0, 10) :
                
                logger.info("Task %d, regressor %s, fold %d..." % (task_id, regressor_name, fold_id))
                
                # if the information for the task/regressor/fold combo is already in the dataframe, skip it
                if len(df_statistics[
                    (df_statistics["task_id"] == task_id) & 
                    (df_statistics["fold_id"] == fold_id) &
                    (df_statistics["regressor_name"] == regressor_name)
                    ]) != 0 :
                    
                    logger.info("- Fold information found in the dataframe, skipping...")
                
                else :
                    # the information has not been computed yet, so we need to run the regressor

                    # get the predefined splits from OpenML-CTR23
                    train_index, test_index = task.get_train_test_split_indices(fold=fold_id)

                    # split the data, but also use a part of the training set as validation
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_set_ratio, random_state=random_seed) 

                    # now, depending on the regressor, we have two different behaviors;
                    # if the regressor is PySR, use the validation set to select an equation
                    # on the final complexity/error Pareto front;
                    # otherwise, use the validation set to perform hyperparameter tuning
                    # using Optuna
                    
                    # in any case, we start by using default hyperparameters
                    regressor = regressor_class(**default_hyperparameters[regressor_name])

                    # we need to measure the time spent on training
                    time_start = time.time()
                    regressor.fit(X_train, y_train)
                    time_on_fold = time.time() - time_start

                    # get predictions and compute metrics
                    y_test_pred = regressor.predict(X_test)

                    # create the dictionary with the information for the row(s)
                    fold_statistics = {
                        'task_id' : [task_id], 
                        'dataset_name' : [dataset.name], 
                        'target_name': [dataset.default_target_attribute], 
                        'n_samples' : ['{:,}'.format(X.shape[0])],
                        'n_features' : ['{:,}'.format(X.shape[1])], 
                        'missing_data' : ['{:,}'.format(missing_data)], 
                        'categorical_features' : [categorical_features], 
                        'fold_id' : [fold_id], 
                        'regressor_name' : [regressor_name], 
                        'time_on_fold' : [time_on_fold]
                    }

                    # now, if the regressor is PySR, we need to select an equation on the validation set
                    # and compute the metrics on the test set using that equation
                    if regressor_name == "PySRRegressor" :

                        logger.info("Performing extra steps for PySRRegressor...")

                        fold_statistics = perform_pysr_extra_steps(task_id, fold_id, regressor, 
                            X_val, y_val, X_test, y_test, metrics, fold_statistics, results_folder, logger)

                    elif perform_hyperparameter_tuning : 
                        # otherwise, if the regressor is a tree-based model, we need to perform hyperparameter tuning using Optuna,
                        # and then compute the metrics on the test set using the best hyperparameters
                        logger.info("Performing extra steps for other regressors...")

                        # first step: check mean time spent by PySRRegressor on folds of the same task,
                        # and compute the total time available for hyperparameter tuning
                        df_pysr = df_statistics[(df_statistics["task_id"] == task_id) & (df_statistics["regressor_name"] == "PySRRegressor")]
                        mean_time_pysr = df_pysr["time_on_fold"].mean()
                        logger.info("- Mean time spent by PySRRegressor on folds of the same task: %.2f seconds" % mean_time_pysr)

                        time_available_for_tuning = max(0, mean_time_pysr - time_on_fold)
                        logger.info("- Time available for tuning: %.2f seconds" % time_available_for_tuning)

                        # TODO run Optuna hyperparameter tuning, with a timeout equal to the time available for tuning

                        # TODO check performance of the best hyperparameters on the test set, and update the metrics in the dataframe

                        for metric_name, metric in metrics.items() :
                            fold_statistics[metric_name] = metric(y_test, y_test_pred)

                    # add the row at the end of the dataframe, and save it
                    df_fold = pd.DataFrame.from_dict(fold_statistics)
                    df_statistics = pd.concat([df_statistics, df_fold], ignore_index=True)
                    df_statistics.to_csv(os.path.join(results_folder, results_file_name), index=False)


    # close logger
    logger.info("Experiment completed.")
    close_logging(logger)

    # end of the script