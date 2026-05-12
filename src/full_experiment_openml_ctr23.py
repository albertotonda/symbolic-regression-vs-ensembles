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

def get_sorted_tasks_ids(results_folder, results_file_name) :
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

    # if a results file already exists, remove the tasks that are already completed
    # TODO scratch this, it will be managed fold by fold
    completed_task_ids = []
    results_file_path = os.path.join(results_folder, results_file_name)
    if os.path.exists(results_file_path) :
        results_df = pd.read_csv(results_file_path)
        completed_task_ids = results_df["task_id"].unique().tolist()
        task_ids = [tid for tid in task_ids if tid not in completed_task_ids]

    return task_ids, completed_task_ids

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

if __name__ == "__main__" :

    # hard-coded variables
    results_folder = "results_20260511/" # I am assuming that the working directory is the root of the repository
    results_file_name = "openml_ctr23_statistics.csv"
    
    random_seed = 42 # random seed
    val_set_ratio = 0.2 # percentage of the training set to use as validation    

    regressor_classes = [PySRRegressor, RandomForestRegressor, XGBRegressor]
    regressor_classes = [RandomForestRegressor, XGBRegressor] # faster, for debugging
    metrics = {'R2': r2_score, 'MSE': mean_squared_error, 'RMSE': root_mean_squared_error}

    hyperparameters = {
        'RandomForestRegressor': {'n_estimators' : 1000, 'random_state' : random_seed, 'n_jobs' : -1},
        'XGBRegressor': {'n_estimators' : 1000, 'random_state' : random_seed, 'n_jobs' : -1},
        'PySRRegressor': {
            'niterations' : 100, 
            'population_size' : 27,
            'binary_operators' : ["+", "-", "*", "/"],
            'unary_operators' : ["sin", "cos", "tan", "log", "exp"], 
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
    task_ids, completed_task_ids  = get_sorted_tasks_ids(results_folder, results_file_name)
    logger.info("Found " + str(len(completed_task_ids)) + " completed tasks: " + str(completed_task_ids))
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
                    + "\",  %d samples, %d features" % (X.shape[0], X.shape[1]))
        
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
                    # TODO
                    regressor = regressor_class(**hyperparameters[regressor_name])

                    # we need to measure the time spent on each fold
                    time_start = time.time()
                    regressor.fit(X_train, y_train)
                    time_on_fold = time.time() - time_start

                    # create the dictionary with the information for the row
                    fold_statistics = {
                        'task_id' : task_id, 
                        'dataset_name' : dataset.name, 
                        'target_name': dataset.default_target_attribute, 
                        'n_samples' : X.shape[0],
                        'n_features' : X.shape[1], 
                        'missing_data' : missing_data, 
                        'categorical_features' : categorical_features, 
                        'fold_id' : fold_id, 
                        'regressor_name' : regressor_name, 
                        'time_on_fold' : time_on_fold
                    }

                    y_test_pred = regressor.predict(X_test)
                    for metric_name, metric in metrics.items() :
                        fold_statistics[metric_name] = metric(y_test, y_test_pred)

                    # add the row at the end of the dataframe, and save it
                    df_statistics.loc[len(df_statistics)] = fold_statistics
                    df_statistics.to_csv(os.path.join(results_folder, results_file_name), index=False)


    # close logger
    logger.info("Experiment completed.")
    close_logging(logger)

    # end of the script