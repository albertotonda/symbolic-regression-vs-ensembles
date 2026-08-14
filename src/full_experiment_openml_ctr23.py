"""
Code refactoring to run the full experiment on the OpenML CTR23 benchmark suite.

Improvements:
- hyperparameter tuning using Optuna for the tree-based models
- better logging
- resuming from incomplete per-fold task results

"""

import json
from pyexpat import model

import numpy as np
import openml
import optuna
import os
import pandas as pd
import pickle
import random
import sys
import time

from pysr import PySRRegressor

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor

from memory_profiler import memory_usage # measure memory usage

# local imports
from common import initialize_logging, close_logging

def train_model(model, X, y) :
    """
    This is just a utility function, to employ the memory profiler
    """
    model.fit(X, y)
    return

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
        # now, this could raise an exception, because PySR might generate
        # equations including divisions by zero or square roots of negative numbers;
        # so we need to catch the exception, and just to be sure we perform a prediction
        # also on the test set (not used), because the test set could raise the exception
        # even if the validation set does not
        try:
            y_test_pred = regressor.predict(X_test, i)
            y_val_pred = regressor.predict(X_val, i)
            r2_value = r2_score(y_val, regressor.predict(X_val, i))
        except Exception as e:
            logger.warning("Exception while predicting with equation index %d: %s" % (i, str(e)))
            r2_value = -np.inf # if there is an error, the equation is bad
        
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
    with open(os.path.join(results_folder, "PySRRegressor_default_equation_task_%d_fold_%d.tex" % (task_id, fold_id)), "w") as fp :
        fp.write(regressor.latex(default_equation_index))
    with open(os.path.join(results_folder, "PySRRegressor_validation_equation_task_%d_fold_%d.tex" % (task_id, fold_id)), "w") as fp :
        fp.write(regressor.latex(best_equation_index))

    return fold_statistics

def optuna_objective(trial, hyperparameters, regressor_class, X_train, y_train, X_val, y_val) :
    """
    Objective function for Optuna hyperparameter tuning of the regressors.
    """
    # convert hyperparameters dictionary to the format expected by Optuna
    params = {}
    for key, value in hyperparameters[regressor_class.__name__].items() :
        if isinstance(value, dict) and 'min' in value and 'max' in value :
            # if the value is a dictionary with 'min' and 'max' keys, we use suggest_int or suggest_float
            if isinstance(value['min'], int) and isinstance(value['max'], int) :
                params[key] = trial.suggest_int(key, value['min'], value['max'])
            elif isinstance(value['min'], float) and isinstance(value['max'], float) :
                params[key] = trial.suggest_float(key, value['min'], value['max'])
        elif isinstance(value, list) :
            # if the value is a list, we use suggest_categorical
            params[key] = trial.suggest_categorical(key, value)
        else :
            # otherwise, we keep the default value, it could be something like 'random_state' or 'n_jobs' that we want to keep fixed
            params[key] = value
    
    model = regressor_class(**params)

    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
    
    return mean_squared_error(y_val, y_val_pred)

if __name__ == "__main__" :

    # hard-coded variables
    #results_folder = "local_results/results_20260810_memory/" # I am assuming that the working directory is the root of the repository
    results_folder = "local_results/results_20260812_memory_hyperparameter_tuning/"
    results_file_name = "openml_ctr23_statistics.csv"
    
    random_seed = 42 # random seed
    val_set_ratio = 0.2 # percentage of the training set to use as validation
    perform_hyperparameter_tuning = True # whether to perform hyperparameter tuning for the tree-based models
    min_time_for_tuning = 0 # minimum time in seconds to perform hyperparameter tuning, if the time available for tuning is less than this value, we skip tuning and use default hyperparameters
    timeout_in_seconds = 1440 # timeout for the longer experiments, corresponding to 24 minutes

    regressor_classes = [PySRRegressor, RandomForestRegressor, XGBRegressor]
    #regressor_classes = [RandomForestRegressor, XGBRegressor] # faster, for debugging
    metrics = {'R2': r2_score, 'MSE': mean_squared_error, 'RMSE': root_mean_squared_error}

    # these are the default hyperparameters for the regressors
    default_hyperparameters = {
        'RandomForestRegressor': {
            #'n_estimators' : 5000, # comment this, used for a single run
            'random_state' : random_seed, 
            'n_jobs' : -1
            },
        'XGBRegressor': {
            #'n_estimators' : 5000, # comment this, used for a single run
            'random_state' : random_seed, 
            'n_jobs' : -1
            },
        'PySRRegressor': { # NOTE: default hyperparameters have been altered for last-minute experiments, comment lines below
            #'elementwise_loss' : "loss(x, y) = max(x, 1e-8) - y + y * log(y / max(x, 1e-8))", # comment, Poisson loss used for ordinal regression
            'binary_operators' : ["+", "-", "*", "/"], # comment this, used for a single run
            'unary_operators' : ["sin", "cos", "log", "exp"], # comment this, used for a single run
            #'maxsize' : 60, # comment this, used for a single run
            #'niterations' : 1000, # comment this, used for a single run
            'random_state' : random_seed, 
            'procs' : None, 
            'parallelism' : 'multiprocessing',
            'temp_equation_file' : True,
            }
    }

    # another dictionary of values, to perform hyperparameter tuning with Optuna
    tuning_hyperparameters = {
        'RandomForestRegressor': {
            'n_estimators' : {'min' : 100, 'max' : 5000}, 
            'max_depth' : {'min' : 3, 'max' : 30},
            'min_samples_split' : {'min' : 2, 'max' : 20},
            'min_samples_leaf' : {'min' : 1, 'max' : 20},
            'max_features' : {'min' : 0.1, 'max' : 1.0},
            'n_jobs': -1,
            'random_state' : random_seed,
            },
        'XGBRegressor': {
            'n_estimators' : {'min' : 100, 'max' : 5000}, 
            'max_depth' : {'min' : 3, 'max' : 30},
            'min_leaves' : { 'min' : 0, 'max' : 64 },
            'learning_rate' : {'min' : 0.001, 'max' : 0.3},
            'subsample' : {'min' : 0.5, 'max' : 1.0},
            'colsample_bytree' : {'min' : 0.5, 'max' : 1.0},
            'colsample_bylevel' : {'min' : 0.5, 'max' : 1.0},
            'min_child_weight' : {'min' : 1, 'max' : 20},
            'gamma' : {'min' : 1e-8, 'max' : 10.0},
            'reg_alpha' : {'min' : 1e-8, 'max' : 10.0},
            'reg_lambda' : {'min' : 1e-8, 'max' : 10.0},
            'booster' : ["gbtree", "dart"], # this can have a big impact on training time
            'grow_policy' : ["depthwise", "lossguide"], # this can also have a big impact on training time
            'bootstrap' : True,
            'n_jobs': -1,
            'random_state' : random_seed,
            },
        'PySRRegressor': {
            'niterations' : 1000, 
            'population_size' : 100,
            'maxsize' : 300,
            'binary_operators' : ["+", "-", "*", "/"],
            'unary_operators' : ["sin", "cos", "log", "exp"], 
            'temp_equation_file' : True,
            'random_state' : random_seed, 
            'procs' : None, 
            'parallelism' : 'multiprocessing',
            'timeout_in_seconds' : timeout_in_seconds
            }
    }

    # create output folder if it does not exist
    if not os.path.isdir(results_folder):
        os.mkdir(results_folder)
    
    # save configuration for reproducibility
    with open(os.path.join(results_folder, "config.json"), "w") as f:
        json.dump({
            "random_seed": random_seed,
            "val_set_ratio": val_set_ratio,
            "perform_hyperparameter_tuning": perform_hyperparameter_tuning,
            "regressor_classes": [cls.__name__ for cls in regressor_classes],
            "metrics": [m for m in metrics],
            "default_hyperparameters": default_hyperparameters,
            "tuning_hyperparameters": tuning_hyperparameters
        }, f, indent=4)

    # start logger
    logger = initialize_logging(results_folder, "full_experiment_openml_ctr23.log")
    logger.info("Starting full experiment on OpenML CTR23 benchmark suite...")

    # get the list of task IDs to process, excluding the ones that are already completed
    task_ids  = get_sorted_tasks_ids() # this stopped working, for some reason
    #task_ids = [361250] # debugging, just one task
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
                if len(df_statistics) > 0 and len(df_statistics[
                    (df_statistics["task_id"] == task_id) & 
                    (df_statistics["fold_id"] == fold_id) &
                    (df_statistics["regressor_name"].str.startswith(regressor_name)) # startswith because we have two rows for PySR
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
                    
                    # in any case, we start by using default hyperparameters, unless we are
                    # in the special case where we are running hyperparameter tuning and the
                    # regressor is PySRRegressor, in that case we go with large populations etc.
                    regressor = regressor_class(**default_hyperparameters[regressor_name])

                    if perform_hyperparameter_tuning and regressor_name == "PySRRegressor" :
                        regressor = regressor_class(**tuning_hyperparameters[regressor_name])
                        
                    # we need to measure the time spent on training
                    time_start = time.time()
                    #regressor.fit(X_train, y_train)
                    mem_samples = memory_usage((train_model, (regressor, X, y)), interval=0.1)
                    time_on_fold = time.time() - time_start
                    memory_usage_MB = sum(mem_samples) / len(mem_samples)
                    
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
                        'time_on_fold' : [time_on_fold],
                        'mean_memory_usage_MB' : [memory_usage_MB],
                        'peak_memory_usage_MB' : [max(mem_samples)],
                    }

                    # now, if the regressor is PySR, we need to select an equation on the validation set
                    # and compute the metrics on the test set using that equation
                    if regressor_name == "PySRRegressor" :

                        logger.info("Performing extra steps for PySRRegressor...")

                        fold_statistics = perform_pysr_extra_steps(task_id, fold_id, regressor, 
                            X_val, y_val, X_test, y_test, metrics, fold_statistics, results_folder, logger)

                    else : 
                        # otherwise, if the regressor is a tree-based model, we might need to
                        # perform hyperparameter tuning using Optuna
                        logger.info("Performing extra steps for other regressors...")

                        # run Optuna hyperparameter tuning, with a timeout equal to the time available for tuning
                        if perform_hyperparameter_tuning :
                            # first step: check mean time spent by PySRRegressor on folds of the same task,
                            # and compute the total time available for hyperparameter tuning
                            df_pysr = df_statistics[
                                (df_statistics["task_id"] == task_id) & (df_statistics["regressor_name"].str.startswith("PySRRegressor"))
                                ]
                            mean_time_pysr = df_pysr["time_on_fold"].mean()
                            logger.info("- Mean time spent by PySRRegressor on folds of the same task: %.2f seconds" % mean_time_pysr)

                            time_available_for_tuning = max(min_time_for_tuning, mean_time_pysr - time_on_fold)
                            logger.info("- Time available for tuning: %.2f seconds" % time_available_for_tuning)

                            study = optuna.create_study(direction="minimize")
                            # we use a lambda function to pass the additional arguments to the objective function
                            hypertuning_start_time = time.time()
                            mem_samples = memory_usage(            
                                study.optimize(
                                    lambda trial : optuna_objective(trial, tuning_hyperparameters, regressor_class, X_train, y_train, X_val, y_val), 
                                    timeout=time_available_for_tuning
                                    )
                            )

                            logger.info("- Best hyperparameters found: " + str(study.best_params))

                            # now we can train a new model with the best hyperparameters
                            best_hyperparameters = study.best_params
                            best_hyperparameters["random_state"] = random_seed
                            best_hyperparameters["n_jobs"] = -1

                            regressor_tuned = regressor_class(**best_hyperparameters)
                            regressor_tuned.fit(X_train, y_train)
                            
                            # evaluate the difference between base model and tuned model
                            y_val_pred_tuned = regressor_tuned.predict(X_val)
                            y_val_pred_default = regressor.predict(X_val)

                            r2_tuned = r2_score(y_val, y_val_pred_tuned)
                            r2_default = r2_score(y_val, y_val_pred_default)
                            time_on_fold_with_tuning = time.time() - hypertuning_start_time

                            logger.info("- R2 on validation set with default hyperparameters: %.4f" % r2_default)
                            logger.info("- R2 on validation set with tuned hyperparameters: %.4f" % r2_tuned)

                            fold_statistics['time_on_fold'][-1] += time_on_fold_with_tuning # update the time on fold to include the time spent on tuning
                            fold_statistics['mean_memory_usage_MB'][-1] = sum(mem_samples) / len(mem_samples)
                            fold_statistics['peak_memory_usage_MB'][-1] = max(mem_samples)

                            # check if the tuned model is better
                            if r2_tuned > r2_default :
                                logger.info("- Tuned model is better, using it to predict on the test set...")
                                y_test_pred = regressor_tuned.predict(X_test)
                            else :
                                logger.info("- Tuned model is not better, keeping the default model for the test set...")

                        # this part below is performed in any case for tree models, hyperparameter tuning or not
                        for metric_name, metric in metrics.items() :
                            fold_statistics[metric_name] = metric(y_test, y_test_pred)

                        # save predictions of the model on the test set, as a CSV file
                        df_pred = pd.DataFrame({'y_test' : y_test, 'y_test_pred' : y_test_pred})
                        df_pred.to_csv(
                            os.path.join(results_folder, "%s_task_%d_fold_%d.csv" % (regressor_name, task_id, fold_id)), 
                            index=False)

                    # this part is performed in any case for all regressors, after the extra steps for PySR and tree-based models;
                    # we save the regressor object as a pickle file
                    regressor_file_name = os.path.join(results_folder, "%s_task_%d_fold_%d.pkl" % (regressor_name, task_id, fold_id))
                    with open(regressor_file_name, 'wb') as f :
                        pickle.dump(regressor, f)

                    # add the row at the end of the dataframe, and save it
                    df_fold = pd.DataFrame.from_dict(fold_statistics)
                    df_statistics = pd.concat([df_statistics, df_fold], ignore_index=True)
                    df_statistics.to_csv(os.path.join(results_folder, results_file_name), index=False)


    # close logger
    logger.info("Experiment completed.")
    close_logging(logger)

    # end of the script