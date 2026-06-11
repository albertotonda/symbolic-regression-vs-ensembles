"""
This script is an even deeper analysis of some of the equations found by PySR.
In this, I aim to load the PySRRegressors saved as pickles, and test each equation
on the test set, verifying the R2.
"""
import numpy as np
import os
import pandas as pd
import pickle
import re as regex

from sklearn.metrics import r2_score

# local imports
from analyze_equations import get_task_clean_data_and_name

if __name__ == "__main__" :

    # hard-coded variables
    output_folder = "results_analyze_equations_deeper/"
    results_folder = "results_server_hyperparameter_tuning_20260519/"
    task_id = 361258

    # pattern for the PySRRegressor pickle files
    pysr_pickle_pattern = "PySRRegressor_task_%d_fold_([0-9]+).pkl" % task_id

    # find all the files
    files = [f for f in os.listdir(results_folder) if regex.match(pysr_pickle_pattern, f)]
    print(files)

    # create output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    # load everything needed for the task
    df_X, df_y, dataset, task, missing_data, categorical_features = get_task_clean_data_and_name(task_id)
    print("Now working with task %d, dataset \"%s\"..." % (task_id, dataset.name))

    # get the data as numpy arrays
    X = df_X.values
    y = df_y.values

    for fold_id in range(0, 10) :
        print("- Now working with fold %d..." % fold_id)
        # start by getting the correct data split
        # get indices
        train_index, test_index = task.get_train_test_split_indices(fold=fold_id)

        # split the data
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # load the PySRRegressor for this fold
        pysr_pickle_file = os.path.join(results_folder, "PySRRegressor_task_%d_fold_%d.pkl" % (task_id, fold_id))
        with open(pysr_pickle_file, "rb") as f :
            pysr_regressor = pickle.load(f)

        # for each equation, get the R2 on the test set
        fold_stats = {'equation' : [], 'R2_test' : []}
        for equation_index, equation_row in pysr_regressor.equations_.iterrows() :
            # get the string representation of the equation
            #print(equation_row)
            equation_str = equation_row["equation"]
            
            # get the R2 on the test set
            try :
                y_pred = pysr_regressor.predict(X_test, index=equation_index)
                r2_value = r2_score(y_test, y_pred)
            except Exception as e :
                print("-- Error while evaluating equation %d: %s" % (equation_index, str(e)))
                r2_value = np.nan
            
            print("-- Equation %d: %s, R2 on test set: %.4f" % (equation_index, equation_str, r2_value))
            fold_stats['equation'].append(equation_str)
            fold_stats['R2_test'].append(r2_value)

        # save results as a CSV file 
        df_fold_stats = pd.DataFrame.from_dict(fold_stats)
        df_fold_stats.to_csv(os.path.join(output_folder, "task_%d_fold_%d_equation_stats.csv" % (task_id, fold_id)), index=False)