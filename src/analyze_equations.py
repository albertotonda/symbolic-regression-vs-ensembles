import cma
import numpy as np
import re as regex
import openml
import sympy as sp
import sys

from scipy.optimize import minimize
from sklearn.metrics import r2_score

# this function is a bit repeated, but it's slightly different from the one inside full_experiment_openml_ctr23.py
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
        
    return df_X, df_y, dataset, task, missing_data, categorical_features

def optimize_from_string(equation_str, X_data, y_data, initial_guess=None, method="Nelder-Mead", silent=False):
    """
    Parses an equation string, automatically identifies 'x' variables and 'c' constants,
    and optimizes the constants against a dataset.
    
    Parameters:
    -----------
    equation_str : str
        The mathematical equation as a string (e.g., "c1 * x4 + x5 * (x3 + c2 * x6)")
    X_data : numpy.ndarray
        2D array of independent variables. Columns MUST match the detected 'x' 
        variables in sorted order (e.g., Column 0 = x3, Column 1 = x4, etc.)
    y_data : numpy.ndarray
        1D array containing the target values.
    initial_guess : list or numpy.ndarray, optional
        Initial values for the 'c' parameters, ordered matching the sorted constants.
        
    Returns:
    --------
    scipy.optimize.OptimizeResult
        The optimization result object.
    list of sympy.Symbol
        The list of optimized constants in the order they map to the result array.
    """
    # 1. Automatically parse the string into a SymPy expression
    # SymPy automatically turns words into Symbol objects
    expr = sp.parse_expr(equation_str)
    
    # 2. Extract and categorize symbols automatically
    all_symbols = expr.free_symbols
    
    # Sort variables and constants alphabetically/numerically to ensure deterministic mapping
    variables = sorted([s for s in all_symbols if s.name.startswith('x')], key=lambda s: s.name)
    constants = sorted([s for s in all_symbols if s.name.startswith('c')], key=lambda s: s.name)
    
    # Validation checks
    if not constants:
        raise ValueError("No constants starting with 'c' were found in the expression.")
    if X_data.shape[1] != len(variables):
        raise ValueError(f"Dataset has {X_data.shape[1]} columns, but the equation requires "
                         f"{len(variables)} variables: {[v.name for v in variables]}")
        
    print("--- Automated Parsing Report ---")
    print(f"Expression parsed successfully: {expr}")
    print(f"Detected variables (mapped to X_data columns 0 to {len(variables)-1}): {[v.name for v in variables]}")
    print(f"Detected constants to optimize: {[c.name for c in constants]}\n")
    
    # 3. Vectorize the mathematical graph using lambdify
    # Ordering: constants first, then variables -> num_func(*constants, *variables)
    num_func = sp.lambdify((*constants, *variables), expr, modules='numpy')
    
    # 4. Define the objective function (Mean Squared Error)
    def objective_function(param_values):
        try:
            # X_data.T unpacks columns to match the ordered variable arguments
            y_pred = num_func(*param_values, *X_data.T)
            mse = np.mean((y_data - y_pred) ** 2)
            
            if np.isnan(mse) or np.isinf(mse):
                return 1e10
            return mse
        except Exception:
            return 1e10

    # 5. Set default initial guess if none provided
    if initial_guess is None:
        initial_guess = np.ones(len(constants))
    elif len(initial_guess) != len(constants):
        raise ValueError(f"Initial guess size ({len(initial_guess)}) doesn't match constants count ({len(constants)})")
        
    # 6. Run the optimization
    print("Optimizing parameters...")
    # instead of using Nelder-Mead, we can employ the good ol' CMA-ES
    if method != 'CMA-ES' :
        result = minimize(objective_function, initial_guess, method=method, options={'maxiter': 5000})
    else :
        x_best, result = cma.fmin2(objective_function, initial_guess, 1.0, options={'popsize' : 100, 'verbose' : -1})
    
    return result, constants

# ==========================================
# EXAMPLE USAGE
# ==========================================
if __name__ == "__main__":

    """
    The script originally managed one task at the time, now I want to modify it so that:
    1. it iterates over several tasks
    2. performs a 10-fold cross-validation with the pre-computed folds
    """

    tasks = {
        #361616 : "c1 * x4 + x5 * (c2 * x3 + c3 * x6) + c4",
        #361234 : "(c3 * x3 + c4 * x4 + c6 * x6 + cC)/(c5 * x5 + c71 * x7 + cA) + cB",
        #361622 : "c0 + c1 * (x1 - x14 + x16) + c2 * (x10 + x12 + x7) - (c3 * x0)/(c4 + c9 * x12 + c10 * x7) + c5 * exp(c6*x1 -x9*c7/sin(x2)) + c8*log(x0)",
        #361264 : "c0 * x4 + (c1 * x4)/(x1 - c2) + c3 * x3",
        #361272 : "c0*(x4 - c1)*(x4 - c2)*(x4 - c3)*(x4 - c4) + c5",
        #361257 : "(x0 - c0)*(c1*x3*(c2*x6 -c3*x2))", # this is using the validation set
        #361257 : "(c1*x0 + c2)*(x6 + c3)*(c4*x3*(c5*x2 + c6*x6 + c7) + c8)", # and this is using the default choice of PySR
        #361254 : "c0*x0 - x10 + c1*x14 + (c2*x14 + c3*x17)/x3 - c4*x7*x8 + c5",
        #361256 : "(c0*x15 - c1*x17) + c2/(c3*x2 + 1/x20 + c4)",
        #361617 : "c0*x4*(c1*x2 + c3*x6 + c4/(x0 - c5) + c6/(x7 - c7))",
        #361236 : " c3*x0 + c4*x2 + c5*x3 + c6*x4 + c7*x5 + c8 + sin(c9*x0 + c10)*sin(x0) + c1*exp(c11*x2*cos(x5) + c12*x3 + c13*x6) + (c14*x0 + c15*x4)/(c16*x3 + c17*x5 + c18*x6 - c2)",
        #361236 : "(c1*x0*x0 + c2*x0 + c3/(x0+c4) + c5*x4 + c6)*(c7*x3 -c8*x5 + c9/(c10-x5) + c11) + c12*x4 + c13",
        #361236 : "sin(x0-c0)*((x3*c1) + exp((x5*x5*c2 + x5*c3 + c4)*(c5/(x6 + c6)) + c7)) + c8",
        #361237 : "c0 + c1*x0 + c2*x1 + c3*x2 + c4*x4 + c5*x7 + (c6*x0 + c7*x1 + c8*x2)/(c9*x3 + c10) + c11*log(c12*x7 + c13) + c14*sin(c15*x1 + c16*x3) + c17*cos(c18*x1 + c19*x3)",
        #361237 : "c0 + (c1*x0 + c2*x1 + c3*x2)/(c4*x3 + c5) + c6/(x7 + c7)",
        #361237 : "c0 + (c1*x0 + c2*x1 + c3*x2)/(c4*x3 + c5 + c6*x7) + c8/(x7 + c9)",
        #361258 : "c0 + c1*x0 + c2*x2 + c3*sin(c4*(x5 + x6) + (c5*x3 + c6*x4 + c6*x7))",
        #361258 : "c0 + c2*x0 + c3*x2 + c4*x4 + c5*(x5 + x6 - x7) + c6*(x3 + c7 + (c8*x4 + c9)*(x5 + x6 - x7)) + c10*x5*x6 + c11*x4*x5",
        #361258 : "c0 + c1*x0 + c2*x2 + c3*(c4*x2 + (c5*x3 + c6)*(x5 + x6 + c7) + c8*x4 + c9*(x0 + x5*x6))",
        #361258 : "c0 + c1*x0 + c2*x2 + (c3*x3 + c4*x4 + c5*x7)*(x5 + x6)",
        #361258 : "c0 + c1*(c2*x0 + c3*x2 + c4*x4 + c5*(x5 + x6 - x7) + c6*(x3 + c7 + (c8*x4 + c9)*(x5+x6-x7)) + c10*x5*x6 + c11*x4*x5 + c12)",
        #361258 : "c0 + c1*x0 + c2*x2 + c3*x4 + c4*(c5*x3 + c6*x4 + c7)*(c8*cx5 + c9*x6 + c10*x7 + c11)",
        #361258 : "c0 - (-x2 + x4 + (x3 + x4*(x5 + x6 - x7)/c2 + c3)*(x5 + x6 + c4)*c5)*c6",
        #361258 : "c0 + c1*x0 + c2*sin(sin(c3 + c4*x0 + c5*x1 + c6*x2 + c7*x3 + c8*x4 + c9*x5 + c10*x6 + c11*x7 + c12*(x3 + x4)*x1))",
        #361258 : "c0*x0 + c1*sin(x2) + c2*sin(x0*x2 - x0 + c3*x1 + c4*x2*x3 + c5*x2 + c6*x4*x3*x3 + c7*x4*x3 + c8*x4 + c9*x1 + c10*x3 + c11*x4 + c12*x5 + c13*x6 + c14*x7 + c15*x4*(0.799 - x2) + c16) + c17",
        # warning: these two below can take A LOT of time
        361258 : "c33*c18*(c15*(c0*x7 + c18*x7 + c18*(c20 + c23 + c7*(c43 + x1 + cos(x3)) + x7 + (c42 + c44*x4)*(c34 + c6*(c18*x1 + x2) + x2*c12))) + c25*c35**c18*(x2 + x7)*(c10*x4*(c18*x1 + x6) + c41) + x4 + cos(x4) + cos(c8*x3*(c0*c18*x4 + c24 + c31*x4)))*cos(c21*(c13*(c0*x6 + c18*x7) + c18*x5)) + c40*(c27*(c22*(c18*x1 + c37 + x4) + (c28*(c18*c29*(x2 + x5) + c36) + cos(c11 + (c26 + x0)*x2*c5 + x1*c19))*x0*c4 + cos(c1 + x2)*cos(c14*c18*(c18*x7 + c32 + x6) + c16 + c18*x6*c9 + c38 + x3 + x5)) + c39 + x2*c3 + sin(c0*(c18*(x0*(c2) + x6*(c17)) + c30 + x5)))",
        #361258 : "c0*x1 + c1*x1*x2 + c11*sin(c18 + c20*x5 + c3*x0*(c5) + c3*x6*(c12)) + c15*x2 + c16*x1*x2 + c16*x1*x5 + c16*x10*x3 + c16*x10*x5 + c16*x2*x8 + c17*x4 + c19*x0 + c21*x3 + c23*x0*cos(c22*x9 + c9 + x0*x9 + x1*(c14)) + c23*cos(c4 + x2)*cos(c10*x7 + c13*6*(c8) + c24 + c26*x6 + x3 + x5) + c25*x5 + c28 + c29*x2*(c6) c43*(c0*x1 + c1*x2*x0*(c12) + c1*x5*x0*(c12) + c11*x2*x4*cos(c26*x5 + c27*x6 + c3*x7) + c14*sin(c25 + c29*x5 + c5*x0*(c8) + c5*x6*(c20)) + c15*cos(c26*x5 + c27*x6 + c3*x7) + c2*x1*cos(c26*x5 + c27*x6 + c3*x7) + c23*x4 + c24*x0*(c12) + c28*x2*cos(c26*x5 + c27*x6 + c3*x7) + c30*x0*(c12)*cos(c17 + c33*x2*(c13) + x0*x2*(c13) + x1*(c22)) + c30*cos(c7 + x2)*cos(c19*x7 + c21*x6*(c16) + c37 + c38*x6 + x3 + x5) + c31*x7*cos(c26*x5 + c27*x6 + c3*x7) + c32*x1*x4*cos(c26*x5 + c27*x6 + c3*x7) + c34 + c35*x2*(c10) + c36*x2*(c18)*cos(c26*x5 + c27*x6 + c3*x7) + c39*cos(x4)*cos(c26*x5 + c27*x6 + c3*x7) + c39*cos(c4*x3 + c40*x3*x4)*cos(c26*x5 + c27*x6 + c3*x7) + c41*x4*x2*(c18)*cos(c26*x5 + c27*x6 + c3*x7) + c42*x4*cos(c26*x5 + c27*x6 + c3*x7) + c6*x2*x4*(c21*x1 + x6)*cos(c26*x5 + c27*x6 + c3*x7) + c6*x7*x4*(c21*x1 + x6)*cos(c26*x5 + c27*x6 + c3*x7) + c9*cos(x3)*cos(c26*x5 + c27*x6 + c3*x7))+ c30*x2*x6 + c31*x4*x6 + c32*x1*x8 + c32*x4 + c32*x7*cos(x4) + c32*x7*cos(c2*x3 + c27*x3*x4) + c7*x2*cos(x3)",
    }
    method = 'CMA-ES'
    guess = None
    results = {}

    # Define your formula simply as a string!
    # No need to declare SymPy symbols beforehand.
    #my_equation_string = "c1 * x4 + x5 * (c2 * x3 + c3 * x6) + c4"
    #task_id = 361616
    #guess = [2.0, 10, 1000.0, 100] # initial guess for the constant values, in order
    #method = 'Nelder-Mead'

    #task_id = 361234
    #my_equation_string = "(c3 * x3 + c4 * x4 + c6 * x6 + cC)/(c5 * x5 + c71 * x7 + cA) + cB" # try removing x7 at denominator
    #guess = None
    #method = 'CMA-ES'

    #task_id = 361622
    #my_equation_string = "c0 + c1 * (x1 - x14 + x16) + c2 * (x10 + x12 + x7) - (c3 * x0)/(c4 + c9 * x12 + c10 * x7) + c5 * exp(c6*x1 -x9*c7/sin(x2)) + c8*log(x0)"
    #guess = None
    #method = 'CMA-ES'

    for task_id, my_equation_string in tasks.items() :
        
        # start preparing the results
        results[task_id] = {'r2_whole_dataset' : 0.0, 'mean_r2' : 0.0, 'std_r2' : 0.0, 'expression_latex' : ""}

        # load the appropriate dataset from the OpenML-CTR23 benchmark suite,
        # performing a bit of preprocessing
        df_X, df_y, dataset, task, missing_data, categorical_features = get_task_clean_data_and_name(task_id)
        print("Now working with task %d, dataset \"%s\"..." % (task_id, dataset.name))

        # another way of identifying variables, more reliable; convert equation to symbolic,
        # then extract free symbols as strings
        expr = sp.parse_expr(my_equation_string)
        all_symbols = expr.free_symbols
        features = sorted([str(s) for s in all_symbols if str(s).startswith('x')])
        #print("Features extracted by sympy:", features)

        # however, we need to first rename (!) all the columns in the dataset, with an index starting from 1 (...)
        new_column_names = ["x%d" % i for i in range(0, len(df_X.columns))]
        df_X.columns = new_column_names

        X = df_X[features].values
        y = df_y.values

        fit_result, optimized_constants = optimize_from_string(
            equation_str=my_equation_string,
            X_data=X,
            y_data=y,
            initial_guess=guess,
            method=method
        )
        
        # Print out results dynamically
        print("\n=== Optimization Results ===")
        best_x = None
        if method != 'CMA-ES' :
            print(f"Success: {fit_result.success}")
            print(f"Final MSE Loss: {fit_result.fun:.6f}")
            for param, value in zip(optimized_constants, fit_result.x) :
                print(f"Optimized {param.name}: {value:.4e}")
            best_x = fit_result.x
        else :
            best_x = fit_result.result.xbest
            print(f"Final MSE Loss: {fit_result.result.fbest:.6f}")

        # Last part: replace constant values inside the equation
        for param, value in zip(optimized_constants, best_x) :
            expr = expr.subs(param, value)
        # let's check a thing manually
        #expr = expr.subs("c1", 2.7)
        #expr = expr.subs("c2", 1000)

        print("Dataset \"%s\", expression with best constant values: %s" % (dataset.name, str(expr)))
        variables = sorted([s for s in all_symbols if s.name.startswith('x')], key=lambda s: s.name)
        lambdified_function = sp.lambdify(variables, expr, modules='numpy')
        y_pred = lambdified_function(*X.T)
        r2 = r2_score(y, y_pred)
        print("R2 score of the equation with constants optimized on the whole dataset: %.4f" % r2)
        results[task_id]['r2_whole_dataset'] = r2

        # small function stolen from stack overflow
        # Source - https://stackoverflow.com/a/48491897
        # Posted by user6655984, modified by community. See post 'Timeline' for change history
        # Retrieved 2026-05-20, License - CC BY-SA 3.0

        def round_expr(expr, num_digits=3):
            return expr.xreplace({n : round(n, num_digits) for n in expr.atoms(sp.Number)})

        rounded_equation_latex = sp.latex(round_expr(expr))
        print("Equation in LaTeX:", rounded_equation_latex)
        print("Original equation string:", sp.latex(expr))
        results[task_id]['expression_latex'] = rounded_equation_latex
        results[task_id]['expression_latex_extended'] = sp.latex(expr)

        print("Now performing a 10-fold cross-validation...")
        r2_values_cv = []
        for fold_id in range(0, 10) :
            # get indices
            train_index, test_index = task.get_train_test_split_indices(fold=fold_id)

            # split the data
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # re-fit the equation
            fit_result, optimized_constants = optimize_from_string(
                equation_str=my_equation_string,
                X_data=X_train,
                y_data=y_train,
                initial_guess=guess,
                method=method,
                silent=True
            )

            # compute R2 on the test set
            for param, value in zip(optimized_constants, best_x) :
                expr = expr.subs(param, value)
            variables = sorted([s for s in all_symbols if s.name.startswith('x')], key=lambda s: s.name)
            lambdified_function = sp.lambdify(variables, expr, modules='numpy')
            y_test_pred = lambdified_function(*X_test.T)
            r2_values_cv.append(r2_score(y_test, y_test_pred))

        mean_r2 = np.mean(r2_values_cv)
        std_r2 = np.std(r2_values_cv)
        print("R2 after a 10-fold cross-validation: %.4f +/- %.4f" % (mean_r2, std_r2))
        print("R2 values for each fold:", r2_values_cv)
        results[task_id]['mean_r2'] = mean_r2
        results[task_id]['std_r2'] = std_r2

    # this is just a test with a highly personalized equation
    #personalized_equation = "-2 * x1 + x4 + 3.523 + (x3 + (x4 + x7 -x5 -x6)/(x5 + 0.143)) * 5.90"
    #expr = sp.sympify(personalized_equation)
    #print("Personalized equation:", expr)
    #variables = sorted([s for s in expr.free_symbols if s.name.startswith('x')], key=lambda s: s.name)
    #print("Variables of the personalized equation:", variables)
    #lambdified_function = sp.lambdify(variables, expr, modules='numpy')
    #X = df_X[[str(v) for v in variables]].values
    #y_pred = lambdified_function(*X.T)
    #r2 = r2_score(y, y_pred)
    #print("My highly personalized equation has R2=%.4f" % r2)

    # report the whole results
    print("Final summary:")
    for task_id in results :
        print("Task %d:" % task_id)
        print("- R2(whole)=%.4f, R2(cv)=%.4f +/- %.4f" %
              (results[task_id]['r2_whole_dataset'], results[task_id]['mean_r2'], results[task_id]['std_r2']))  
        print("- Expression=\"%s\"" % results[task_id]['expression_latex'])
        print("- Expression (extended)=\"%s\"" % results[task_id]['expression_latex_extended'])