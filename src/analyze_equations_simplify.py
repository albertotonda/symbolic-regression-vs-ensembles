"""
The idea here is to try and get a simplified version of the equations. We can employ sympy to perform a series of steps.
The steps have been taken from a Stack Overflow thread + recommendations from ChatGPT.
"""

import os
import re as regex
import sympy as sp

from sympy import cancel, factor, simplify, cse
from sympy.parsing.latex import parse_latex
from sympy.printing.latex import latex

# local imports
from analyze_equations import get_task_clean_data_and_name

if __name__ == "__main__" :

    # hard-coded values
    task_id = 361258
    results_folder = "results_server_hyperparameter_tuning_20260519/"
    equation_file_pattern = "PySRRegressor_validation_equation_task_%d_fold_([0-9]).tex" % task_id
    output_folder = "latex"
    output_filename = "simplified_equation_block_task_%d.tex" % task_id
    
    # let's start by creating the folder
    os.makedirs(output_folder, exist_ok=True)

    # now, let's list all the files in the folder
    equation_files = [f for f in os.listdir(results_folder) if regex.match(equation_file_pattern, f) is not None]
    print("Found %d equation files: %s" % (len(equation_files), str(equation_files)))

    # let's start building our LaTeX block
    latex_block = ""

    for eqf in equation_files :
        equation = ""
        with open(os.path.join(results_folder, eqf), "r") as fp :
            equation = fp.read()
        #print(equation)

        match = regex.match(equation_file_pattern, eqf)
        fold_id = int(match.group(1))
        print("Now analyzing equation for fold %d..." % fold_id)

        # transform the LaTeX text into a sympy expression; now, calling simplify() takes a LONG time (10 minutes, and it's not finished),
        # so I am changing the code a bit to perform simplifications without calling simplify()
        print("- parsing...")
        sympy_expr = parse_latex(equation)
        print("- ops:", sympy_expr.count_ops())    
        expr = sympy_expr.doit()
        print("- canceling...")
        expr = cancel(expr)
        print("- factoring...")
        expr = factor(expr)
        print("- collecting common sub-expressions...")
        subs, expr = cse(expr)
        print("- final expression: \"%s\"" % latex(expr))

        # prepare the LaTeX block
        latex_equation = r'\begin{align}' + "\n" + r'\begin{split}' + "\n"
        # first: prepend "y ="
        latex_equation += r'\hat{y} = & '
        # add the expression
        latex_equation += latex(expr) + "\n"
        # label the equation
        latex_equation += r'\label{eq:' + str(task_id) + 'simplified-fold-' + str(fold_id) + '}' + "\n"
        latex_equation += r'\end{split}' + "\n" + r'\end{align}' + "\n" + "\n"

        latex_block += latex_equation

    # save the final result
    with open(os.path.join(output_folder, output_filename), "w") as fp :
        fp.write(latex_block)