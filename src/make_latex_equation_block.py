"""
This script generates a LaTeX equation block, reading a list of LaTeX files. It is necessary to perform
some modifications on the original expressions because very often they do not fit on a single line.
"""

import os
import re as regex
import sys

def my_split_latex_equation(latex_str : str, max_chars : int = 60) -> str :

    operators = ['\\times', '\\cdot', '\\pm', '+', '-', '*', '/']
    
    # Simple maps to find matching tokens for implied multiplication
    closing_tokens = [')', ']', '\\right)']
    opening_tokens = ['(', '[', '\\left(']

    i = 0
    n = len(latex_str)
    curly_brackets_depth = 0
    split_points = []

    # TODO let's go over the string a first time, transforming all ")(" multiplications into ") \cdot ("

    # this is too simple, it is not taking into account unsplittable parts, like 'frac' and so on
    while i < n :

        if latex_str[i] == '{' :
            curly_brackets_depth += 1
        if latex_str[i] == '}' :
            curly_brackets_depth -= 1

        # check if the current part of the string starts with an operator
        for op in operators :
            if latex_str[:i].endswith(op) and curly_brackets_depth == 0 :
                split_points.append((i-len(op), i))

        for ct in closing_tokens :
            if latex_str[:i].endswith(ct) and curly_brackets_depth == 0 :
                # look ahead for opening tokens immediately after
                for ot in opening_tokens :
                    if latex_str[i:].startswith(ot) :
                        split_points.append((i,i))

        # go to the next character
        i += 1
    
    # final split point is the end of the string
    split_points.append((i,i))

    # put everything together, with some debugging printouts
    print("Original equation: \"%s\"" % latex_str)
    print("Split points: %s" % str(split_points))
    i = 0
    split_index = 0
    composed_string = ""
    while i < n :
        current_piece = ""
        stop_and_add_current_piece = False
        while not stop_and_add_current_piece :
            if len(current_piece) < max_chars and split_index < len(split_points)-1 :
                current_piece += latex_str[split_points[split_index][0]:split_points[split_index+1][0]]
                i = split_points[split_index+1][0]
                split_index += 1
            else :
                stop_and_add_current_piece = True
        
        if composed_string != "" :
            composed_string += r'\\ &'     
        composed_string += current_piece
        print("- Composed string: \"%s\"" % composed_string)

    return composed_string


def split_latex_equation_v2(latex_str: str, max_chars: int = 60) -> str:
    """
    Splits a long LaTeX equation string into an 'align' environment.
    Detects implied multiplication between parentheses/brackets at depth 0 
    and allows splitting there using a \cdot operator.

    Written by Gemini.
    """
    # Clean up multi-line inputs or extra spaces
    latex_str = " ".join(latex_str.split())
    
    brace_depth = 0
    lr_depth = 0 # Track \left and \right
    split_points = []
    
    relations = ['\\approx', '\\propto', '\\neq', '\\le', '\\ge', '=', '<', '>]']
    operators = ['\\times', '\\cdot', '\\pm', '+', '-']
    
    # Simple maps to find matching tokens for implied multiplication
    closing_tokens = [')', ']', '\\right)']
    opening_tokens = ['(', '[', '\\left(']
    
    i = 0
    n = len(latex_str)
    while i < n:
        # 1. Look ahead for implicit multiplication at top-level depth 0
        if True : #brace_depth == 0 and lr_depth == 0: # let's try to ignore this for the moment
            # Check if current position starts a split between parentheses
            # e.g., ")(", "](", ")\left(", or \right)\left(
            match_found = False
            
            # Check if we are sitting right between a closing and an opening block
            for close_t in closing_tokens:
                if latex_str[:i].endswith(close_t):
                    for open_t in opening_tokens:
                        if latex_str.startswith(open_t, i):
                            # We found an implicit multiplication boundary!
                            split_points.append((i, i, 'implied_mult', '\\cdot'))
                            match_found = True
                            break
                if match_found:
                    break
        
        # 2. Track \left and \right depth
        if latex_str.startswith('\\left', i):
            lr_depth += 1
            i += 5
            continue
        elif latex_str.startswith('\\right', i):
            lr_depth -= 1
            i += 6
            continue
            
        # 3. Track regular brace depths
        char = latex_str[i]
        if char in '{[(':
            brace_depth += 1
            i += 1
        elif char in '}])':
            brace_depth -= 1
            i += 1
        elif brace_depth == 0 and lr_depth == 0:
            # Check for relations
            found_rel = False
            for rel in relations:
                if latex_str.startswith(rel, i):
                    split_points.append((i, i + len(rel), 'relation', rel))
                    i += len(rel)
                    found_rel = True
                    break
            if found_rel:
                continue
                
            # Check for explicit operators
            found_op = False
            for op in operators:
                if latex_str.startswith(op, i):
                    split_points.append((i, i + len(op), 'operator', op))
                    i += len(op)
                    found_op = True
                    break
            if found_op:
                continue
            
            i += 1
        else:
            i += 1

    # 4. Break the string into structural elements
    elements = []
    last_idx = 0
    for start, end, type_, text in split_points:
        if start > last_idx:
            txt = latex_str[last_idx:start].strip()
            if txt:
                elements.append((txt, 'text'))
        elements.append((text, type_))
        last_idx = end
    if last_idx < n:
        txt = latex_str[last_idx:].strip()
        if txt:
            elements.append((txt, 'text'))

    # 5. Assemble lines based on max_chars
    lines = []
    current_line = ""
    has_relation = False
    
    for idx, (text, elem_type) in enumerate(elements):
        if elem_type == 'relation':
            mod_text = f" & {text} "
            if len(current_line) + len(mod_text) > max_chars and current_line:
                lines.append(current_line.rstrip() + " \\\\")
                current_line = mod_text.lstrip()
            else:
                if current_line and not current_line.endswith(" "):
                    current_line += " "
                current_line += mod_text.lstrip()
            has_relation = True
            
        elif elem_type in ('operator', 'implied_mult'):
            # Grab next text element to see if the chunk fits
            next_text = ""
            if idx + 1 < len(elements) and elements[idx+1][1] == 'text':
                next_text = elements[idx+1][0]
                
            added_len = len(text) + len(next_text) + 4
            
            # If it's an implied multiplication, it only shows up if we actually split!
            if len(current_line) + added_len > max_chars and current_line:
                lines.append(current_line.rstrip() + " \\\\")
                prefix = "& \quad " if has_relation else "& "
                current_line = f"{prefix}{text} " # Keep \cdot on new line
            else:
                # If we DON'T split, we don't force a \cdot for implied_mult 
                # unless you want a dot everywhere. Let's keep it clean (no dot) if it fits.
                if elem_type == 'operator':
                    if current_line and not current_line.endswith(" "):
                        current_line += " "
                    current_line += f"{text} "
                else:
                    # Implied multiplication that fits on one line: just add space
                    if current_line and not current_line.endswith(" "):
                        current_line += " "
                
        else: # Regular text/macros
            if current_line and not current_line.endswith(" "):
                current_line += " "
            current_line += text
            
    if current_line:
        lines.append(current_line.rstrip())
        
    return "\n".join(lines)


if __name__ == "__main__" :

    # hard-coded values
    task_id = 361258
    split_large_equations = False

    results_folder = "./latex"
    results_filename = "equation_block_task_%d.tex" % task_id

    #source_folder = "results_server_hyperparameter_tuning_20260519"
    source_folder = "results_server_default_hyperparameters_20260519"
    source_filename_template = "PySRRegressor_validation_equation_task_%d_fold_([0-9]).tex" % task_id
    #source_filename_template = "PySRRegressor_default_equation_task_%d_fold_([0-9]).tex" % task_id

    # let's start by creating the folder
    os.makedirs(results_folder, exist_ok=True)

    # now, let's list all the files in the folder
    equation_files = [f for f in os.listdir(source_folder) if regex.match(source_filename_template, f) is not None]
    print("Found %d equation files: %s" % (len(equation_files), str(equation_files)))

    # let's start building our LaTeX block
    latex_block = ""

    for eqf in equation_files :
        equation = ""
        with open(os.path.join(source_folder, eqf), "r") as fp :
            equation = fp.read()
        #print(equation)

        match = regex.match(source_filename_template, eqf)
        fold_id = int(match.group(1))
        print("Now analyzing equation for fold %d..." % fold_id)
        
        # let's start preparing the latex
        latex_equation = r'\begin{align}' + "\n" + r'\begin{split}' + "\n"
        # first: prepend "y ="
        latex_equation += r'y = & '

        # now, the most complex part: we need to split the equation if it is too long;
        # we can only use heuristics, so for example we can attempt to find
        if split_large_equations :
            latex_equation += split_latex_equation_v2(equation)
            #latex_equation += my_split_latex_equation(equation)
        else :
            latex_equation += equation

        latex_equation += r'\label{eq:' + str(task_id) + '-fold-' + str(fold_id) + '}' + "\n"
        latex_equation += r'\end{split}' + "\n" + r'\end{align}' + "\n" + "\n"

        latex_block += latex_equation

        # TODO comment this, it's for debugging
        #sys.exit(0)

    # save the final result
    with open(os.path.join(results_folder, results_filename), "w") as fp :
        fp.write(latex_block)