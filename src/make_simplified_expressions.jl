"""
Simplify expressions using the SymbolicUtils.jl, which should employ more sophisticated rules, like equality saturation graphs.
"""

using Pkg
packages = ["CondaPkg", "Latexify", "PythonCall", "SymbolicUtils"]
Pkg.add(packages)

# we need to set some environmental variable to tell PythonCall where to find the environment
#ENV["JULIA_PYTHONCALL_EXE"] = "@conda"      # Forces use of system Conda
#ENV["JULIA_PYTHONCALL_ENV"] = "base-311"  # Name of your conda environment

using PythonCall
using SymbolicUtils
using Latexify
using CondaPkg

CondaPkg.add("sympy") # this installs sympy in the local Python environment managed by PythonCall
CondaPkg.add("antlr4-python3-runtime", version="4.11.*") # and this installs the other Python package needed for LaTeX parsing
CondaPkg.add("pysr")

"""
    parse_latex_to_symbolicutils(latex_str::String)
Parses a LaTeX equation string into a `SymbolicUtils.BasicSymbolic` expression.
"""
function parse_latex_to_symbolicutils(latex_str::String)
    # 1. Load SymPy's LaTeX parsing module
    sympy_parsing = pyimport("sympy.parsing.latex")
    py_expr = sympy_parsing.parse_latex(latex_str)
    
    # 2. Convert Python formatting to a base Julia syntax string
    py_str = pyconvert(String, py_expr.__str__())
    julia_str = replace(py_str, "**" => "^")
    julia_str = replace(julia_str, r"\bE\b" => "exp(1)") 
    julia_str = replace(julia_str, r"\bI\b" => "im")     
    
    # --- FIX STEP A: Strip curly braces from the expression string ---
    julia_str = replace(julia_str, "{" => "", "}" => "")
    # -----------------------------------------------------------------

    # 3. Dynamically discover and declare free symbols
    for py_sym in py_expr.free_symbols
        raw_name = pyconvert(String, py_sym.name)
        
        # --- FIX STEP B: Strip curly braces from the registered variable names ---
        clean_name = replace(raw_name, "{" => "", "}" => "")
        var_name = Symbol(clean_name)
        
        Core.eval(@__MODULE__, :(@syms $var_name))
    end
    
    # 4. Parse and evaluate to a SymbolicUtils object tree
    julia_expr = Meta.parse(julia_str)
    return Core.eval(@__MODULE__, julia_expr)
end

# ok, let's try to do something more complex; we load the pickled PySRRegressor files,
# and directly access the SymPy expressions without having to bother with parsing

# hard-coded values
task_id = 361258
fold_id = 0
results_directory = "results_server_hyperparameter_tuning_20260519/"
pickled_pysr_file = results_directory * "PySRRegressor_task_$(task_id)_fold_$(fold_id).pkl"

# import the necessary Python packages
pickle = pyimport("pickle")

println("Reading \"" * pickled_pysr_file * "\"...")
pysrregressor = pickle.loads(read(pickled_pysr_file))

selected_expression_file_name = results_directory * "PySRRegressor_validation_equation_task_$(task_id)_fold_$(fold_id).tex"
println("Reading .tex file with the selected expression \"" * selected_expression_file_name * "\"...")
selected_latex = read(selected_expression_file_name, String)
println(selected_latex)

# we can iterate over the equations, looking for the one which was selected by the validation process
for (index, row) in enumerate(pysrregressor.equations_.itertuples())
    # convert fields to variable types usable in Julia
    comp = pyconvert(Int, row.complexity)
    loss = pyconvert(Float64, row.loss)

    # get the sympy expression
    println("Now analyzing the equation with complexity $comp and loss $loss...")

    # let's try to understand whether the equation is actually the same as the one in the file
    current_latex = pyconvert(String, pysrregressor.latex(index-1))
    if current_latex == selected_latex
        println("- Found it!")
    end

    sympy_expr = row.sympy_format

end


# ==============================================================================
# Pipeline Execution
# ==============================================================================
# 1. Define the path to your input text/TeX file
#file_path = "results_server_hyperparameter_tuning_20260519/PySRRegressor_validation_equation_task_361258_fold_0.tex"

# 2. Read the entire file content as a raw string
#latex_input = read(file_path, String)

# Input messy LaTeX string
#latex_input = raw"\frac{\sin^2(x) + \cos^2(x)}{\exp(y) \cdot \exp(-y)} + \log(x)"

#println("Processing expression...")
#sym_expr = parse_latex_to_symbolicutils(latex_input)

# Simplify using SymbolicUtils engine
#simplified_expr = SymbolicUtils.simplify(sym_expr)
#println("Simplified Expression: ", simplified_expr)

# ==============================================================================
# New Step: Convert Back to LaTeX and Save to File
# ==============================================================================

# 1. Convert the symbolic object to a LaTeX string using Latexify
# env=:raw strips out any outer equation or math block environments (e.g., \[ ... \])
#latex_output_str = latexify(simplified_expr, env=:raw)

# 2. Define target text file path
#output_filename = "simplified_equation.tex"

# 3. Open file and write the raw LaTeX string
#open(output_filename, "w") do file
#    write(file, latex_output_str)
#end

#println("Success! Saved LaTeX output to: $output_filename")
#println("Raw file contents: ", latex_output_str)
