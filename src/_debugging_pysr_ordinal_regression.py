"""
Separate small script to try and debug issues with ordinal regression in PySR.
"""
from pysr import PySRRegressor

if __name__ == "__main__" :
    
    loss_function = """
function eval_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)
    if !flag
        return L(Inf)
    end
    
    return sum((prediction .- dataset.y) .^ 2) / dataset.n
end
"""