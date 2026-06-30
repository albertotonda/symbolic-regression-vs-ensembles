# Symbolic Regression vs Tree-based Ensembles
The idea of this repository is to perform a proper, fair comparison of Symbolic Regression and tree-based ensembles. Such ensembles, like Random Forest and XGBoost, are commonly considered the state of the art for tabular regression. Our hypothesis is that, given enough computational resources, Symbolic Regression will be able to catch up and eventually perform on par with the tree-based ensembles.

## Run the experiment
From the root of the repository:
```
python full_experiment_openml_ctr23.py
```
This will take a long time. Future works will add a JSON configuration file to be read from command line.

## Publications
* (Submitted) Tonda A. and Lutton E. 2026. "Symbolic Regression can Perform on Par with Tree-based Ensembles", submitted to Artificial Evolution 2026