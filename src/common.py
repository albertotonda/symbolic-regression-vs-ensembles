import datetime
import logging
import numpy as np
import os

from logging.handlers import RotatingFileHandler
from scipy.stats import t

def initialize_logging(path: str, log_name: str, date: bool = True) -> logging.Logger:
    """
    Function that initializes the logger
    """
    if date:
        log_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "-" + log_name
    log_name = os.path.join(path, log_name)

    # create log folder if it does not exists
    if not os.path.isdir(path):
        os.mkdir(path)

    # remove old logger if it exists
    if os.path.exists(log_name):
        os.remove(log_name)

    # create an additional logger
    logger = logging.getLogger(log_name)

    # format log file
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(levelname)s %(asctime)s] %(message)s",
                                  "%Y-%m-%d %H:%M:%S")

    # the 'RotatingFileHandler' object implements a log file that is automatically limited in size
    fh = RotatingFileHandler(log_name,
                             mode='a',
                             maxBytes=100*1024*1024,
                             backupCount=2,
                             encoding=None,
                             delay=0)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.info("Starting " + log_name + "!")

    return logger


def close_logging(logger: logging.Logger):
    """
    Function that properly closes the logger
    """
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)

# interestingly, these two functions below are part of the scikit-learn library,
# but they are implemented only as an example in the documentation, and not as actual
# functions in the library, so I cut/pasted them here from 
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_stats.html#id11

def corrected_std(differences, n_train, n_test):
    """
    Corrects standard deviation using Nadeau and Bengio's approach.

    Parameters
    ----------
    differences : ndarray of shape (n_samples,)
        Vector containing the differences in the score metrics of two models.
    n_train : int
        Number of samples in the training set.
    n_test : int
        Number of samples in the testing set.

    Returns
    -------
    corrected_std : float
        Variance-corrected standard deviation of the set of differences.
    """
    # kr = k times r, r times repeated k-fold crossvalidation,
    # kr equals the number of times the model was evaluated
    kr = len(differences)
    corrected_var = np.var(differences, ddof=1) * (1 / kr + n_test / n_train)
    corrected_std = np.sqrt(corrected_var)
    
    return corrected_std


def compute_corrected_ttest(differences, df, n_train, n_test):
    """
    Computes right-tailed paired t-test with corrected variance.

    Parameters
    ----------
    differences : array-like of shape (n_samples,)
        Vector containing the differences in the score metrics of two models.
    df : int
        Degrees of freedom.
    n_train : int
        Number of samples in the training set.
    n_test : int
        Number of samples in the testing set.

    Returns
    -------
    t_stat : float
        Variance-corrected t-statistic.
    p_val : float
        Variance-corrected p-value.
    """
    mean = np.mean(differences)
    std = corrected_std(differences, n_train, n_test)
    t_stat = mean / std
    p_val = t.sf(np.abs(t_stat), df)  # right-tailed t-test
    
    return t_stat, p_val


def holm_bonferroni(p_values, alpha=0.05):
    """
    Performs the Holm-Bonferroni correction on a list of p-values, implemented by Gemini.
    
    Parameters:
    p_values (list): List of uncorrected p-values.
    alpha (float): Total desired Family-Wise Error Rate (significance level).
    
    Returns:
    list: Boolean list where True means 'Reject Null Hypothesis' (Statistically Significant)
    """
    m = len(p_values)
    # Store original indices so we can return the results in the original order
    indexed_p = sorted(enumerate(p_values), key=lambda x: x[1])
    
    reject = [False] * m
    
    for rank, (original_index, p) in enumerate(indexed_p):
        # Rank is 0-indexed, so remaining hypotheses = m - rank
        remaining_hypotheses = m - rank
        current_threshold = alpha / remaining_hypotheses
        
        if p < current_threshold:
            reject[original_index] = True
        else:
            # Step-down procedure: stop the moment one hypothesis fails to reject
            break
            
    return reject