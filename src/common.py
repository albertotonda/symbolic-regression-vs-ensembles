import datetime
import logging
import numpy as np
import os
import pandas as pd

from itertools import combinations
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


def compute_corrected_ttest(differences, n_train, n_test, degrees_of_freedom=None):
    """
    Computes right-tailed paired t-test with corrected variance.

    Parameters
    ----------
    differences : array-like of shape (n_samples,)
        Vector containing the differences in the score metrics of two models.
    n_train : int
        Number of samples in the training set.
    n_test : int
        Number of samples in the testing set.
    degrees_of_freedom : int or None
        Degrees of freedom. If None, it is set to len(differences) - 1
    
        
    Returns
    -------
    t_stat : float
        Variance-corrected t-statistic.
    p_val : float
        Variance-corrected p-value.
    """
    if degrees_of_freedom is None :
        degrees_of_freedom = len(differences) - 1

    mean = np.mean(differences)
    std = corrected_std(differences, n_train, n_test)
    t_stat = mean / std
    p_val = t.sf(np.abs(t_stat), degrees_of_freedom)  # right-tailed t-test
    
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

# other two functions written by Gemini, to perform Nadeau-Bengio corrected TOST equivalence
def check_tost_equivalence(r2_list1, r2_list2, margin, alpha=0.05, cv_correction=True):
    """
    Performs a Two One-Sided Tests (TOST) for equivalence on two paired lists of R2 values.
    
    Parameters:
    - r2_list1, r2_list2: paired lists/arrays of R2 values of length k
    - margin: equivalence margin (Delta)
    - alpha: significance level (default 0.05)
    - cv_correction: bool, whether to apply Nadeau-Bengio correction (default True)
    """
    if len(r2_list1) != len(r2_list2):
        raise ValueError("The two lists must have the same size (k).")
        
    k = len(r2_list1)
    
    d = np.array(r2_list1) - np.array(r2_list2)
    mean_d = np.mean(d)
    sd = np.std(d, ddof=1)
    
    # Calculate Standard Error
    if cv_correction:
        # Standard k-fold CV ratio: n_test / n_train = 1 / (k - 1)
        se = np.sqrt((1 / k + 1 / (k - 1))) * sd
    else:
        se = sd / np.sqrt(k)
        
    if se == 0:
        is_equivalent = -margin < mean_d < margin
        return is_equivalent, 0.0 if is_equivalent else 1.0

    t1 = (mean_d - (-margin)) / se
    t2 = (mean_d - margin) / se
    df = k - 1
    
    p1 = t.sf(t1, df)  # Right-tail for H01: mean_d <= -margin
    p2 = t.cdf(t2, df) # Left-tail for H02: mean_d >= margin
    
    p_val = max(p1, p2)
    is_equivalent = p_val < alpha
    
    return is_equivalent, float(p_val)


def pairwise_tost_holm(methods_dict, margin, alpha=0.05, cv_correction=True):
    """
    Performs pairwise TOST equivalence tests across M methods with Holm-Bonferroni correction.
    """
    method_names = list(methods_dict.keys())
    pairs = list(combinations(method_names, 2))
    results = []
    raw_p_values = []
    
    for m1, m2 in pairs:
        r2_1 = methods_dict[m1]
        r2_2 = methods_dict[m2]
        
        mean_diff = np.mean(np.array(r2_1) - np.array(r2_2))
        
        _, p_val = check_tost_equivalence(
            r2_1, r2_2, 
            margin=margin, 
            alpha=alpha, 
            cv_correction=cv_correction
        )
        
        results.append({
            'Pair': f"{m1} vs {m2}",
            'Mean Diff (M1 - M2)': mean_diff,
            'p_raw': p_val
        })
        raw_p_values.append(p_val)
        
    # Holm-Bonferroni Step-Down Correction
    m = len(results)
    sorted_indices = sorted(range(m), key=lambda i: raw_p_values[i])
    
    adj_p_values = [0.0] * m
    running_max = 0.0
    
    for rank, idx in enumerate(sorted_indices):
        raw_p = raw_p_values[idx]
        multiplier = m - rank
        unadjusted_adj_p = raw_p * multiplier
        
        running_max = max(running_max, unadjusted_adj_p)
        adj_p_values[idx] = min(1.0, running_max)
        
    for i, res in enumerate(results):
        res['p_adjusted'] = adj_p_values[i]
        res['Is Equivalent'] = adj_p_values[i] < alpha

    return pd.DataFrame(results)[['Pair', 'Mean Diff (M1 - M2)', 'p_raw', 'p_adjusted', 'Is Equivalent']]

import numpy as np
import pandas as pd
from scipy.stats import t
from itertools import combinations

def compute_sbe(r2_list1, r2_list2, alpha=0.05, cv_correction=True):
    """
    Calculates the Smallest Bound of Equivalence (SBE) between two paired lists of R2 values.
    
    Parameters:
    - r2_list1, r2_list2: paired lists/arrays of R2 fold values (length k)
    - alpha: significance level (default 0.05)
    - cv_correction: bool, whether to apply Nadeau-Bengio correction (default True)
    
    Returns:
    - sbe: float, the minimum equivalence margin Delta needed to claim equivalence at level alpha.
    """
    if len(r2_list1) != len(r2_list2):
        raise ValueError("The two lists must have the same size (k).")
        
    k = len(r2_list1)
    if k < 2:
        raise ValueError("At least 2 folds are required.")
        
    d = np.array(r2_list1) - np.array(r2_list2)
    mean_d = np.mean(d)
    sd = np.std(d, ddof=1)
    
    # Calculate Standard Error
    if cv_correction:
        se = np.sqrt((1 / k + 1 / (k - 1))) * sd
    else:
        se = sd / np.sqrt(k)
        
    df = k - 1
    # Critical value for 1-alpha quantile of Student's t
    t_crit = t.ppf(1 - alpha, df)
    
    # Exact SBE formula
    sbe = np.abs(mean_d) + t_crit * se
    return float(sbe)


def pairwise_sbe(methods_dict, alpha=0.05, cv_correction=True, adjust_bonferroni=True):
    """
    Computes pairwise SBE across M methods.
    
    Parameters:
    - methods_dict: dict mapping method names to lists/arrays of R2 fold scores.
    - alpha: float, target significance level (default 0.05)
    - cv_correction: bool, apply Nadeau-Bengio CV correction
    - adjust_bonferroni: bool, adjust alpha for multiple comparisons across all pairs (alpha / m)
    
    Returns:
    - pandas.DataFrame containing mean differences and SBE values.
    """
    method_names = list(methods_dict.keys())
    pairs = list(combinations(method_names, 2))
    m = len(pairs)
    
    # Optional Bonferroni correction on alpha across m comparisons
    effective_alpha = (alpha / m) if adjust_bonferroni else alpha
    
    results = []
    for m1, m2 in pairs:
        r2_1 = methods_dict[m1]
        r2_2 = methods_dict[m2]
        
        mean_diff = np.mean(np.array(r2_1) - np.array(r2_2))
        sbe_val = compute_sbe(r2_1, r2_2, alpha=effective_alpha, cv_correction=cv_correction)
        
        results.append({
            'Pair': f"{m1} vs {m2}",
            'Mean Diff (M1 - M2)': mean_diff,
            'Abs Mean Diff': np.abs(mean_diff),
            f'SBE (alpha={effective_alpha:.4f})': sbe_val
        })
        
    return pd.DataFrame(results)

# other two functions written by Gemini; this time, we compute the Smallest Bound of Equivalence
# between different methods
def compute_sbe(r2_list1, r2_list2, alpha=0.05, cv_correction=True):
    """
    Calculates the Smallest Bound of Equivalence (SBE) between two paired lists of R2 values.
    """
    if len(r2_list1) != len(r2_list2):
        raise ValueError("The two lists must have the same size (k).")
        
    k = len(r2_list1)
    d = np.array(r2_list1) - np.array(r2_list2)
    mean_d = np.mean(d)
    sd = np.std(d, ddof=1)
    
    if cv_correction:
        se = np.sqrt((1 / k + 1 / (k - 1))) * sd
    else:
        se = sd / np.sqrt(k)
        
    df = k - 1
    t_crit = t.ppf(1 - alpha, df)
    
    sbe = np.abs(mean_d) + t_crit * se
    return float(sbe)


def pairwise_sbe_holm(methods_dict, alpha=0.05, cv_correction=True):
    """
    Computes pairwise Smallest Bound of Equivalence (SBE) across M methods 
    using a step-down Holm-Bonferroni adjustment.
    
    Parameters:
    - methods_dict: dict mapping method names to lists/arrays of R2 fold scores.
    - alpha: float, overall family-wise significance level (default 0.05)
    - cv_correction: bool, apply Nadeau-Bengio CV correction (default True)
    
    Returns:
    - pandas.DataFrame containing mean differences, raw SBE, and Holm-adjusted SBE.
    """
    method_names = list(methods_dict.keys())
    pairs = list(combinations(method_names, 2))
    m = len(pairs)
    
    # 1. Compute raw SBE at nominal alpha for all pairs
    raw_sbe_data = []
    for m1, m2 in pairs:
        r2_1 = methods_dict[m1]
        r2_2 = methods_dict[m2]
        mean_diff = np.mean(np.array(r2_1) - np.array(r2_2))
        sbe_raw = compute_sbe(r2_1, r2_2, alpha=alpha, cv_correction=cv_correction)
        
        raw_sbe_data.append({
            'Pair': f"{m1} vs {m2}",
            'r2_1': r2_1,
            'r2_2': r2_2,
            'Mean Diff': mean_diff,
            'SBE_raw': sbe_raw
        })
        
    # 2. Sort pairs by raw SBE (smallest SBE = rank 0, strongest equivalence)
    sorted_indices = sorted(range(m), key=lambda i: raw_sbe_data[i]['SBE_raw'])
    
    holm_sbe_results = [0.0] * m
    running_max_sbe = 0.0
    
    # 3. Apply Holm-Bonferroni step-down alpha allocation
    for rank, idx in enumerate(sorted_indices):
        item = raw_sbe_data[idx]
        alpha_j = alpha / (m - rank)  # Step-down alpha
        
        # Calculate SBE with adjusted alpha
        sbe_step = compute_sbe(item['r2_1'], item['r2_2'], alpha=alpha_j, cv_correction=cv_correction)
        
        # Enforce monotonicity across ranks
        running_max_sbe = max(running_max_sbe, sbe_step)
        holm_sbe_results[idx] = running_max_sbe
        
    # 4. Construct output DataFrame
    results = []
    for i, item in enumerate(raw_sbe_data):
        results.append({
            'Pair': item['Pair'],
            'Mean Diff (M1 - M2)': item['Mean Diff'],
            f'SBE Raw (alpha={alpha})': item['SBE_raw'],
            f'SBE Holm (alpha={alpha})': holm_sbe_results[i]
        })
        
    return pd.DataFrame(results)

# another function coded by Gemini; in this case, we are not considering equivalence, but just non-inferiority
def find_equivalent_best(dataset_results, margin=0.03, alpha=0.05, cv_correction=True):
    """
    Identifies the best model and finds all models that are non-inferior (practically equivalent) 
    to it, using Nadeau-Bengio corrected non-inferiority testing and Holm-Bonferroni.
    
    Parameters:
    - dataset_results: dict mapping model names to lists of R2 scores for a single dataset.
    - margin: the non-inferiority margin (Delta). Positive float.
    - alpha: family-wise significance level (default 0.05).
    - cv_correction: bool, apply Nadeau-Bengio CV correction (default True).
    
    Returns:
    - DataFrame summarizing the best model, mean scores, p-values, and equivalence status.
    """
    # Calculate means to find the empirical best model
    means = {model: np.mean(scores) for model, scores in dataset_results.items()}
    best_model_name = max(means, key=means.get)
    best_scores = np.array(dataset_results[best_model_name])
    
    k = len(best_scores)
    df = k - 1
    
    results = []
    p_values_raw = []
    
    # 1. Run Non-Inferiority test against all OTHER models
    other_models = [m for m in dataset_results.keys() if m != best_model_name]
    
    for model in other_models:
        other_scores = np.array(dataset_results[model])
        
        # d is positive (Best - Other)
        d = best_scores - other_scores
        mean_d = np.mean(d)
        sd = np.std(d, ddof=1)
        
        if cv_correction:
            se = np.sqrt((1 / k + 1 / (k - 1))) * sd
        else:
            se = sd / np.sqrt(k)
            
        if se == 0:
            p_val = 0.0 if mean_d < margin else 1.0
        else:
            # Non-inferiority test: H0: mean_d >= margin  --> H1: mean_d < margin
            t_stat = (mean_d - margin) / se
            p_val = t.cdf(t_stat, df) # Left-tail probability
            
        p_values_raw.append(p_val)
        
        results.append({
            'Model': model,
            'Mean R2': means[model],
            'Diff from Best': mean_d,
            'p_raw': p_val
        })
        
    # 2. Apply Holm-Bonferroni correction across the (M-1) comparisons
    m = len(results)
    sorted_indices = sorted(range(m), key=lambda i: p_values_raw[i])
    
    adj_p_values = [0.0] * m
    running_max = 0.0
    
    for rank, idx in enumerate(sorted_indices):
        raw_p = p_values_raw[idx]
        multiplier = m - rank
        unadjusted_adj_p = raw_p * multiplier
        
        running_max = max(running_max, unadjusted_adj_p)
        adj_p_values[idx] = min(1.0, running_max)
        
    # 3. Format final output, adding the Best model back in
    final_output = [{
        'Model': best_model_name,
        'Mean R2': means[best_model_name],
        'Diff from Best': 0.0,
        'p_raw': np.nan,
        'p_adjusted': np.nan,
        'Equivalent Best': True,
        'Status': 'Empirical Best'
    }]
    
    for i, res in enumerate(results):
        res['p_adjusted'] = adj_p_values[i]
        res['Equivalent Best'] = adj_p_values[i] < alpha
        res['Status'] = 'Equivalent' if adj_p_values[i] < alpha else 'Inferior'
        final_output.append(res)
        
    return pd.DataFrame(final_output).sort_values(by='Mean R2', ascending=False).reset_index(drop=True)