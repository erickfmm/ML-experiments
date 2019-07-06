#taken from https://machinelearningmastery.com/statistical-hypothesis-tests-in-python-cheat-sheet/
#This section lists statistical tests that you can use to check if two samples are related.

from scipy.stats import pearsonr
def pearson_coeff(data1, data2) -> (float, float):
    """Tests whether two samples have a linear relationship.
    Assumptions
    -----------
    * Observations in each sample are independent and identically distributed (iid).
    * Observations in each sample are normally distributed.
    * Observations in each sample have the same variance.
    Interpretation
    --------------
    * H0: the two samples are independent.
    * H1: there is a dependency between the samples.

    Parameters
    ----------
    data1: (N,) array_like
    data2: (N,) array_like

    Returns
    -------
    corr : float
        Pearson's correlation coefficient  
    p : float
        2-tailed p-value  
    """
    corr, p = pearsonr(data1, data2)
    return corr, p

from scipy.stats import spearmanr
def spearman_rank(data1, data2):
    """Tests whether two samples have a monotonic relationship.
    Assumptions
    -----------
    * Observations in each sample are independent and identically distributed (iid).
    * Observations in each sample can be ranked.
    Interpretation
    --------------
    * H0: the two samples are independent.
    * H1: there is a dependency between the samples.

    Parameters
    ----------
    data1: (N,) array_like
    data2: (N,) array_like

    Returns
    -------
    corr : float or ndarray (2-D square)
        Spearman correlation matrix or correlation coefficient (if only 2  
    variables are given as parameters. Correlation matrix is square with  
    length equal to total number of variables (columns or rows) in `a`  
    and `b` combined.  
    p : float
        The two-sided p-value for a hypothesis test whose null hypothesis is  
    that two sets of data are uncorrelated, has same dimension as rho. 
    """
    corr, p = spearmanr(data1, data2)
    return corr, p

from scipy.stats import kendalltau
def kendall_rank(data1, data2) -> (float, float):
    """Tests whether two samples have a monotonic relationship.
    Assumptions
    -----------
    * Observations in each sample are independent and identically distributed (iid).
    * Observations in each sample can be ranked.
    Interpretation
    --------------
    * H0: the two samples are independent.
    * H1: there is a dependency between the samples.

    Parameters
    ----------
    data1: (N,) array_like
    data2: (N,) array_like

    Returns
    -------
    corr : float
        The tau statistic.  
    p : float
        The two-sided p-value for a hypothesis test whose null hypothesis is an absence of association, tau = 0. 
    """
    corr, p = kendalltau(data1, data2)
    return corr, p

from scipy.stats import chi2_contingency
def chi_squared(table):
    """Tests whether two categorical variables are related or independent.
    Assumptions
    -----------
    * Observations used in the calculation of the contingency table are independent.
    * 25 or more examples in each cell of the contingency table.
    Interpretation
    --------------
    * H0: the two samples are independent.
    * H1: there is a dependency between the samples.

    Parameters
    ----------
    table: array_like

    Returns
    -------
    stat : float
        The test statistic.  
    p : float
        The p-value of the test  
    dof : int
        Degrees of freedom  
    expected : ndarray, same shape as observed
        The expected frequencies, based on the marginal sums of the table.  
    """
    stat, p, dof, expected = chi2_contingency(table)
    return stat, p, dof, expected