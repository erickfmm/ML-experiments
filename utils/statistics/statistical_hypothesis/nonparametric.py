#taken from https://machinelearningmastery.com/statistical-hypothesis-tests-in-python-cheat-sheet/


from scipy.stats import mannwhitneyu
def mann_whitney_u(data1, data2) -> (float, float):
    """Tests whether the distributions of two independent samples are equal or not.
    Assumptions
    -----------
    * Observations in each sample are independent and identically distributed (iid).
    * Observations in each sample can be ranked.
    Interpretation
    --------------
    * H0: the distributions of both samples are equal.
    * H1: the distributions of both samples are not equal.

    Parameters
    ----------
    data1, data2 : array_like
        Array of samples, should be one-dimensional. 

    Returns
    -------
    stat : float
        The Mann-Whitney U statistic, equal to min(U for x, U for y) if  
        `alternative` is equal to None (deprecated; exists for backward  
        compatibility), and U for y otherwise.  
    p : float
        p-value assuming an asymptotic normal distribution. One-sided or 
        two-sided, depending on the choice of `alternative`.  
    """
    stat, p = mannwhitneyu(data1, data2)


from scipy.stats import wilcoxon
def wilcoxon_signed_rank(data1, data2) -> (float, float):
    """Tests whether the distributions of two paired samples are equal or not.
    Assumptions
    -----------
    * Observations in each sample are independent and identically distributed (iid).
    * Observations in each sample can be ranked.
    * Observations across each sample are paired.
    Interpretation
    --------------
    * H0: the distributions of both samples are equal.
    * H1: the distributions of both samples are not equal.

    Parameters
    ----------
    data1 : array_like
        The first set of measurements.  
    data2 : array_like, optional
        The second set of measurements.  If `y` is not given, then the `x`  
        array is considered to be the differences between the two sets of  
        measurements.  

    Returns
    -------
    stat : float
        The sum of the ranks of the differences above or below zero, whichever is smaller.  
    p : float
        The two-sided p-value for the test.  
    """
    stat, p = wilcoxon(data1, data2)



from scipy.stats import kruskal
def kruskal_wallis_h(data1, data2) -> (float, float):
    """Tests whether the distributions of two or more independent samples are equal or not.
    Assumptions
    -----------
    * Observations in each sample are independent and identically distributed (iid).
    * Observations in each sample can be ranked.
    Interpretation
    --------------
    * H0: the distributions of all samples are equal.
    * H1: the distributions of one or more samples are not equal.

    Parameters
    ----------
    data1, data2 : array_like
    Two or more arrays with the sample measurements can be given as arguments.  

    Returns
    -------
    stat : float
        The Kruskal-Wallis H statistic, corrected for ties  
    p : float
        The p-value for the test using the assumption that H has a chi-square distribution  
    """
    stat, p = kruskal(data1, data2, ...)



from scipy.stats import friedmanchisquare
def friedman(data1, data2) -> (float, float):
    """Tests whether the distributions of two or more paired samples are equal or not.
    Assumptions
    -----------
    * Observations in each sample are independent and identically distributed (iid).
    * Observations in each sample can be ranked.
    * Observations across each sample are paired.
    Interpretation
    --------------
    * H0: the distributions of all samples are equal.
    * H1: the distributions of one or more samples are not equal.

    Parameters
    ----------
    data1, data2 : array_like
        Arrays of measurements.  All of the arrays must have the same number  
        of elements.  At least 3 sets of measurements must be given.  

    Returns
    -------
    stat : float
        the test statistic, correcting for ties  
    p : float
        the associated p-value assuming that the test statistic has a chi-squared distribution  
    """
    stat, p = friedmanchisquare(data1, data2)#, ...)
