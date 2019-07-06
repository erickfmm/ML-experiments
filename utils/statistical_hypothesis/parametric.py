#taken from https://machinelearningmastery.com/statistical-hypothesis-tests-in-python-cheat-sheet/
#This section lists statistical tests that you can use to compare data samples.
import numpy as np

from scipy.stats import ttest_ind
def student(data1, data2):
    """Tests whether the means of two independent samples are significantly different.
    Assumptions
    -----------
    * Observations in each sample are independent and identically distributed (iid).
    * Observations in each sample are normally distributed.
    * Observations in each sample have the same variance.
    Interpretation
    --------------
    * H0: the means of the samples are equal.
    * H1: the means of the samples are unequal.

    Parameters
    ----------
    data1, data2: array_like
        The arrays must have the same shape. 

    Returns
    -------
    stat : float or array
        The calculated t-statistic.  
    p : float or array
        The two-tailed p-value. 
    """
    stat, p = ttest_ind(data1, data2)
    return stat, p

from scipy.stats import ttest_rel
def paired_student(data1, data2):
    """Tests whether the means of two paired samples are significantly different.
    Assumptions
    -----------
    * Observations in each sample are independent and identically distributed (iid).
    * Observations in each sample are normally distributed.
    * Observations in each sample have the same variance.
    * Observations across each sample are paired.
    Interpretation
    --------------
    * H0: the means of the samples are equal.
    * H1: the means of the samples are unequal.

    Parameters
    ----------
    data1, data2: array_like
        The arrays must have the same shape. 

    Returns
    -------
    stat : float or array
        t-statistic  
    p : float or array
        two-tailed p-value  
    """
    stat, p = ttest_rel(data1, data2)
    return stat, p


from scipy.stats import f_oneway
def ANOVA(data1, data2):
    """Analysis of Variance Test (ANOVA)
    Tests whether the means of two or more independent samples are significantly different.
    Assumptions
    -----------
    * Observations in each sample are independent and identically distributed (iid).
    * Observations in each sample are normally distributed.
    * Observations in each sample have the same variance.
    Interpretation
    --------------
    * H0: the means of the samples are equal.
    * H1: one or more of the means of the samples are unequal.

    Parameters
    ----------
    data1, data2: array_like
        The sample measurements for each group (must have the same shape).

    Returns
    -------
    stat : float
        The computed F-value of the test.  
    p : float
        The associated p-value from the F-distribution. 
    """
    stat, p = f_oneway(data1, data2)#, ...)
    return stat, p
