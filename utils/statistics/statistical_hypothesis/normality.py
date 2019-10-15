#taken from https://machinelearningmastery.com/statistical-hypothesis-tests-in-python-cheat-sheet/

#This section lists statistical tests that you can use to check if your data has a Gaussian distribution

from scipy.stats import shapiro
def ShapiroWilk(data):
    """Tests whether a data sample has a Gaussian distribution.
    Assumptions
    -----------
    Observations in each sample are independent and identically distributed (iid).
    Interpretation
    --------------
    H0: the sample has a Gaussian distribution.
    H1: the sample does not have a Gaussian distribution.

    Parameters
    ----------
    data: array of sample data to test.

    Returns
    -------
    stat:float: The test statistics
    p:float: The p-value for the hypothesis test.
    """
    stat, p = shapiro(data)
    return stat, p
	
from scipy.stats import normaltest
def DAgostinosK2(data):
    """Tests whether a data sample has a Gaussian distribution.
    Assumptions
    -----------
    * Observations in each sample are independent and identically distributed (iid).
    Interpretation
    --------------
    * H0: the sample has a Gaussian distribution.
    * H1: the sample does not have a Gaussian distribution.

    Parameters
    ----------
    data: array of sample data to test.

    Returns
    -------
    stat : float or array
        `s^2 + k^2`, where `s` is the z-score returned by `skewtest` and `k` is the z-score returned by `kurtosistest`.  
    p : float or array
        A 2-sided chi squared probability for the hypothesis test.
    """
    stat, p = normaltest(data)
    return stat, p

from scipy.stats import anderson
def AndersonDarling(data):
    """Tests whether a data sample has a Gaussian distribution.
    Assumptions
    -----------
    * Observations in each sample are independent and identically distributed (iid).
    Interpretation
    --------------
    * H0: the sample has a Gaussian distribution.
    * H1: the sample does not have a Gaussian distribution.


    Parameters
    ----------
    data: array of sample data to test.

    Returns
    -------
    stat:float: the test statistics.
    critical:list: critical values.
    sig:list: the significance values for the criticals in percents.
    """
    stat, critical, sig = anderson(data)
    return stat, critical, sig
