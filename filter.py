import numpy as np

#*******************************************************

from collections import deque,Counter
from bisect import insort, bisect_left
from itertools import islice

from scipy.interpolate import interp1d
from scipy import arange, array, exp
import scipy
from scipy.signal import medfilt
import bottleneck as bn
from bottleneck import move_median


def RunningMedian(seq, M):
    """
     Purpose: Find the median for the points in a sliding window (odd number in size)
              as it is moved from left to right by one point at a time.
      Inputs:
            seq -- list containing items for which a running median (in a sliding window)
                   is to be calculated
              M -- number of items in window (window size) -- must be an integer > 1
      Otputs:
         medians -- list of medians with size N - M + 1
       Note:
         1. The median of a finite list of numbers is the "center" value when this list
            is sorted in ascending order.
         2. If M is an even number the two elements in the window that
            are close to the center are averaged to give the median (this
            is not by definition)
    """
    seq = iter(seq)
    s = []
    m = M // 2

    # Set up list s (to be sorted) and load deque with first window of seq
    s = [item for item in islice(seq,M)]
    d = deque(s)

    # Simple lambda function to handle even/odd window sizes
    median = lambda : s[m] if bool(M&1) else (s[m-1]+s[m])*0.5

    # Sort it in increasing order and extract the median ("center" of the sorted window)
    s.sort()
    medians = [median()]

    # Now slide the window by one point to the right for each new position (each pass through
    # the loop). Stop when the item in the right end of the deque contains the last item in seq
    for item in seq:
        old = d.popleft()          # pop oldest from left
        d.append(item)             # push newest in from right
        del s[bisect_left(s, old)] # locate insertion point and then remove old
        insort(s, item)            # insert newest such that new sort is not required
        medians.append(median())
    return medians

def binary_search(array, target):
    lower = 0 # mimimum value
    upper = len(array)-1 # maximum value
    while lower < upper:   # use < instead of <=
        x = lower + (upper - lower) // 2
        val = array[x]
        if target == val:
            return x
        elif target > val:
            if lower == x:   # these two are the actual lines
                break        # you're looking for
            lower = x
        elif target < val:
            upper = x

def running_median_filter(x, N):
    """
    >>> running_median_filter([0, 3, 9, 18, 24], 5)
    array([  0.,   3.,   9.,  18.,  24.])
    """
    assert N % 2 == 1
    # Code cannot run unless timetarget is an odd number. Time target can be even, like 6 days,
    #but the number of data points in the data it gives has to be odd, for example like 6 days is after 7613 data point
    x = np.asarray(x)
    extra_count = (N - 1) // 2
    left = []
    right = []
    #This for-loop makes an extrapolation in both ends of the median filter dataself.
    #This makes sure that the data sets is of equal length
    #The way it works is following
    #Say you're time target is 11 days then there is 5 days in each ends
    #Here we take make tiny chunks of the points in the end of the data and then take the median of that
    # Then we take a tinier chunk and a tinier chunk and moving more and more towards the center of the end pieces
    for i in range(extra_count):
        left.append(np.median(x[:2*i+1]))
        right.append(np.median(x[-2*i-1:]))
    right.reverse()
    y = running_median(x, N)
    return np.concatenate([left, y, right])


def running_median(x, N):
    """
    >>> running_median([0, 1, 2, 3, 4], 3)
    array([1, 2, 3])
    """
    assert N % 2 == 1
    # Code cannot run unless timetarget is an odd number. Time target can be even, like 6 days,
    #but the number of data points in the data it gives has to be odd, for example like 6 days is after 7613 data point
    x = np.asarray(x)
    filt = scipy.signal.medfilt(x, N) # Using the scipy median filter
    margin = (N - 1) // 2
    return filt[margin:-margin]

def find_time(array, array_target):
    i=0 #start from zero
    sign=-1 #check when sign changes positive to negative
    while sign<0 and i< len(array):
        sign = array[i] - array[0] - array_target
        i= i+1
    return i-2

def isOdd(x):
    if x % 2 == 0:
        return x+1
    else:
        return x

def autocorr(data, lags=40):
    """
    Autocorrelation done using Python statistics module. The function is basically an interface to the existing funciton - not a re-implementationself.
    Not all of the functionalit of the original function is used.

        Arguments:
            - 'data' : the data which to run the autocorrelation on.
            - 'lags' : Number of lags to return autocorrelation for (default is 40 in the original function)

    """
    #importing the function
    from statsmodels.tsa.stattools import acf

    #run the autocorrelation
    corr = acf(data,nlags=lags)

    #return desired values
    return corr

def running_mean(x, N):
    """
    This calculates the running mean using the cumulative sum in a
    specified window.
    (see wikipedia -> Moving average -> Cumulative moving average.)
    Arguments:
        - 'x': Data series
        - 'N': The size of the window
    The output has N-1 fewer entries than x.
    >>> running_mean([0, 1, 2, 3, 4], 2)
    array([ 0.5,  1.5,  2.5,  3.5])
    """
    x = np.asarray(x)
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N

def running_mean_filter(x, N):
    """
    This fixes the length of the running mean so it has the same length
    as the inputted data series by repeating the last entry $N-1$ times.
    Arguments:
        - 'x': Data series
        - 'N': The size of the window, which need to be a uneven number.
    >>> running_mean_filter_2([0, 3, 9, 15, 18], 5)
    array([  0.,   4.,   9.,  14.,  18.])
    """
    assert N % 2 == 1
    x = np.asarray(x)
    extra_count = (N - 1) // 2
    left = []
    right = []
    for i in range(extra_count):
        left.append(np.mean(x[:2*i+1]))
        right.append(np.mean(x[-2*i-1:]))
    right.reverse()
    y = running_mean(x, N)
    return np.concatenate([left, y, right])
