"""
This file defines a function to calculate the power spectrum of a star
"""
# Import modules
import numpy as np
from time import time as now
import os
import scipy.signal
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from matplotlib import cm
import itertools
from math import e
from uncertainties import ufloat
import scipy.stats
from scipy.ndimage import gaussian_filter

# Make nice plots
def matplotlib_setup():
    """ The setup, which makes nice plots for the report"""
    fig_width_pt = 328
    inches_per_pt = 1.0 / 72.27
    golden_mean = (np.sqrt(5) - 1.0) / 2.0
    fig_width = fig_width_pt * inches_per_pt
    fig_height = fig_width * golden_mean
    fig_size = [fig_width, fig_height]
    matplotlib.rc('text', usetex=True)
    matplotlib.rc('figure', figsize=fig_size)
    matplotlib.rc('font', size=8, family='serif')
    matplotlib.rc('axes', labelsize=8)
    matplotlib.rc('legend', fontsize=8)
    matplotlib.rc('xtick', labelsize=8)
    matplotlib.rc('ytick', labelsize=8)
    matplotlib.rc('text.latex', preamble=
                  r'\usepackage[T1]{fontenc}\usepackage{lmodern}')

#matplotlib_setup()
import matplotlib.pyplot as plt
import seaborn as sns

# Activate Seaborn color aliases
sns.set_palette('colorblind')
sns.set_color_codes(palette='colorblind')
#plt.style.use('ggplot')
#sns.set_context('poster')
sns.set_style("ticks")


def power_spectrum(time, amplitude, weight=None, minfreq=None, maxfreq=None,
                   oversample=None, memory_use=None, freq=None):
    """
    This function returns the power spectrum of the desired star.
    Arguments:
        - 'time': Time in megaseconds from the timeserie analysis.
        - 'amplitude': Photometry data from the timeserie analysis.
        - 'weight': Weights for each point in the time series.
        - 'minfreq': The lower bound for the frequency interval
        - 'maxfreq': The upper bound for the frequency interval
        - 'oversample': The resolution of the power spectrum.
        - 'memory_use': The amount of memory used for this calculation.
        - 'freq': Override minfreq, maxfreq, ... and use these frequencies instead.
    """
    # The default longest wavelength is the length of the time series.
    if minfreq is None:
        #minfreq = 1 / (time[-1] - time[0]) #takes the difference between the last data point and the first data point
        #minfreq = 1 / (time[-1] - time[0]) #takes the difference between the last data point and the first data point
        #minfreq = 775 # this is in microhertz
        minfreq = 0.1 # this is in microhertz


    # The default greatest frequency is the Nyquist frequency.
    if maxfreq is None:
        #maxfreq = 1 /(2 * np.median(np.diff(time))) #This the expression for the Nyquist frequency
        #maxfreq = 1 /(2 * np.median(np.diff(time*86400)))*10**6 #This the expression for the Nyquist frequency
        #maxfreq= 4200 # is in microhertz
        maxfreq = 8400 # is in microhertz
    # By default oversample 4 times
    if oversample is None:
        oversample = 4 # This is used when we make our timestep interval
        #oversample=100
    # By default use 500000 memory cells (8 bytes each).
    if memory_use is None:
        memory_use = 500000
    if weight is None:
        weight = np.ones(amplitude.shape)
    else:
        weight = np.asarray(weight)
        assert weight.shape == amplitude.shape

    if freq is None:
        # Generate cyclic frequencies
        #step = 1 / (oversample*(time[-1] - time[0])) #Here we can see how the oversample from before is used to determine the time step interval
        #step = 1 / (oversample *((time[-1] - time[0])*86400))*10**6 #Here we can see how the oversample from before is used to determine the time step interval
        step = 0.1
        freq = np.arange(minfreq, maxfreq, step)
    print("Computing power spectrum for frequencies " +
          "[%g, %g] with step %g" %
          (freq.min(), freq.max(), np.median(np.diff(freq))))

    # Generate list to store the calculated power
    alpha = np.zeros((len(freq),))
    beta = np.zeros((len(freq),))

    # Convert frequencies to angular frequencies
    nu = 2 * np.pi * freq
    #nu=freq
    # Iterate over the frequencies
    timerStart = now()

    # After this many frequencies, print progress info
    print_every = 75e6 // len(time)

    # Define a chunk for the calculation in order to save time
    chunksize = memory_use // len(time)
    chunksize = max(chunksize, 1)

    # Ensure chunksize divides print_every
    print_every = (print_every // chunksize) * chunksize

    for i in range(0, len(nu), chunksize):
        # Define chunk
        j = min(i + chunksize, len(nu))
        rows = j - i

        # Info-print
        if i % print_every == 0:
            elapsedTime = now() - timerStart
            if i == 0:
                totalTime = 0.004 * len(nu)
            else:
                totalTime = (elapsedTime / i) * len(nu)

            print("Progress: %.2f%% (%d of %d)  "
                  "Elapsed: %.2f s  Total: %.2f s"
                  % (np.divide(100.0*i, len(nu)), i, len(nu),
                     elapsedTime, totalTime))

        """
        The outer product is calculated. This way, the product between
        time and ang. freq. will be calculated elementwise; one column
        per frequency. This is done in order to save computing time.
        """
        nutime = np.outer(time*86400/10**6, nu[i:j]) #the time is in megaseconds and frequency is in  microhertz and converted from cycles pr day

        """
        An array with the measured amplitude is made so it has the same size
        as "nutime", since we want to multiply the two.
        """
        amplituderep = amplitude.reshape(-1, 1)
        weightrep = weight.reshape(-1, 1)

        # The Fourier subroutine
        sin_nutime = np.sin(nutime)
        cos_nutime = np.cos(nutime)

        # Making the Fourier Transformation
        s = np.sum(weightrep * sin_nutime * amplituderep, axis=0)
        c = np.sum(weightrep * cos_nutime * amplituderep, axis=0)
        ss = np.sum(weightrep * sin_nutime ** 2, axis=0)
        cc = np.sum(weightrep * cos_nutime ** 2, axis=0)
        sc = np.sum(weightrep * sin_nutime * cos_nutime, axis=0)

        alpha[i:j] = ((s * cc) - (c * sc)) / ((ss * cc) - (sc ** 2)) # calculates the alpha values
        beta[i:j] = ((c * ss) - (s * sc)) / ((ss * cc) - (sc ** 2)) # calculates the beta values

    alpha = alpha.reshape(-1, 1)
    beta = beta.reshape(-1, 1)
    freq = freq.reshape(-1, 1)#*86400*10**6
    power = alpha ** 2 + beta ** 2 # calculates the Power
    elapsedTime = now() - timerStart
    print('Computed power spectrum in %.2f s' % (elapsedTime))
    return (freq, power, alpha, beta)



def calculate_v_nl_modes(delta_nu, epsilon, dnu, nmin, nmax):
	"""
	Solves the asymptotic relation
    Depening on which small seperation you look for dnu should be divided with something else. For dnu_02 it is 6
	"""
	v_nl = [delta_nu*(n+l/2+epsilon)-(dnu/6)*(l*(l+1)) for n,l in itertools.product(range(nmin, nmax), (0,1,2,3))]
	return v_nl

def calculate_v_nl(delta_nu, epsilon, dnu, nmin, nmax):
	"""
	Solves the asymptotic relation
    Depening on which small seperation you look for dnu should be divided with something else. For dnu_02 it is 6
	"""
	v_nl = [delta_nu*(n+l/2+epsilon)-(dnu/6)*(l*(l+1)) for n,l in itertools.product(range(nmin, nmax), (0,2))]
	return v_nl

def calculate_MF(included_peak, v_nl, resnu):
    #Calculates the best matching values for given peaks and v_nl
    MF = 0
    for nui in included_peak:
        for nuni in v_nl:
            MF += np.exp(-((nui-nuni)/(2*resnu))**2)
    return MF


def matched_filter(epsilon, dnu02, included_peak, d_nu):
    delta_nu = d_nu         # insert your large frequency splitting here.
    nmin = 0            # degree of note that it run over. These may be modifies since
                        # 0 and 45 is quit extreme but serves as a prove of concept.
    nmax = 45
    resnu = 0.5        # resolution of the peak. Depending of how good your dataset is
                        # you might be able to make this more narrow (i.e 0.5) or enlargen
                        # if you can't fint something good

    v_nls = calculate_v_nl(delta_nu, epsilon, dnu02, nmin, nmax)
    return calculate_MF(included_peak, v_nls, resnu)




# In order to perform the fit, we choose to weight the data by fitting the model to logaritmic bins.
def running_median(freq, powerden, weights=None, bin_size=None, bins=None):
    if bin_size is not None and bins is not None:
        raise TypeError('cannot specify both bin_size and bins')
    freq = np.squeeze(freq)
    powerden = np.squeeze(powerden)
    n, = freq.shape
    n_, = powerden.shape
    assert n == n_

    if weights is None:
        weights = np.ones(n, dtype=np.float32)

    # Sort data by frequency
    sort_ind = np.argsort(freq)
    freq = freq[sort_ind]
    powerden = powerden[sort_ind]
    weights = weights[sort_ind]

    # Compute log of frequencies
    log_freq = np.log10(freq)
    # Compute bin_size
    if bin_size is None:
        if bins is None:
            bins = 10000
        df = np.diff(log_freq)
        d = np.median(df)
        close = df < 100*d
        span = np.sum(df[close])
        bin_size = span / bins
    bin_index = np.floor((log_freq - log_freq[0]) / bin_size)
    internal_boundary = 1 + (bin_index[1:] != bin_index[:-1]).nonzero()[0]
    boundary = [0] + internal_boundary.tolist() + [n]

    bin_freq = []
    bin_pden = []
    bin_weight = []
    for i, j in zip(boundary[:-1], boundary[1:]):
        bin_freq.append(np.mean(freq[i:j]))
        bin_pden.append(np.median(powerden[i:j]))
        bin_weight.append(np.sum(weights[i:j]))
    return np.array(bin_freq), np.array(bin_pden), np.array(bin_weight)

# Eq. 1 in mentioned paper
def background_fit_2(nu, sigma, tau):
    k1 = ((4 * sigma ** 2 * tau) /(1 + (2 * np.pi * nu * tau) ** 2 +(2 * np.pi * nu * tau) ** 4))
    return k1

def background_fit(nu, sigma_0, tau_0, sigma_1, tau_1, P_n):
    k1 = background_fit_2(nu=nu, sigma=sigma_0, tau=tau_0)
    k2 = background_fit_2(nu=nu, sigma=sigma_1, tau=tau_1)
    return P_n + k1 + k2

def logbackground_fit(nu, sigma_0, tau_0, sigma_1, tau_1, P_n):
        assert nu.all() > 0
        assert np.all(np.isfinite(nu)) == True

        xs = background_fit(nu, sigma_0, tau_0, sigma_1, tau_1, P_n)
        invalid = xs <= 0
        xs[invalid] = 1
        log_xs = np.log10(xs)
        log_xs[invalid] = -10000  # return a very low number for log of something negative
        return log_xs

def gridsearch(f, xs, ys, params):
        # Save l2-norm in a dictionary for the tuple of chosen parameters
        score = {}
        dxs = np.diff(np.log10(xs))
        dxs = np.concatenate([dxs, [dxs[-1]]])
        for p in itertools.product(*params):
            print('\rNow %f %f %f %f %f, Done %f' %
                  (*p, len(score)/np.product([len(x) for x in params])),
                  end='')
            zs = f(xs, *p)
            score[p] = np.sum((ys- zs) ** 2)
        print('')
        return min(score.keys(), key=lambda p: score[p])

def fit_large_freq_separation(data, a0, a1, a2, a3, a4, a5, a6,delta_nu, k,c):
    return (a0*np.exp(-(data-0.5*delta_nu)**2/c**2) + a1*np.exp(-(data-delta_nu)**2/c**2)+
            a2*np.exp(-(data-1.5*delta_nu)**2/c**2) + a3*np.exp(-(data-2*delta_nu)**2/c**2)+
            a4*np.exp(-(data-2.5*delta_nu)**2/c**2) + a5*np.exp(-(data-3*delta_nu)**2/c**2)+
            a6*np.exp(-(data-3.5*delta_nu)**2/c**2) + k)


def linear_regression(x, y, prob):
    """
    Return the linear regression parameters and their <prob> confidence intervals.
    ex:
    >>> linear_regression([.1,.2,.3],[10,11,11.5],0.95)
    """
    x = np.array(x)
    y = np.array(y)
    n = len(x)
    xy = x * y
    xx = x * x

    # estimates

    b1 = (xy.mean() - x.mean() * y.mean()) / (xx.mean() - x.mean()**2)
    b0 = y.mean() - b1 * x.mean()
    s2 = 1./n * sum([(y[i] - b0 - b1 * x[i])**2 for i in range(n)])
    print('b0 = ',b0)
    print('b1 = ',b1)
    print('s2 = ',s2)

    #confidence intervals

    alpha = 1 - prob
    c1 = scipy.stats.chi2.ppf(alpha/2.,n-2)
    c2 = scipy.stats.chi2.ppf(1-alpha/2.,n-2)
    print('the confidence interval of s2 is: ',[n*s2/c2,n*s2/c1])

    c = -1 * scipy.stats.t.ppf(alpha/2.,n-2)
    bb1 = c * (s2 / ((n-2) * (xx.mean() - (x.mean())**2)))**.5
    print('the confidence interval of b1 is: ',[b1-bb1,b1+bb1])
    Delta_nu_lower = b1-bb1
    Delta_nu_upper = b1+bb1

    bb0 = c * ((s2 / (n-2)) * (1 + (x.mean())**2 / (xx.mean() - (x.mean())**2)))**.5
    print('the confidence interval of b0 is: ',[b0-bb0,b0+bb0])
    return Delta_nu_lower, Delta_nu_upper

def echelle(freq, power , delta_nu, save=None):

    fres = (freq[-1] - freq[0]) / (len(freq)-1)
    numax = (delta_nu / 0.263) ** (1 / 0.772)
    nmax = int(np.round(((numax - freq[0]) / delta_nu) - 1))
    nx = int(np.round(delta_nu / fres))
    print(nx)
    assert nx % 2 == 0  # we shift by nx/2 pixels below
    dnu = nx * fres
    print(dnu)
    ny = int(np.floor(len(power) / nx))

    startorder = nmax - 7
    print(startorder)
    endorder = nmax + 3
    # print("%s pixel rows of %s pixels" % (endorder-startorder, nx))

    start = int(startorder * nx)
    endo = int(endorder * nx)
    #start = 0
    #endo = len(freq)
    print(start)
    print(endo)
    apower = power[start:endo]
    pixeldata = np.reshape(apower, (-1, nx))

    def plot_position(freqs):
        o = freqs - freq[start]
        x = o % dnu
        y = start * fres + dnu * np.floor(o / dnu)
        return x, y

    h = plt.figure()
    plt.xlabel(r'Frequency mod $\Delta\nu$ [$\mu$Hz]' % dnu)
    plt.ylabel(r'Frequency [$\mu$Hz]')
    # Subtract half a pixel in order for data points to show up
    # in the middle of the pixel instead of in the lower left corner.
    plt.xlim([-fres/2, dnu-fres/2])
    plt.ylim([start * fres, endo * fres])
    for row in range(pixeldata.shape[0]):
        bottom = (start + (nx * row)) * fres
        top = (start + (nx * (row + 1))) * fres
        blur_data = gaussian_filter(pixeldata[row:row+1], 75)
        plt.imshow(blur_data, aspect='auto', cmap='Blues',
                   interpolation='gaussian', origin='lower',
                   extent=(-fres/2, dnu-fres/2, bottom, top))
    if save is None:
        plt.savefig('/home/simonbanerjee/Dropbox/Speciale/Master_Thesis/%s_echelle_%0.2f.eps' % ('16_Cyg_a', delta_nu),
                    bbox_inches='tight')
    return h, plot_position
