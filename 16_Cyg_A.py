import numpy as np
import statistics as stat
import math
import astropy
import matplotlib.pyplot as plt
import glob
import scipy
import time
import sys
#sys.path.insert(0, '/home/simonbanerjee/Dropbox/Speciale/Lightcurve_to_powerspectrum')
sys.path.insert(0, r"/home/simonbanerjee/Dropbox/Speciale/Lightcurve_to_powerspectrum")
from astropy.convolution import Gaussian1DKernel, convolve
#from scipy.signal import find_peaks
from peak_finder import peak_finder
from filter import autocorr, running_mean_filter, isOdd, find_time
from scipy.optimize import curve_fit
from astropy.modeling import models, fitting
from time_series_powerspectrum import power_spectrum, calculate_v_nl, calculate_MF, matched_filter, fit_large_freq_separation
from time_series_powerspectrum import background_fit_2, background_fit, logbackground_fit, gridsearch, running_median,  linear_regression, echelle
from math import e
from uncertainties import ufloat
from scipy import optimize
from scipy.optimize import curve_fit
#matplotlib_setup()
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from scipy.signal import find_peaks, peak_prominences
from scipy import stats
import statsmodels.api as sm
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

plt.style.use('seaborn-poster')
plt.style.use('seaborn-colorblind')

stepsize = 0.1 #mHz
nyquist=8400 #mHz
data= np.loadtxt(r"/home/simonbanerjee/Dropbox/Speciale/Lightcurve_to_powerspectrum/16_Cyg_A/collected_powerspectrum.txt")
#data= np.loadtxt(r"power_other_half.txt")
freq = data[:,0]
power = data[:,1]



power_final_smoothed =power
powerden=power_final_smoothed

nu_max = 2200
P_n = np.arange(0.1, 0.25, step=0.05)  #[np.median(powerden[freq > f]) for f in np.arange(2000, 6000, step=500)]
guess_sigma_0 =  [n * np.sqrt(np.mean(powerden ** 2)) for n in np.arange(10, 45, step=5)]
guess_tau_0 = [n * (1 / nu_max) for n in np.arange(0.01, 0.1, step=0.05)]
guess_sigma_1 = [n * np.sqrt(np.mean(powerden ** 2)) for n in np.arange(10, 45, step=5)]
guess_tau_1 = [n * (1 / nu_max) for n in np.arange(0.01, 0.1, step=0.05)]

print('Parameterspace is %f-%f, %f-%f, %f-%f, %f-%f, and %f-%f' % (
    np.min(guess_sigma_0), np.max(guess_sigma_0), np.min(guess_tau_0), np.max(guess_tau_0),
    np.min(guess_sigma_1), np.max(guess_sigma_1), np.min(guess_tau_1), np.max(guess_tau_1),
    np.min(P_n), np.max(P_n)))

# Cut out around the signals in order not to overfit them
minimum = 500
maximum = 1500

filt = (freq > minimum) & (freq < maximum)
freq_filt = freq[~filt]
powerden_filt = powerden[~filt]


freq_filt, powerden_filt, ws = running_median(freq_filt, powerden_filt, bin_size=1e-3)

def cost(popt):
    return np.mean((logbackground_fit(freq_filt, *popt) - np.log10(powerden_filt)) ** 2)

freq_fit, powerden_fit, ws = running_median(freq, powerden, bin_size=1e-4)

z0 = [guess_sigma_0, guess_tau_0, guess_sigma_1, guess_tau_1, P_n]
popt = gridsearch(logbackground_fit, freq_fit, np.log10(powerden_fit), z0)
# popt = [52.433858, 0.000885, 81.893752, 0.000167, 0.220056]

print('Best parameter for background were: s_0= %f, t_0= %f, s_1= %f, t_1 =%f, P_n= %f' % tuple(popt))
# Fit
#z0 = [guess_sigma_0, guess_tau_0, guess_sigma_1, guess_tau_1]
popt, pcov = curve_fit(logbackground_fit, freq_fit, np.log10(powerden_fit), p0=popt, maxfev=10000)

print('Best parameter for background were: s_0= %f, t_0= %f, s_1= %f, t_1 =%f, P_n= %f' % tuple(popt))

print('Cost = %f' % cost(popt))
freq_plot = freq#[::1000]
powerden_plot = powerden#[::1000]
#rpowerden_plot = rpowerden[::1000]

f1,ax1=plt.subplots()
plt.loglog(freq_plot, powerden_plot, '0.2', basex=10, basey=10, linewidth=0.5)
plt.loglog(freq_plot, background_fit(freq_plot, *popt), 'steelblue', linestyle='-', basex=10,
               basey=10)
#plt.loglog(freq_plot, popt[4] + background_fit_2(freq_plot, *popt[:2]), 'steelblue', linestyle='--',
#               basex=10, basey=10)
#plt.loglog(freq_plot, popt[4] + background_fit_2(freq_plot, *popt[2:4]), 'steelblue', linestyle='--',
#               basex=10, basey=10)
#plt.loglog(freq_plot, np.ones(len(freq_plot)) * popt[4], 'royalblue', linestyle='--')
# plt.title(r'The power density spectrum of %s' % starname)
axins = ax1.inset_axes([0.47, 0.47, 0.5, 0.5])

axins.loglog(freq_plot, powerden_plot, '0.2', basex=10, basey=10, linewidth=0.5)
axins.loglog(freq_plot, background_fit(freq_plot, *popt), 'steelblue', linestyle='-', basex=10,
               basey=10)
x1, x2, y1, y2 = 100, 3500, 0.01, 1
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_xticklabels('')
axins.set_yticklabels('')
ax1.indicate_inset_zoom(axins)
plt.xlim([freq[5000] , freq[-1] ])
"""
ax1 = f1.add_subplot(111)
ax1.tick_params(axis='both', which='major', labelsize=25)
ax1.tick_params(axis='both', which='minor', labelsize=25)
plt.xlabel(r'Frequency [$\mu$Hz]',fontsize = 25)
plt.ylabel(r'Power density [ppm$^2\, \mu$Hz$^{-1}$]', fontsize = 25)
"""
plt.tight_layout()
#plt.ylim([np.amin(powerden_plot), np.amax(powerden_plot)])

plt.show()
#plt.savefig('%s_backgroundfit_%s_%s.pdf' % (starname,
                #minfreq, maxfreq), bbox_inches='tight')

powerden_background_corrected = powerden - background_fit(freq_plot, *popt)
power=powerden_background_corrected
power_final_smoothed =power

xstart=15000
xend= 30000


std=10
g= Gaussian1DKernel(stddev=std,mode='integrate')
power_final_smoothed=convolve(power,g)
powerden=power_final_smoothed

freq_window=freq[xstart:xend]
#power_final = power_final[35000:50000]
power_final_smoothed=power_final_smoothed[xstart:xend]




power_peak, properties = find_peaks(power_final_smoothed, prominence=0.015)
prominences = peak_prominences(power_final_smoothed, power_peak)[0]
print(prominences)









#Running the autocorrelation
freq_lags = round(len(freq_window)/2)
ACF = autocorr(power_final_smoothed,lags=freq_lags)

# autocorr don't give you x-values, so these you must obtain
#I know the timesteps and the range of my frequencies
lags= np.arange(len(ACF))*stepsize
#lags = lags[500:-1]
#ACF = ACF[500:-1]
lags = lags
ACF = ACF
#max_ACF = max(ACF)  # Find the maximum y value
#delta_nu = lags[ACF.argmax()]  # Find the x value corresponding to the maximum y value
#np.savetxt('delta_nu.txt',np.fabs(delta_nu))
#print("delta nu is %.20f microhertz using autocorrelation before subtracting background" %delta_nu)



p1=0.35
p2= 0.06
p3= 0.2
p4= 0.06
p5= 0.05
p6= 0.05
p7= 0.05
c= 2.5
delta_nu_fit_me = 100
K=0.05

fit_me = p1*np.exp(-(lags-0.5*delta_nu_fit_me)**2/c**2) + p2*np.exp(-(lags-delta_nu_fit_me)**2/c**2)+p3*np.exp(-(lags-1.5*delta_nu_fit_me)**2/c**2) + p4*np.exp(-(lags-2*delta_nu_fit_me)**2/c**2)+ p5*np.exp(-(lags-2.5*delta_nu_fit_me)**2/c**2) + p6*np.exp(-(lags-3*delta_nu_fit_me)**2/2**2)+p7*np.exp(-(lags-3.5*delta_nu_fit_me)**2/c**2) + K

popt, pcov = curve_fit(fit_large_freq_separation, lags, ACF, p0=[p1, p2, p3, p4, p5, p6, p7, delta_nu_fit_me, K, c])
print(popt)
print("delta nu is %.20f microhertz using autocorrelation " %popt[7])
delta_nu = popt[7]
#delta_nu =25
freq_mod_delta_nu =np.mod(freq_window[power_peak],delta_nu)


f2=plt.figure(2)
plt.plot(lags,ACF)
#plt.plot(lags,fit_me)
ax2 = f2.add_subplot(111)
ax2.tick_params(axis='both', which='major', labelsize=25)
ax2.tick_params(axis='both', which='minor', labelsize=25)
plt.plot(lags, fit_large_freq_separation(lags, *popt), 'r-', label="Gaussian Fit")
plt.legend(prop={'size': 30})
plt.xlabel(r'Frequency Shift ($\mu$Hz)',fontsize =25)
plt.ylabel('Autocorrelation', fontsize = 25)
plt.tight_layout()
#plt.title(r'16 Cyg A')

echelle_0= np.loadtxt(r"/home/simonbanerjee/Dropbox/Speciale/Lightcurve_to_powerspectrum/16_Cyg_A/echelle_diagram_for_0.txt")
echelle_1= np.loadtxt(r"/home/simonbanerjee/Dropbox/Speciale/Lightcurve_to_powerspectrum/16_Cyg_A/echelle_diagram_for_1.txt")
echelle_2= np.loadtxt(r"/home/simonbanerjee/Dropbox/Speciale/Lightcurve_to_powerspectrum/16_Cyg_A/echelle_diagram_for_2.txt")
echelle_3= np.loadtxt(r"/home/simonbanerjee/Dropbox/Speciale/Lightcurve_to_powerspectrum/16_Cyg_A/echelle_diagram_for_3.txt")

freq_echelle_for_0 = echelle_0[:,1]
freq_echelle_mod_freq_for_0=echelle_0[:,0]

freq_echelle_for_1=echelle_1[:,1]
freq_echelle_mod_freq_for_1=echelle_1[:,0]

freq_echelle_for_2=echelle_2[:,1]
freq_echelle_mod_freq_for_2=echelle_2[:,0]

freq_echelle_for_3=echelle_3[:,1]
freq_echelle_mod_freq_for_3=echelle_3[:,0]


f3=plt.figure(3)
color ='Blues'
#plt.subplot(2, 1, 1)
#plt.plot(freq_window,np.sqrt(power[xstart:xend]),'#d6d8db')
plt.plot(freq_window,power_final_smoothed,'k-')
#plt.plot(freq_window[power_peak], power_final_smoothed[power_peak], "v")
#plt.plot(freq_peak,np.sqrt(power_peak),'x')
#plt.title('Smoothed, stddev=%d'%std)
ax3 = f3.add_subplot(111)
ax3.tick_params(axis='both', which='major', labelsize=25)
ax3.tick_params(axis='both', which='minor', labelsize=25)
plt.xlabel(r'Frequency [$\mu$Hz]', fontsize = 25)
plt.ylabel(r'Power density [ppm$^2\, \mu$Hz$^{-1}$]', fontsize = 25)
plt.title(r'16 Cyg A', fontsize = 35)
plt.tight_layout()
#plt.grid(True)
plt.xlim([2075,2275])
plt.show(f3)

f4=plt.figure(4)
ax4 = plt.gca()
#plt.subplot(2, 1, 2)
#plt.scatter(freq_mod_delta_nu,freq_peak, c=power_peak, cmap=color)
#plt.scatter(delta_nu+freq_mod_delta_nu,freq_peak, c=power_peak, cmap=color)
#plt.plot(freq_mod_delta_nu, freq_window[power_peak],'bo')
#plt.plot(delta_nu+freq_mod_delta_nu,freq_peak,'bo')
N=len(freq_mod_delta_nu)
area = (30 * np.random.rand(N))**2  # 0 to 15 point radii
#plt.scatter(freq_mod_delta_nu,freq_window[power_peak],s=area,c='blue', alpha=0.5)
#np.savetxt('echelle_diagram_for_0.txt',np.transpose([freq_mod_delta_nu,freq_window[power_peak]]))
plt.text(60,2800,r'$\triangledown: 0$  $\bigcirc: 1$ $\blacksquare : 2$  $\diamond : 3$ ', dict(size=20) )
plt.plot(freq_echelle_mod_freq_for_0,freq_echelle_for_0,'kv')
plt.plot(freq_echelle_mod_freq_for_1,freq_echelle_for_1,'ko')
plt.plot(freq_echelle_mod_freq_for_2,freq_echelle_for_2,'ks')
plt.plot(freq_echelle_mod_freq_for_3,freq_echelle_for_3,'kD')
#plt.title('Echelle Diagram')
plt.xlabel(r'Frequency mod %.2f $\mu$Hz' %delta_nu, fontsize =20)
plt.ylabel(r'Frequency ($\mu$Hz)', fontsize =20)
ax4 = f4.add_subplot(111)
ax4.tick_params(axis='both', which='major', labelsize=20)
ax4.tick_params(axis='both', which='minor', labelsize=20)
#plt.grid(True)
plt.tight_layout()
#plt.savefig('/home/simonbanerjee/Dropbox/Speciale/Master_Thesis/16_cyg_A_echelle.eps')
#plt.show()

#np.savetxt('freq_window.txt',freq_window)
#echelle(freq_window, power_final_smoothed, delta_nu, save=None)

std=(4*(delta_nu))/(2*np.sqrt(2*np.log(2))) # I smooth the the with Gaussian kernal and a std of 4*delta_nu
xstart_after_background=xstart
xend_after_background=xend
freq=freq[xstart_after_background:xend_after_background]
powerden=powerden[xstart_after_background:xend_after_background]

g1= Gaussian1DKernel(stddev=std/stepsize,mode='integrate')
power_gauss2=convolve(powerden,g1)
max_power_smoothed2 = max(power_gauss2)  # Find the maximum y value
nu_max = freq[power_gauss2.argmax()]  # Find the x value corresponding to the maximum y value
print("nu max is %.2f microhertz using gaussian smooth after subtracting the background" %nu_max)


f5=plt.figure(5)
plt.plot(freq,powerden,'b-')
plt.plot(freq, (power_gauss2),'r-')
plt.xlabel(r'Frequency [$\mu$Hz]')
plt.ylabel(r'power [$ppm^2$]')
#plt.show()
#Now I wanna find the small frequency seperation dnu_02 using a matched filter
dnu02_start= 4
dnu02_end=9
epsilon_start= 1.0
epsilon_end= 2.0
resolution = 250

dnu02 = np.linspace(dnu02_start, dnu02_end, resolution)
epsilons = np.linspace(epsilon_start, epsilon_end, resolution)
MF = np.zeros((len(dnu02), len(epsilons)), dtype='float64')     # opsamler data

iter = 0
for k, epsilon in enumerate(epsilons):
    percentage = iter/len(epsilons)*100
    print("The match filter is %f%% done" %percentage)
    iter=iter+1
    for j, dnu in enumerate(dnu02):
        MF[k,j] = matched_filter(epsilon, dnu, freq_window[power_peak],delta_nu)



X, Y = np.meshgrid(epsilons, dnu02)
value_epsilon, value_dnu02 = np.unravel_index(np.argmax(MF), MF.shape)
print("epsilon;", epsilons[value_epsilon])
print("dnu02;", dnu02[value_dnu02])


"""
f6 = plt.figure(6)
ax = f5.gca(projection='3d')
x=np.linspace(dnu02_start,dnu02_end,num=len(MF))
y=np.linspace(epsilon_start,epsilon_end, num= len(MF))
X, Y = np.meshgrid(x,y)
Z=MF
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
f6.colorbar(surf, shrink=0.5, aspect=5)
"""



#fig, ax = plt.subplots()
f6=plt.figure(6)
plt.imshow(MF, origin='lower', extent=[dnu02_start, dnu02_end, epsilon_start, epsilon_end])
ax6 = f6.add_subplot(111)
ax6.tick_params(axis='both', which='major', labelsize=25)
ax6.tick_params(axis='both', which='minor', labelsize=25)
plt.title(r'Matched filter', fontsize = 35)
plt.xlabel(r'$\delta\nu_{02}$', fontsize = 25)
plt.ylabel(r'$\varepsilon$', fontsize = 25)
#plt.tight_layout()
plt.colorbar()
plt.axis('auto')

#axins = ax.inset_axes([0.5, 0.5, 0.1, 0.1])
#axins.imshow(MF, origin='lower', extent=[dnu02_start, dnu02_end, epsilon_start, epsilon_end])
#x1, x2, y1, y2 = dnu02-0.5, dnu02+0.5, epsilon-0.05, epsilon+0.05
#axins.set_xlim(x1, x2)
#axins.set_ylim(y1, y2)
#plt.gca().invert_yaxis()
#axins.set_xticklabels('')
#axins.set_yticklabels('')

#ax.indicate_inset_zoom(axins)

#plt.show()



#Performing scaling relations using Earl Bellinger new approach main sequence stars



Teff_16Cyg =  5825 # K  Ramírez et al. 2009
Fe_H_16Cyg = 0.096  # dex Ramírez et al. 2009
age_Sun = ufloat(4.569, 0.006)

# Enter some data, for example a solar twin
# ufloat holds a measurement and its uncertainty
nu_max_star= ufloat(nu_max, nu_max * 0.01) # muHz, with 1% uncertainty
delta_nu_star= ufloat(delta_nu, delta_nu * 0.001) # muHz, with 0.1% uncertainty
d02_nu_star= ufloat(dnu02, dnu02 * 0.04) # muHz, with 4% uncertainty
Teff_star= ufloat(Teff_16Cyg,Teff_16Cyg * 0.01) # K, with 1% uncertainty
Fe_H_star= ufloat(Fe_H_16Cyg, 0.1) # dex, 0.1 dex uncertainty


# Take the powers from Table 1, here given with more precision
#P       = [      alpha,           beta,           gamma,          delta,        epsilon]
P_age    = [-6.55598425,     9.05883854,     -1.29229053,    -4.24528340,    -0.42594767]
P_mass   = [ 0.97531880,    -1.43472745,               0,     1.21647950,     0.27014278]
P_radius = [ 0.30490057,    -1.12949647,               0,     0.31236570,     0.10024562]
P_R_seis = [ 0.88364851,    -1.85899352,               0,              0,              0]

# Apply the scaling relation
def scalingrelations(nu_max, Delta_nu, delta_nu, Teff, Fe_H, P=P_age,
            nu_max_Sun = ufloat(3090, 30),      # muHz
            Delta_nu_Sun = ufloat(135.1, 0.1),  # muHz
            delta_nu_Sun = ufloat(8.957, 0.059),# muHz
            Teff_Sun = ufloat(5772, 0.8),       # K
            Fe_H_Sun = ufloat(0,    0)):        # dex

    alpha, beta, gamma, delta, epsilon = P

    # Equation 5
    return ((nu_max / nu_max_Sun) ** alpha *
        (Delta_nu / Delta_nu_Sun) ** beta *
        (delta_nu / delta_nu_Sun) ** gamma *
        (Teff / Teff_Sun) ** delta *
        (e**Fe_H / e**Fe_H_Sun) ** epsilon)


scaling_mass = scalingrelations(nu_max_star, delta_nu_star, d02_nu_star, Teff_star, Fe_H_star, P=P_mass)
scaling_radius = scalingrelations(nu_max_star, delta_nu_star, d02_nu_star, Teff_star, Fe_H_star, P=P_radius)
scaling_age = scalingrelations(nu_max_star, delta_nu_star, d02_nu_star, Teff_star, Fe_H_star, P=P_age) * age_Sun

print('Scaling relations using Earl Bellingers scaling relations for main sequence stars')
print('Mass:',scaling_mass,'[solar units]')
print('Radius:', scaling_radius, '[solar units]')
print('Age:', '{:.2u}'.format(scaling_age), '[Gyr]')





"""
#Performing scaling relations using Earl Bellinger new approach for giants
"""



# Enter some data, for example a solar twin
# ufloat holds a measurement and its uncertainty
nu_max_giant= ufloat(nu_max, nu_max * 0.01) # muHz, with 1% uncertainty
delta_nu_giant= ufloat(delta_nu, delta_nu * 0.001) # muHz, with 0.1% uncertainty
#period_spacing_giant= ufloat(1,  0.04) # muHz, with 4% uncertainty
Teff_star_giant= ufloat(Teff_16Cyg,Teff_16Cyg * 0.01) # K, with 1% uncertainty
Fe_H_star_giant= ufloat(Fe_H_16Cyg, 0.1) # dex, 0.1 dex uncertainty


# Take the powers from Table 1, here given with more precision
#P       = [      alpha,           beta,                   delta,        epsilon]
P_age_giant    = [     -8.65,            11.68,                      -10.35,        0.2914]
P_mass_giant   = [     3.176,           -4.195,                       1.076,       -0.0704]
P_radius_giant = [     1.079,           -2.091,                       0.3709,      -0.02565]


# Apply the scaling relation
def scalingrelations_giant(nu_max, Delta_nu, Teff, Fe_H, P_giant=P_age_giant,
            nu_max_Sun = ufloat(3090, 30),      # muHz
            Delta_nu_Sun = ufloat(135.1, 0.1),  # muHz
            Teff_Sun = ufloat(5772, 0.8),       # K
            Fe_H_Sun = ufloat(0,    0)):        # dex

    alpha_giant, beta_giant, delta_giant, epsilon_giant = P_giant

    # Equation 5
    return ((nu_max / nu_max_Sun) ** alpha_giant *
        (Delta_nu / Delta_nu_Sun) ** beta_giant *
        (Teff / Teff_Sun) ** delta_giant *
        (e**Fe_H / e**Fe_H_Sun) ** epsilon_giant)


scaling_mass_giant = scalingrelations_giant(nu_max_giant, delta_nu_giant, Teff_star_giant, Fe_H_star_giant, P_giant=P_mass_giant)
scaling_radius_giant = scalingrelations_giant(nu_max_giant, delta_nu_giant, Teff_star_giant, Fe_H_star_giant, P_giant=P_radius_giant)
scaling_age_giant = scalingrelations_giant(nu_max_giant, delta_nu_giant, Teff_star_giant, Fe_H_star_giant, P_giant=P_age_giant) * age_Sun

print('Scaling relations using Earl Bellingers scaling relations for giants')
print('Mass:',scaling_mass_giant,'[solar units]')
print('Radius:', scaling_radius_giant, '[solar units]')
print('Age:', '{:.2u}'.format(scaling_age_giant), '[Gyr]')





"""
#Performing scaling relations using the normal way
"""




# Enter some data, for example a solar twin
# ufloat holds a measurement and its uncertainty
nu_max_star_normal= ufloat(nu_max, nu_max * 0.01) # muHz, with 1% uncertainty
Teff_star_normal= ufloat(Teff_16Cyg, Teff_16Cyg * 0.01) # K, with 1% uncertainty
delta_nu_star_normal= ufloat(delta_nu, delta_nu * 0.001) # muHz, with 0.1% uncertainty

#P              = [      alpha,           beta,           gamma]
P_mass_normal   = [        3,              -4,             3/2]
P_radius_normal = [        1,              -2,             1/2]


# Apply the scaling relation
def scalingrelations_normal(nu_max, Delta_nu, Teff, P_normal=P_mass_normal,
            nu_max_Sun = ufloat(3090, 30),      # muHz
            Delta_nu_Sun = ufloat(135.1, 0.1),  # muHz
            Teff_Sun = ufloat(5772, 0.8)):       # K


    alpha_normal, beta_normal, gamma_normal = P_normal

    # Equation 5
    return ((nu_max / nu_max_Sun) ** alpha_normal *
        (Delta_nu / Delta_nu_Sun) ** beta_normal *
        (Teff / Teff_Sun) ** gamma_normal)


scaling_mass_normal = scalingrelations_normal(nu_max_star_normal, delta_nu_star_normal, Teff_star_normal, P_normal=P_mass_normal)
scaling_radius_normal = scalingrelations_normal(nu_max_star_normal, delta_nu_star_normal, Teff_star_normal, P_normal=P_radius_normal)

print('Scaling relations using the standard method')
print('Mass:',scaling_mass_normal,'[solar units]')
print('Radius:', scaling_radius_normal, '[solar units]')
