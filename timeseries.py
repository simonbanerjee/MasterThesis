import numpy as np
import statistics as stat
import math
from scipy.stats import norm
import pandas as pd
import bottleneck as bn
from astropy.stats import sigma_clip
import astropy
from astropy.io import fits
from filter import running_median_filter, binary_search, find_time, running_median, isOdd


import matplotlib.pyplot as plt
plt.style.use('seaborn-poster')
plt.style.use('seaborn-colorblind')


names_of_all_stars= np.loadtxt('list_of_all_quaters.txt')


#for i in range(len(names_of_all_stars)):
for i in names_of_all_stars:


	# If file is a .fits file use this
	#hdulist = fits.open('/home/simonbanerjee/Dropbox/Speciale/Lightcurve_to_powerspectrum/16_Cyg_A/kplr100003551-2011208035123_slc.fits' %i, mode='readonly')
	#hdulist = fits.open('/home/simonbanerjee/Dropbox/Speciale/Lightcurve_to_powerspectrum/16_Cyg_A/kplr%d-%d_slc.fits' %(names_of_all_stars[i,0],names_of_all_stars[i,1]), mode='readonly')
	hdulist = fits.open('/home/simonbanerjee/Dropbox/Speciale/Lightcurve_to_powerspectrum/HD 203949/tess00%d-s01-c0120-dr01-v04-tasoc_lc.fits' %i, mode='readonly')

	#hdulist = fits.open('/home/simonbanerjee/Dropbox/Speciale/Lightcurve_to_powerspectrum/KIC8751420/kplr100004295-2013131215648_slc.fits', mode='readonly')

	LightCurve = hdulist['LIGHTCURVE'].data
	hdulist.close()

	time = LightCurve.field('TIME')
	flux = LightCurve.field('FLUX_RAW')

	"""
	# If file is a .dat use this
	data = np.loadtxt('/home/simonbanerjee/Dropbox/Speciale/Lightcurve_to_powerspectrum/KIC8478994/kplr008478994-%d_slc.dat' %i)
	time = data[:,0]
	rawflux = data[:,1]
	rawflux_error= data[:,2]
	flux = data[:,3]
	flux_error= data[:,4]
	"""


	flux[np.isneginf(flux)] = np.nan


	#Normalizing by dividing y-axis values with the mean value
	flux=flux/np.nanmean(flux)

	# We will try to find tau long and and tau short using the find_time routine
	# The routine checks when the sign change depending on the chosen time target
	days_for_tau_long = 2 # 2 days
	days_for_tau_short = 2/24 # 2 hours
	tau_long= find_time(time,days_for_tau_long)
	tau_short= find_time(time,days_for_tau_short)

	# The isOdd function makes sure that the number find_time spits outs is always odd
	tau_long=isOdd(tau_long)
	tau_short=isOdd(tau_short)

	#Using Bottlenecks move_median fitler
	# Bottleneck takes the median by setting the midpoint at the end and the taking all the previuos points in that chunk

	tau_long_flux = bn.move_median(flux,tau_long,min_count=1)
	tau_short_flux= bn.move_median(flux-tau_long_flux,tau_short,min_count=1)+tau_long_flux

	#I remove a chunk of data in the start of the data due to the way bottleneck calculates the median
	#I use my find_time function to do this
	tau_long_data_removal =find_time(time,days_for_tau_long)
	tau_short_data_removal =find_time(time,days_for_tau_short)

	# I make sure that my data starts from tau and ends with the last data point
	tau_long_flux = tau_long_flux[tau_long_data_removal:-1]
	tau_short_flux= tau_short_flux[tau_short_data_removal:-1]
	tau_long_time = time[tau_long_data_removal:-1]
	tau_short_time= time[tau_short_data_removal:-1]

	#Now I shift my data with tau/2 so that my data fits with the median filter
	tau_long_time = tau_long_time-days_for_tau_long/2
	tau_short_time= tau_short_time-days_for_tau_short/2

	#Here I look at how many points there is in the start and the end between the data set and the median filter
	tau_long_number_points_start=find_time(time,tau_long_time[0]-time[0])
	tau_long_number_points_end=find_time(time,time[-1]-tau_long_time[-1])

	tau_short_number_points_start=find_time(time,tau_short_time[0]-time[0])
	tau_short_number_points_end=find_time(time,time[-1]-tau_short_time[-1])

	#Now I make some lists of ones that I am going to multiply with the first data point of the median filter (for left side) and the last data point of the median filter (right side)
	tau_long_left_extrapol = np.ones(tau_long_number_points_start+1)
	tau_long_right_extrapol = np.ones(tau_long_number_points_end)
	tau_short_left_extrapol = np.ones(tau_short_number_points_start+1)
	tau_short_right_extrapol = np.ones(tau_short_number_points_end)

	# Now I multiply the ones that makes up the left hand side with the inital value of the median filter
	# and I multiply the ones that makes up the rigth hand side with the final value of the median filter
	tau_long_left_extrapol  = tau_long_left_extrapol  * tau_long_flux[0]
	tau_long_right_extrapol = tau_long_right_extrapol * tau_long_flux[-1]
	tau_short_left_extrapol = tau_short_left_extrapol * tau_short_flux[0]
	tau_short_right_extrapol= tau_short_right_extrapol* tau_short_flux[-1]

	# I make sure that the median times goes out in both ends
	tau_long_time_start = np.linspace(time[0],tau_long_time[0], tau_long_number_points_start+1)
	tau_long_time_end  	= np.linspace(tau_long_time[-1], time[-1],tau_long_number_points_end)
	tau_long_time       = np.concatenate([tau_long_time_start, tau_long_time, tau_long_time_end])

	tau_short_time_start= np.linspace(time[0],tau_short_time[0], tau_short_number_points_start+1)
	tau_short_time_end  = np.linspace(tau_short_time[-1], time[-1],tau_short_number_points_end)
	tau_short_time 	    = np.concatenate([tau_short_time_start, tau_short_time, tau_short_time_end])

	# Now I make sure that the median fluxes goes out in both ends using a constant value in the start and a constant value in the end
	tau_long_flux = np.concatenate([tau_long_left_extrapol, tau_long_flux, tau_long_right_extrapol])
	tau_short_flux = np.concatenate([tau_short_left_extrapol, tau_short_flux, tau_short_right_extrapol])


	#Plotting the data with the two different median filters

	"""
	plt.figure
	plt.plot(time,flux,'k-',time,tau_long_flux,'r-')
	plt.title(r'$\tau$-long')
	plt.xlabel('time [Days]')
	plt.ylabel('Flux')
	plt.show()
	#plt.savefig('median_filter_tau_long.png')



	plt.figure
	plt.plot(time,flux,'k-',time,tau_short_flux,'r-')
	plt.title(r'$\tau$-short')
	plt.xlabel('time [Days]')
	plt.ylabel('Flux')
	plt.show()
	#plt.savefig('median_filter_tau_short.eps')
	"""

	# To detect the presence of sharp features, I construct this diagnostic signal
	w_flux= (tau_long_flux/tau_short_flux)-1
	"""
	#Plotting the diagnostic signal vs time to see sharp features
	plt.figure
	plt.plot(time,w_flux)
	plt.title('w [diagnostic signal] vs time')
	plt.xlabel('Time [Days]',fontsize = 25)
	plt.ylabel('w', fontsize = 25)
	#plt.savefig('/home/simonbanerjee/Dropbox/Speciale/Lightcurve_to_powerspectrum/16_Cyg_A/w_vs_time/w_vs_time_%d.eps'%names_of_all_stars[i,1])
	plt.clf()
	plt.show()
	"""

	#I calculate the MAD (moving absolute deviation) of the diagnostic signal
	sigma_w_flux =running_median_filter(np.fabs(w_flux-running_median_filter(w_flux,tau_long)),tau_long)
	"""
	#Plotting the MAD to see if it's correct
	plt.figure
	plt.plot(time,sigma_w_flux)
	plt.title('MAD vs time')
	plt.xlabel('time [days]')
	plt.ylabel('MAD')
	plt.show()
	"""
	#Creating the final filter as weighted mean between the short and long filters
	# First we create the c values, which is the turnover function between the two filters
	# which returns 1 or 0 given sigma_w/mean_sigma_w.

	# These values are used as default
	mu_to = 5 # Center of the graph
	sigma_to = 1 # standard deviation

	sigma_ratio_flux= w_flux/sigma_w_flux

	c_flux= norm.cdf(sigma_ratio_flux,mu_to,sigma_to) # c values are calculated using the cumulatative distribution function

	# Now I make the final filter
	filter_flux = c_flux*tau_short_flux+(1-c_flux)*tau_long_flux
	#this one should be used normally
	flux_filt=(10**6)*((flux/filter_flux)-1)

	#used to make a figure
	flux_filt=((flux/filter_flux)-1)
	"""

	plt.figure()
	plt.plot(time,flux_filt,'b')
	plt.title(r'Filter applied')
	plt.xlabel('Time [Days]',fontsize = 25)
	plt.ylabel('Flux [ppm]',fontsize = 25)
	plt.show()
	#plt.savefig('/home/simonbanerjee/Dropbox/Speciale/Lightcurve_to_powerspectrum/16_Cyg_A/final_filter/final_filter_%d.png'%names_of_all_stars[i,1])
	"""
	#Now it's time to do some sigma clipping
	# Astropy has a sigma clipping function, however this function doesn't work with NaN or inf values, which means these must be deleted first

	finite_filter = np.isfinite(flux_filt) # makes a filter that checks in the flux column for only finite values (No, NaN, inf or is inf)
	#aplly filter to both time and flux
	time_filt= time[finite_filter]
	flux_filt= flux_filt[finite_filter]
	#Performing the sigma clipping to the data with only finite values
	filter_sigma_cliping = astropy.stats.sigma_clip(flux_filt, sigma=4, iters=5,stdfunc=astropy.stats.mad_std)
	time_filt= time_filt[~filter_sigma_cliping.mask]
	flux_filt = flux_filt[~filter_sigma_cliping.mask]
	#print(flux_filt.shape)
	"""
	plt.figure()
	plt.plot(time_filt,flux_filt, '-', color='#1f77b4', label="flux_filt")
	plt.title('Corrected time series data')
	plt.xlabel('Time [days]')
	plt.ylabel('Corrected flux [ppm]')
	plt.legend(loc='best', numpoints=1)
	plt.tight_layout() #This makes sure that the legends do not overlap
	plt.show()
	#plt.savefig('/home/simonbanerjee/Dropbox/Speciale/Lightcurve_to_powerspectrum/16_Cyg_A/timeseries/timeseries_%d.png'%names_of_all_stars[i,1])
	plt.clf()
	plt.show()
	#np.savetxt('/home/simonbanerjee/Dropbox/Speciale/Lightcurve_to_powerspectrum/nu_indi/time_series_for_%d.txt' %names_of_all_stars[i],np.transpose([time_filt,flux_filt]))
	"""

##################################################################################
#figure removing transits for the thesis

	fig1=plt.figure(1)
	plt.plot(time,flux-1+0.04,'b-')
	plt.plot(time,tau_long_flux-1+0.04,'r-', linewidth=10)
	plt.plot(time,flux-1+0.02,'b-')
	plt.plot(time,tau_short_flux-1+0.02,'r-', linewidth=10)

	plt.plot(time_filt,flux_filt, 'b-')
	plt.xlabel('Time [Days]',fontsize=25)
	plt.ylabel('Flux [ppm]',fontsize=25)
	ax = fig1.add_subplot(111)
	ax.tick_params(axis='both', which='major', labelsize=25)
	ax.tick_params(axis='both', which='minor', labelsize=25)
	#plt.title(r'$\beta$ Hydri', fontsize=45)
	plt.xlim(1347, 1349)
	plt.ylim(-0.01, 0.05)
	plt.tight_layout()
	#plt.savefig('/home/simonbanerjee/Dropbox/Speciale/Master_Thesis/timeseries_HD203949.eps', bbox_inches="tight")
	plt.show()
#####################################################################################






	# Estimating the errors by saying  sigma= k*movingmedian(|x_filt|,tau_long)
	#k=1.4826 # conversion factor going from MAD to standard deviation
	#errors= k*bn.move_median(np.fabs(flux_filt),tau_long,min_count=1)
	#np.savetxt('errors.txt',errors)
