#######################################################################################
### Theory of Kramers-Moyal expansion applied to the estimation of system stability.
### Suitable for running window analyses of potentially bifurcating/tipping systems.
### Early Warning Signal (EWS) for tipping is a decrease in linearized feedback lambda.
### It is estimated below, along with the standard EWS of variance and AC(1).
# Authors: Andreas Morr, Keno Riechers, Leonardo Rydin Gorjao, Niklas Boers




import numpy as np
from kramersmoyal import km

### Parameters to be specified for the use of kramersmoyal by Rydin Gorjao et al.
bin_number = 50
bins = np.array([bin_number])
powers = np.array([[0], [1], [2]])
###

### Only some fraction of state-space bins centered around the stable state are used for further analysis
centerkmfraction = 0.4 
###

def linear_detrending(data_array, keep_center = False):
    center = np.mean(data_array)
    fit = np.polyfit(range(len(data_array)), data_array, 1)
    data_array = data_array - np.array([x * fit[0] + fit[1] for x in range(len(data_array))])
    if keep_center:
        return data_array + center
    else:
        return data_array

def full_km_analysis(data_array, Delta_t, bins = bins, powers = powers, centerkmfraction = centerkmfraction):
    data_array = linear_detrending(data_array, keep_center=True)
    bin_number=bins[0]
    bw = 2*(max(data_array)-min(data_array))/bin_number
    ### Determine stable state
    center = np.mean(data_array)
    ### Get estimations of Kramers-Moyal coefficients and keep only the specified fraction around the stable state
    kmc, edges = km(data_array, bw = bw, bins = bins, powers = powers)
    edges = edges[0]
    km1 = kmc[1]/Delta_t
    km2 = kmc[2]/Delta_t
    center_index = np.searchsorted(edges,center)
    edges = edges[center_index - round(bin_number*centerkmfraction/2):center_index + round(bin_number*centerkmfraction/2)]
    km1 = km1[center_index - round(bin_number*centerkmfraction/2):center_index + round(bin_number*centerkmfraction/2)]
    km2 = km2[center_index - round(bin_number*centerkmfraction/2):center_index + round(bin_number*centerkmfraction/2)]
    ### Estimate lambda as the best linear fit to the first order KM coefficient
    lambda_est = -1*np.polyfit(edges,km1,1)[0]
    return [edges, km1, km2, lambda_est]

def lambda_estimator(data_array, Delta_t, bins = bins, powers = powers, centerkmfraction = centerkmfraction):
    return full_km_analysis(data_array, Delta_t, bins = bins, powers = powers, centerkmfraction = centerkmfraction)[3]



def variance_estimator(data_array):
    data_array = linear_detrending(data_array)
    return np.sum(np.array([x**2 for x in data_array]))/len(data_array)

def ac1_estimator(data_array, lag=1):
    data_array = linear_detrending(data_array)
    return np.sum(np.array([data_array[i]*data_array[i + lag] for i in range(len(data_array)-lag)]))/(len(data_array)-lag)/variance_estimator(data_array)