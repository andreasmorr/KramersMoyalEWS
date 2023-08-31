#######################################################################################
### Theory of Kramers-Moyal expansion applied to the estimation of system stability.
### Suitable for running window analyses of potentially bifurcating/tipping systems.
### Early Warning Signal (EWS) for tipping is a decrease in linearized feedback lambda.
### It is estimated below, along with the standard EWS of variance and AC(1).
# Authors: Andreas Morr, Keno Riechers, Leonardo Rydin Gorjao, Niklas Boers




import numpy as np
from scipy import stats
import itertools
from kramersmoyal import km

### Parameters to be specified for the use of kramersmoyal by Rydin Gorjao et al.
bin_number = 50
###

### Only some fraction of state-space bins centered around the stable state are used for further analysis
centerkmfraction = 0.5
###

def linear_detrending_1D(data_array, keep_center = False):
    data_array = np.copy(data_array)
    center = np.mean(data_array)
    fit = np.polyfit(range(len(data_array)), data_array, 1)
    data_array = data_array - np.array([x * fit[0] + fit[1] for x in range(len(data_array))])
    if keep_center:
        return data_array + center
    else:
        return data_array

def full_km_analysis_1D(data_array, Delta_t, bin_number = bin_number, bw=None, centerkmfraction = centerkmfraction):
    powers = np.array([[0], [1], [2]])

    data_array = linear_detrending_1D(data_array, keep_center=True)
    bins = np.array([bin_number])
    if bw is None:
        bw = 14*np.std(data_array)/bin_number
    #print(bw/np.std(data_array))
    ### Determine stable state
    center = np.mean(data_array)
    ### Get estimations of Kramers-Moyal coefficients and keep only the specified fraction around the stable state
    kmc, edges = km(data_array, bw = bw, bins = bins, powers = powers)
    edges = edges[0]
    km1 = kmc[1]/Delta_t
    km2 = (kmc[2] - np.square(kmc[1])/2)/Delta_t
    center_index = np.searchsorted(edges,center)
    left_index = max(0,center_index - round(bin_number*centerkmfraction/2))
    right_index = min(bin_number-1, center_index + round(bin_number*centerkmfraction/2))
    edges = edges[left_index:right_index]
    km1 = km1[left_index:right_index]
    km2 = km2[left_index:right_index]
    a = km1
    b = np.sqrt(2*km2)
    ### Estimate lambda as the best linear fit to the first order KM coefficient
    lambda_est = -1*stats.linregress(edges,km1)[0]
    return [edges, a, b, lambda_est]

def linear_detrending_nD(data_array, keep_center = False):
    data_array = np.copy(data_array)
    n = data_array.shape[0]
    center = [np.mean(data_array[i]) for i in range(n)]
    for i in range(n):
        fit = np.polyfit(range(len(data_array[i])), data_array[i], 1)
        data_array[i] = data_array[i] - np.array([x * fit[0] + fit[1] for x in range(len(data_array[i]))])
    if keep_center:
        return np.array([data_array[i] + center[i] for i in range(n)])
    else:
        return data_array

def full_km_analysis_nD(data_array, Delta_t, bin_number = bin_number, bw = None, centerkmfraction = centerkmfraction):
    n = data_array.shape[0]
    powers = np.array([[0]*n] + [[0]*i + [1] + [0]*(n-1-i) for i in range(n)])
    bins = (np.ones(n)*bin_number).astype(int)
    data_array = linear_detrending_nD(data_array, keep_center=True)
    stds = np.array([np.std(data_array[i]) for i in range(n)])
    if bw is None:
        bw = n*14/bin_number
    ### Determine stable states
    data_array = np.array([1/stds[i]*(data_array[i]-np.mean(data_array[i]))+np.mean(data_array[i]) for i in range(n)])
    center = np.array([np.mean(data_array[i]) for i in range(n)])
    ### Get estimations of Kramers-Moyal coefficients
    kmc, edges = km(np.transpose(data_array), bw = bw, bins = bins, powers = powers)
    km1 = kmc[1:]/Delta_t
    ### Keep only the specified fraction around the stable state
    for i in range(n):
        center_index = np.searchsorted(edges[i],center[i])
        left_index = max(0,center_index - round(bin_number*centerkmfraction/2))
        right_index = min(bin_number-1, center_index + round(bin_number*centerkmfraction/2))
        edges[i] = edges[i][left_index: right_index]
        km1 = list(km1)
        for j in range(n):
            km1[j] = km1[j].take(indices=range(left_index, right_index),axis=i)
    mesh = np.meshgrid(*edges, indexing="ij")
    X = np.array([[1] + [mesh[i][tupel] for i in range(n)] for tupel in list(itertools.product(*[range(len(mesh[j])) for j in range(n)]))])
    Y = np.array([[km1[i][tupel] for i in range(n)] for tupel in list(itertools.product(*[range(len(mesh[j])) for j in range(n)]))])
    B = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(X),X)),np.transpose(X)),Y)
    B = B[1:,:]
    ### Estimate lambda as the real part of the eigenvalues of the best linear fit to the first order KM coefficient
    eigvals = -1*np.sort(np.real(np.linalg.eigvals(B)))
    return [mesh, km1, eigvals]

def lambda_estimator(data_array, Delta_t = 1, bin_number = bin_number, bw = None, centerkmfraction = centerkmfraction):
    if data_array.ndim == 1:
        return full_km_analysis_1D(data_array, Delta_t, bin_number = bin_number, bw = bw, centerkmfraction = centerkmfraction)[3]
    if data_array.ndim > 1:
        return full_km_analysis_nD(data_array, Delta_t, bin_number = bin_number, bw = bw, centerkmfraction = centerkmfraction)[2]

def variance_estimator(data_array):
    data_array = linear_detrending_1D(data_array)
    return np.sum(np.array([x**2 for x in data_array]))/len(data_array)

def ac1_estimator(data_array, lag=1):
    data_array = linear_detrending_1D(data_array)
    return np.sum(np.array([data_array[i]*data_array[i + lag] for i in range(len(data_array)-lag)]))/(len(data_array)-lag)/variance_estimator(data_array)