import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import chi2

set_1 = pd.DataFrame({
    'x': [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
    'y': [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]})
set_2 = pd.DataFrame({
    'x': [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
    'y': [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]})
set_3 = pd.DataFrame({
    'x': [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
    'y': [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]})
set_4 = pd.DataFrame({
    'x': [8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8],
    'y': [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89]})

data_sets = [set_1, set_2, set_3, set_4]

#########
sigma = 2.5 #specify uncertainty here - either 1.24, 0.75, or 2.5
#########

uncertainties = sigma * np.ones_like(set_1['y'])  # create an array of uncertainties for set 1
fig, ax = plt.subplots(2, 2, figsize=(10, 8)) #set up a 2x2 grid of subplots for the four datasets

def model(x,a,b):
    return a*x + b
guess = [1, 0] # initial guess for the fit parameters


for i, set in enumerate(data_sets): #plot data
    x = set['x']
    y = set['y']
    ax[i//2, i%2].set_title(f'Set {i+1}')
    ax[i//2, i%2].errorbar(x, y, yerr=uncertainties, fmt='o', color='blue')

    # fit the linear function to the data and plot it
    popt, pcov = curve_fit(model, x, y, p0=guess, sigma=uncertainties, absolute_sigma=True) 
    ax[i//2, i%2].plot(x, model(x, *popt), color='red', label=fr'Fit: $ y = ( {popt[0]:.2f} \pm {np.sqrt(pcov[0, 0]):.2f})x + ({popt[1]:.2f} \pm {np.sqrt(pcov[1, 1]):.2f}) $')
    
    #find chi2 and p-value for the fit and annotate it on the plot
    residuals = y - model(x, *popt)
    chi_squared = np.sum((residuals / uncertainties) ** 2)
    dof = len(y) - len(popt)
    p_value = chi2.sf(chi_squared, dof)
    ax[i//2, i%2].annotate(f'Chi^2: {chi_squared:.2f}\nChi^2/DOF: {chi_squared/dof:.2f}\np-value: {p_value:.3f}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=10, verticalalignment='top')
    ax[i//2, i%2].legend(loc='lower right')
    

plt.tight_layout()
plt.show()
 
