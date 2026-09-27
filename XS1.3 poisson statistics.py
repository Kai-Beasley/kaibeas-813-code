import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.stats import chi2, norm

#import data
with open("22-9 cosmic watch top.txt","r", encoding="utf-8") as file:
	lines = file.readlines()[7:]

#split nicely into array
data = [line.split() for line in lines if line.strip()]

#create dataframe with column names
data = pd.DataFrame(data, columns=["Event", "Time", "Date", "TimeStamp[ms]", "ADC1", "ADC2", "SiPM[mV]", 
								   "Temp[C]", "Pressure[Pa]", "DeadTime[us]", "Coincident ID","location?"])

##############

# Specify the inclusive range of ten-second intervals to analyze.
first_interval = 0 #starts from 0
last_interval = 99 #note the largest acceptable last interval is 42261 for 10-second intervals, which corresponds 4 days, 21 hours, 23 minutes, and 30 seconds
interval_count = last_interval - first_interval + 1
interval_duration = 100  # length of interval, in seconds

##############

#convert the date and time information into a raw number of seconds since the first timestamp  (this is AI code)
timestamps = pd.to_datetime(data["Date"].astype(str) + " " + data["Time"].astype(str), errors="coerce",dayfirst=True)

elapsed_seconds = (timestamps - timestamps.min()).dt.total_seconds()
interval_numbers = (elapsed_seconds // interval_duration).dropna().astype(int)
interval_numbers = interval_numbers[(interval_numbers >= first_interval) & (interval_numbers <= last_interval)]
counts = interval_numbers.value_counts().reindex(range(first_interval, last_interval + 1), fill_value=0).sort_index()

#calculate a running mean for the data
running_mean = [counts.values[:i+1].mean() for i in range(len(counts.values))]
#the error in the running mean is the standard deviation of the counts divided by the square root of the number of intervals
running_error = [counts.values[:i+1].std() / ((i + 1) ** 0.5) for i in range(len(counts.values))]
print(running_mean[-1])
print(running_error[-1])

plt.scatter(counts.index, counts.values, s=5, alpha=0.75,label="Counts per Interval") #plot the counts per interval
plt.errorbar(counts.index, running_mean, yerr=running_error, color="orange", label="Running Mean", capsize=1) #plot the running mean with error bars


#chart appearance parameters
plt.xlabel("Interval Number")
plt.ylabel("Event Count")
plt.title(f"Events per {interval_duration}-Second Interval")
plt.xticks(range(first_interval, last_interval + 1, max(1, int(round(interval_count / 10, -1)))))
plt.grid(True)
plt.annotate(f"Mean: {running_mean[-1]:.2f} \nStandard Deviation: {running_error[-1]:.2f} \nTotal Intervals: {interval_count}", 
			 (first_interval + 2, max(counts.values) * 0.9), fontsize=10)
#plt.tight_layout()
plt.legend(loc='upper right')
#plt.show()


# Plot a histogram of counts per interval for the specified inclusive interval range.
plt.figure()
bin_width =np.round(np.log10(interval_duration)) #bin width is log10 of duration, centered on the integer counts
histogram_bins = np.arange(counts.min() - 0.5, counts.max() + 1.5, bin_width) 
histogram_counts, bin_edges, _ = plt.hist(counts.to_numpy(), bins=histogram_bins, edgecolor='black', alpha=0.75)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
#add error bars
histogram_errors = np.where(histogram_counts == 0, 1, np.sqrt(histogram_counts))
plt.errorbar(bin_centers, histogram_counts, yerr=histogram_errors, fmt="none", ecolor="black", capsize=1)

# Poisson prediction for each count bin, using the mean count per interval.
poisson_mean = running_mean[-1]

#equivalent to the standard formula, but using the log-gamma function to avoid overflow for large factorials
poisson_counts = interval_count * np.exp(
	np.array([
		-poisson_mean + int(count) * np.log(poisson_mean) - math.lgamma(int(count) + 1)
		for count in bin_centers
	]))

plt.plot(bin_centers, poisson_counts, "o-", color="red", label="Poisson prediction",markersize=3)


# calculate the chi2 and related statistics for this fit 
valid_bins = (poisson_counts > 0)
observed_counts = histogram_counts[valid_bins].astype(float)
expected_counts = poisson_counts[valid_bins].astype(float)
errors = histogram_errors[valid_bins].astype(float)

chi_squared = np.sum((observed_counts - expected_counts) ** 2 / errors) #error squared is equal to the observed counts
# One fitted parameter (the Poisson mean) was estimated from the same data.
dof = max(len(expected_counts) - 1 - 1, 1)
chi_squared_per_dof = chi_squared / dof
p_value = chi2.sf(chi_squared, dof)

# Gaussian prediction for the same count bins, fitted using the sample mean and standard deviation.


gaussian_mean = counts.mean()
gaussian_std = counts.std(ddof=1)
# Convert the Gaussian density to expected counts per histogram bin.
gaussian_counts = interval_count * norm.pdf(bin_centers, loc=gaussian_mean, scale=gaussian_std) * bin_width
plt.plot(bin_centers, gaussian_counts, "s--", color="blue", label="Gaussian prediction", markersize=3)

# Calculate chi-squared statistics for the Gaussian fit.
gaussian_valid_bins = gaussian_counts > 0
gaussian_observed = histogram_counts[gaussian_valid_bins].astype(float)
gaussian_expected = gaussian_counts[gaussian_valid_bins].astype(float)
gaussian_errors = histogram_errors[gaussian_valid_bins].astype(float)
gaussian_chi_squared = np.sum(
	(gaussian_observed - gaussian_expected) ** 2 / gaussian_errors ** 2
)
# Two fitted parameters (mean and standard deviation) were estimated from the data.
gaussian_dof = max(len(gaussian_expected) - 2 - 1, 1)
gaussian_chi_squared_per_dof = gaussian_chi_squared / gaussian_dof
gaussian_p_value = chi2.sf(gaussian_chi_squared, gaussian_dof)

plt.annotate( #add the chi2 and p-value to the plot
	f"Poisson:\n$\\chi^2$: {chi_squared:.2f}\n$\\chi^2/dof$: {chi_squared_per_dof:.2f}\n$p$: {p_value:.3g}\n\n"
	f"Gaussian:\n$\\chi^2$: {gaussian_chi_squared:.2f}\n$\\chi^2/dof$: {gaussian_chi_squared_per_dof:.2f}\n$p$: {gaussian_p_value:.3g}",
    xy=(0.98, 0.8),xycoords="axes fraction",ha="right",va="top",fontsize=10)



#plot appearance parameters
plt.xlabel(f"Event Count per {interval_duration}-Second Interval")
plt.ylabel("Number of Intervals")
plt.title(f"Histogram of Counts per {interval_duration}-Second Interval")
plt.grid(True, axis="y",alpha=0.5)
plt.legend()


plt.show()


