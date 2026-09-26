import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
last_interval = 42261 #note the largest acceptable last interval is 42261, which corresponds 4 days, 21 hours, 23 minutes, and 30 seconds
interval_count = last_interval - first_interval + 1

##############

#convert the date and time information into a raw number of seconds since the first timestamp  (this is AI code)
timestamps = pd.to_datetime(data["Date"].astype(str) + " " + data["Time"].astype(str), errors="coerce",dayfirst=True)

elapsed_seconds = (timestamps - timestamps.min()).dt.total_seconds()
interval_numbers = (elapsed_seconds // 10).dropna().astype(int)
interval_numbers = interval_numbers[(interval_numbers >= first_interval) & (interval_numbers <= last_interval)]
counts = interval_numbers.value_counts().reindex(range(first_interval, last_interval + 1), fill_value=0).sort_index()

#calculate a running mean for the data
running_mean = [counts.values[:i+1].mean() for i in range(len(counts.values))]
#the error in the running mean is the standard deviation of the counts divided by the square root of the number of intervals
running_error = [counts.values[:i+1].std() / ((i + 1) ** 0.5) for i in range(len(counts.values))]
print(running_mean[-1])
print(running_error[-1])

plt.scatter(counts.index, counts.values, s=5, alpha=0.75,label="Counts per Interval") #plot the counts per 10-seocnd interval
plt.errorbar(counts.index, running_mean, yerr=running_error, color="orange", label="Running Mean", capsize=1) #plot the running mean with error bars


#chart appearance parameters
plt.xlabel("Interval Number")
plt.ylabel("Event Count")
plt.title("Events per 10-Second Interval")
plt.xticks(range(first_interval, last_interval + 1, max(1, int(round(interval_count / 10, -1)))))
plt.grid(True)
plt.annotate(f"Mean: {running_mean[-1]:.2f} \nStandard Deviation: {running_error[-1]:.2f} \nTotal Intervals: {interval_count}", 
			 (first_interval + 2, max(counts.values) * 0.9), fontsize=10)
#plt.tight_layout()
plt.legend(loc='upper right')
plt.show()


# Plot a histogram of the distribution of counts per interval.
plt.figure()
histogram_bins = [x - 0.5 for x in range(int(counts.values.min()), int(counts.values.max()) + 2)]
plt.hist(counts.values, bins=histogram_bins, edgecolor='black', alpha=0.75)

# Overlay a Gaussian with the same mean and standard deviation as the data.
data_mean = counts.values.mean()
data_std = counts.values.std()
x = np.linspace(counts.values.min(), counts.values.max(), 500)
gaussian = np.exp(-0.5 * ((x - data_mean) / data_std) ** 2) / (data_std * np.sqrt(2 * np.pi))
gaussian *= len(counts.values) * (histogram_bins[1] - histogram_bins[0])
plt.plot(x, gaussian, color="red", linewidth=2, label="Gaussian fit")

plt.xlabel("Event Count per 10-Second Interval")
plt.ylabel("Number of Intervals")
plt.title("Histogram of Counts per 10-Second Interval")
plt.grid(True, axis="y")
plt.legend()


plt.show()


