import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Generate a date range
start_date = '2022-01-01'
end_date = '2022-01-31'
dates = pd.date_range(start=start_date, end=end_date, freq='H')

# Generate a sine wave with mean 5 and seasonal period of 24
amplitude = 1  # Amplitude of the sine wave
mean = 5  # Mean of the sine wave
seasonal_period = 24  # Seasonal period of the sine wave
values = mean + amplitude * np.sin(2 * np.pi * np.arange(len(dates)) / seasonal_period)

# Create a DataFrame with the date and values
df = pd.DataFrame({'Date': dates, 'Value': values})

# Set the date as the index
df.set_index('Date', inplace=True)

# Plot the time-series data
plt.plot(df.index, df['Value'])
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Time-Series Data')
plt.show()

# Save the DataFrame as a CSV file
df.to_csv('C:\\Users\\Manas\\Desktop\\TimeSeriesVisualizer\\data\\SampleSinusoid_Period24.csv')
