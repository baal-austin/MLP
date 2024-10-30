import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('input_data.csv')

# Calculate the IQR for 'Irradiance'
Q1 = df['Irradiance'].quantile(0.25)
Q3 = df['Irradiance'].quantile(0.75)
IQR = Q3 - Q1

# Define outlier boundaries
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers = df[(df['Irradiance'] < lower_bound) | (df['Irradiance'] > upper_bound)]
outlier_count = outliers.shape[0]
total_count = df.shape[0]

# Output results
print(f'Total count of Irradiance values: {total_count}')
print(f'Number of outliers in Irradiance values: {outlier_count}')


## 啊啊
# Set the figure size
plt.figure(figsize=(12, 8))

# Draw box plots for each numerical column
sns.boxplot(data=df, orient='h')  # Horizontal orientation for better visibility

# Set title and labels
plt.title('Box Plots of All Attributes', fontsize=16)
plt.xlabel('Value', fontsize=14)
plt.ylabel('Attributes', fontsize=14)

# Show the plot
plt.tight_layout()
plt.show()
