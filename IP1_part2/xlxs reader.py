import pandas as pd

# Load the Excel file into a pandas DataFrame
df = pd.read_excel('kwh pricing from18.07 to 27.08 2022.xlsx')

# Extract day, month, and hour from the first column
df['Day'] = pd.to_datetime(df.iloc[:, 0], format='%d/%m/%Y %H:%M:%S').dt.day
df['Month'] = pd.to_datetime(df.iloc[:, 0], format='%d/%m/%Y %H:%M:%S').dt.month
df['Hour'] = pd.to_datetime(df.iloc[:, 0], format='%d/%m/%Y %H:%M:%S').dt.hour

# Extract the price from the second column
df['Value'] = df.iloc[:, 1].str.replace('€', '').str.replace(' ', '').str.replace(',', '.').astype(float)/1000

# Select only the relevant columns
df = df[['Day', 'Month', 'Hour', 'Value']]
day=19
month=7
# Filter the dataframe to include only rows between 10pm and 7am the following day
df_filtered = df[((df['Hour'] >= 22) & (df['Day'] == day) & (df['Month'] == month)) | ((df['Hour'] <= 7) & (df['Day'] == day+1) & (df['Month'] == month))]

# Compute the sum of the filtered prices column
sum_prices = df_filtered['Value'].sum()
# Print the resulting DataFrame
print(sum_values)

