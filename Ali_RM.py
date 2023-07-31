# -*- coding: utf-8 -*-
"""
@author: Ali Roshani Moghaddam
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from datetime import datetime

file_path = "C:\\Users\\Lenovo\\Desktop\\Validus\\SPX_Monthly_Option_data_300121_300421.csv"



# Read the CSV file into a pandas DataFrame
df = pd.read_csv(file_path)

# Generate the range of dates from February 1, 2021, to April 30, 2021
start_date = '2021-02-01'
end_date = '2021-04-30'
date_range = pd.date_range(start=start_date, end=end_date)

#################################################################################################
#################################################################################################
##Q1##

# Let's say we want to find the first index of the date '2021-02-10' in the 'Date' column
target_date_str = '2/1/2021'
column_name = 'date'

# Convert the column to pandas datetime objects using the appropriate format
date_format = '%m/%d/%Y'
df[column_name] = pd.to_datetime(df[column_name], format=date_format)

# Convert the target_date_str to a pandas datetime object
target_date = pd.to_datetime(target_date_str, format=date_format)

# Find the first index of the target_date in the specified column
first_occurrence_index = df[df[column_name] == target_date].index[0]

#################################################################################################

# Let's say we want to find the cell(s) in the 'expiration' column that start with '2' or Feb and end with '2021'
# whihc is expiration date (since, we just know that it is the third friday of the month, so this is the only expiration
# date which is in the second month of 2021)
start_with = '02'
end_with = '2021'
filtered_cells = df[(df['date'] == target_date)& df['expiration'].str.startswith(start_with) & df['expiration'].str.endswith(end_with)]
mat_date = filtered_cells.iloc[0]["expiration"]

#################################################################################################

# Strike Computation (0.01 OTM)
OTM = 0.01
Nominal_Strike = (1+OTM)*filtered_cells['adjusted close'].iloc[0]

#################################################################################################

# Let's say we want to find the cell in the 'Strike' column that has the closest value to Nominal Strike
target_value = Nominal_Strike
column_name_1 = 'strike'

# Calculate the absolute difference between each value in the 'Strike' column and the target value
abs_diff = (filtered_cells[column_name_1] - target_value).abs()

# Find the index of the cell with the minimum absolute difference
closest_index = abs_diff.idxmin()

# Access the value in the cell with the closest value
closest_value = filtered_cells.at[closest_index, column_name_1]

#################################################################################################

# Computation of Selling Price
filtered_cells_plus = filtered_cells[(filtered_cells['strike'] == closest_value) & (filtered_cells['call/put'] == "C")]
price = 0.5 * (filtered_cells_plus["bid"] + filtered_cells_plus["ask"])
#################################################################################################

#Computing the MTMs
date_price=[]
price_all = []
MTM =[]

count = 0
for i in date_range:
    date_exists = any(df['date'] == i)
    if date_exists:
        filtered_cells_price = df[(df['date'] == i)]['adjusted close'].iloc[0]
        loc_finder = (df['date'] == i) & df['expiration'].str.startswith(start_with) & df['expiration'].str.endswith(end_with) & (df['strike'] == closest_value) & (df['call/put'] == "C")
        if i <= pd.to_datetime(mat_date):
            filtered_cells_MTM = 0.5 * (df[loc_finder].iloc[0]["bid"] + df[loc_finder].iloc[0]['ask'])
            price_all.append(filtered_cells_price)
            date_price.append(i)
            MTM.append(price-filtered_cells_MTM)
            MTM[count] = MTM[count].iloc[0]
            count = count + 1
        else:
            price_all.append(filtered_cells_price)

            date_price.append(i)
            MTM.append(0)
 

#################################################################################################

data = {
    'Date': date_price,
    'Y': price_all
}
df_1 = pd.DataFrame(data)

# Convert dates to pandas datetime objects
df_1['Date'] = pd.to_datetime(df_1['Date'])


# Plot the polynomial function using matplotlib
plt.figure(figsize=(10, 6))

# Plot data points
plt.plot(df_1['Date'], df_1['Y'], '-', color='blue', markersize=8)


plt.xlabel('Date', fontsize=14)
plt.ylabel('S&P500 Adjusted Closed Day Price', fontsize=14)
plt.title('S&P500 Adjusted Closed Day Price', fontsize=16)
plt.legend(fontsize=12)
plt.tight_layout()

# Save the plot with higher DPI for improved figure quality
plt.savefig('S&P500 Adjusted Closed Day Price.png', dpi=300)

plt.show()
#################################################################################################
data_1 = {
    'Date': date_price,
    'Y': MTM
}
df_2 = pd.DataFrame(data_1)

# Convert dates to pandas datetime objects
df_2['Date'] = pd.to_datetime(df_2['Date'])


# Plot the polynomial function using matplotlib
plt.figure(figsize=(10, 6))

# Plot data points
plt.plot(df_2['Date'], df_2['Y'], '-', color='blue', markersize=8)
#, label='Data Points'

plt.xlabel('Date', fontsize=14)
plt.ylabel('Mark to Market', fontsize=14)
plt.title('Strategy 1 - Mark to Market', fontsize=16)
plt.legend(fontsize=12)
plt.tight_layout()

# Save the plot with higher DPI for improved figure quality
plt.savefig('Strategy 1 - Mark to Market.png', dpi=300)

plt.show()

#################################################################################################
#################################################################################################

# Q2

# Let's say we want to find the cell(s) in the 'expiration' column that start with '3' or Mar and end with '2021'
# whihc is expiration date (since, we just know that it is the thirs friday of the month)
start_with_2 = '03'
end_with = '2021'
filtered_cells_2 = df[(df['date'] == target_date)& df['expiration'].str.startswith(start_with_2) & df['expiration'].str.endswith(end_with)]
mat_date_2 = filtered_cells_2.iloc[0]["expiration"]

#################################################################################################

# Let's say we want to find the cell in the 'Strike' column that has the closest value to Nominal Strike
# Calculate the absolute difference between each value in the 'Strike' column and the target value
abs_diff_2 = (filtered_cells_2[column_name_1] - target_value).abs()

# Find the index of the cell with the minimum absolute difference
closest_index_2 = abs_diff_2.idxmin()

# Access the value in the cell with the closest value
closest_value_2 = filtered_cells_2.at[closest_index_2, column_name_1]

#################################################################################################

# Computation of Selling Price
filtered_cells_plus_2 = filtered_cells_2[(filtered_cells_2['strike'] == closest_value_2) & (filtered_cells_2['call/put'] == "C")]
price_2 = 0.5 * (filtered_cells_plus_2["bid"] + filtered_cells_plus_2["ask"])

#################################################################################################

#Computing the MTMs
date_price_2=[]
price_all_2 = []
MTM_2 =[]

count = 0
for i in date_range:
    date_exists = any(df['date'] == i)
    if date_exists:
        filtered_cells_price_2 = df[(df['date'] == i)]['adjusted close'].iloc[0]
        loc_finder = (df['date'] == i) & df['expiration'].str.startswith(start_with_2) & df['expiration'].str.endswith(end_with) & (df['strike'] == closest_value_2) & (df['call/put'] == "C")
        if i <= pd.to_datetime(mat_date_2):
            filtered_cells_MTM_2 = 0.5 * (df[loc_finder].iloc[0]["bid"] + df[loc_finder].iloc[0]['ask'])
            price_all_2.append(filtered_cells_price_2)
            date_price_2.append(i)
            MTM_2.append(price_2-filtered_cells_MTM_2)
            MTM_2[count] = MTM_2[count].iloc[0]
            count = count + 1
        else:
            price_all_2.append(filtered_cells_price_2)
            date_price_2.append(i)
            MTM_2.append(0)

print(len(date_price_2))
print(len(price_all_2))
print(len(MTM_2))
                        

#################################################################################################
data_1 = {
    'Date': date_price_2,
    'Y': MTM_2
}
df_2 = pd.DataFrame(data_1)

# Convert dates to pandas datetime objects
df_2['Date'] = pd.to_datetime(df_2['Date'])


# Plot the polynomial function using matplotlib
plt.figure(figsize=(10, 6))

# Plot data points
plt.plot(df_2['Date'], df_2['Y'], '-', color='blue', markersize=8)
#, label='Data Points'

plt.xlabel('Date', fontsize=14)
plt.ylabel('Mark to Market', fontsize=14)
plt.title('Strategy 2 - Mark to Market', fontsize=16)
plt.legend(fontsize=12)
plt.tight_layout()

# Save the plot with higher DPI for improved figure quality
plt.savefig('Strategy 2 - Mark to Market.png', dpi=300)

plt.show()

#################################################################################################
#################################################################################################

# Q3

# Let's say we want to find the cell(s) in the 'expiration' column that start with '3' or Mar and end with '2021'
# whihc is expiration date (since, we just know that it is the thirs friday of the month)
start_with_2 = '03'
end_with = '2021'
filtered_cells_3 = df[(df['date'] == target_date)& df['expiration'].str.startswith(start_with_2) & df['expiration'].str.endswith(end_with)]
mat_date_3 = filtered_cells_3.iloc[0]["expiration"]

#################################################################################################
# Strike Computation (0.02 OTM)
OTM_p = 0.02
Nominal_Strike_p = (1 + OTM_p)*filtered_cells['adjusted close'].iloc[0]

#################################################################################################

# Let's say we want to find the cell in the 'Strike' column that has the closest value to Nominal Strike
target_value = Nominal_Strike_p
column_name_1 = 'strike'

# Calculate the absolute difference between each value in the 'Strike' column and the target value
abs_diff_3 = (filtered_cells_3[column_name_1] - target_value).abs()

# Find the index of the cell with the minimum absolute difference
closest_index_3 = abs_diff_3.idxmin()

# Access the value in the cell with the closest value
closest_value_3 = filtered_cells_3.at[closest_index_3, column_name_1]

#################################################################################################

# Computation of Selling Price
filtered_cells_plus_3 = filtered_cells_3[(filtered_cells_3['strike'] == closest_value_3) & (filtered_cells_3['call/put'] == "C")]
price_3 = 0.5 * (filtered_cells_plus_3["bid"] + filtered_cells_plus_3["ask"])

#################################################################################################

#Computing the MTMs
date_price_2=[]
price_all_2 = []
MTM_3 =[]

count = 0
for i in date_range:
    date_exists = any(df['date'] == i)
    if date_exists:
        filtered_cells_price_2 = df[(df['date'] == i)]['adjusted close'].iloc[0]
        loc_finder = (df['date'] == i) & df['expiration'].str.startswith(start_with_2) & df['expiration'].str.endswith(end_with) & (df['strike'] == closest_value_3) & (df['call/put'] == "C")
        if i <= pd.to_datetime(mat_date_3):
            filtered_cells_MTM_2 = 0.5 * (df[loc_finder].iloc[0]["bid"] + df[loc_finder].iloc[0]['ask'])
            price_all_2.append(filtered_cells_price_2)
            date_price_2.append(i)
            MTM_3.append(-price_3 + filtered_cells_MTM_2)
            MTM_3[count] = MTM_3[count].iloc[0]
            count = count + 1
        else:
            price_all_2.append(filtered_cells_price_2)
            date_price_2.append(i)
            MTM_3.append(0)

print(len(date_price_2))
print(len(price_all_2))
print(len(MTM_3))

MTM_4 = []
for i in range(len(MTM)):
    MTM_4.append(MTM_2[i]+MTM_3[i])
                           

#################################################################################################
data_1 = {
    'Date': date_price_2,
    'Y': MTM_4
}
df_2 = pd.DataFrame(data_1)

# Convert dates to pandas datetime objects
df_2['Date'] = pd.to_datetime(df_2['Date'])


# Plot the polynomial function using matplotlib
plt.figure(figsize=(10, 6))

# Plot data points
plt.plot(df_2['Date'], df_2['Y'], '-', color='blue', markersize=8)
#, label='Data Points'

plt.xlabel('Date', fontsize=14)
plt.ylabel('Mark to Market', fontsize=14)
plt.title('Strategy 3 - Mark to Market', fontsize=16)
plt.legend(fontsize=12)
plt.tight_layout()

# Save the plot with higher DPI for improved figure quality
plt.savefig('Strategy 3 - Mark to Market.png', dpi=300)

plt.show()

#################################################################################################
#################################################################################################

# Q4

delta = []

for i in range(0, 63):
    if i == 0:
        delta.append(0)
    else:
        delta.append((MTM_4[i] - MTM_4[i-1])/(price_all_2[i] - price_all_2[i-1]))

print(len(delta))

data_1 = {
    'Date': date_price_2,
    'Y': delta
}
df_2 = pd.DataFrame(data_1)

# Convert dates to pandas datetime objects
df_2['Date'] = pd.to_datetime(df_2['Date'])


# Plot the polynomial function using matplotlib
plt.figure(figsize=(10, 6))

# Plot data points
plt.plot(df_2['Date'], df_2['Y'], '-', color='blue', markersize=8)
#, label='Data Points'

plt.xlabel('Date', fontsize=14)
plt.ylabel('Delta', fontsize=14)
plt.title('Strategy 3 - Delta', fontsize=16)
plt.legend(fontsize=12)
plt.tight_layout()

# Save the plot with higher DPI for improved figure quality
plt.savefig('Strategy 3 - Delta.png', dpi=300)

plt.show()

#################################################################################################
#################################################################################################

# Q5


N = norm.cdf

def bs_call(S, K, T, r, vol):
    d1 = (np.log(S/K) + (r + 0.5*vol**2)*T) / (vol*np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    return S * norm.cdf(d1) - np.exp(-r * T) * K * norm.cdf(d2)

def bs_vega(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)

def find_vol(target_value, S, K, T, r, *args):
    MAX_ITERATIONS = 500
    PRECISION = 1.0e-5
    sigma = 0.5
    for i in range(0, MAX_ITERATIONS):
        price = bs_call(S, K, T, r, sigma)
        vega = bs_vega(S, K, T, r, sigma)
        diff = target_value - price  # our root
        if (abs(diff) < PRECISION):
            return sigma
        sigma = sigma + diff/vega # f(x) / f'(x)
    return sigma # value wasn't found, return best guess so far

def days_between_dates(date1, date2):
    # Convert the string date to a Pandas Timestamp object
    if isinstance(date1, str):
        date1 = pd.to_datetime(date1)
        
    # Convert the Timestamp date to a datetime object
    if isinstance(date2, pd.Timestamp):
        date2 = date2.to_pydatetime()
    
    # Calculate the difference between the two dates
    delta = date1 - date2
    
    # Extract the number of days from the timedelta object
    return delta.days

print(days_between_dates(mat_date_2, date_price_2[0]))


        
imp_vol = []

for i in range (0, len(MTM_2)):
    if MTM_2[i] == 0:
        imp_vol.append(0)
    else:
        S = price_all_2[i]              
        K = closest_value_2
        T = (days_between_dates(mat_date_2, date_price_2[i]))/365
        r = 0.0
        V_market = price_2.iloc[0] - MTM_2[i]
        imp_vol.append(find_vol(V_market, S, K, T, r))
    print(i)
    print (imp_vol[i])



data_1 = {
    'Date': date_price_2,
    'Y': imp_vol
}
df_2 = pd.DataFrame(data_1)

# Convert dates to pandas datetime objects
df_2['Date'] = pd.to_datetime(df_2['Date'])


# Plot the polynomial function using matplotlib
plt.figure(figsize=(10, 6))

# Plot data points
plt.plot(df_2['Date'], df_2['Y'], '-', color='blue', markersize=8)
#, label='Data Points'

plt.xlabel('Date', fontsize=14)
plt.ylabel('Implied Volatility', fontsize=14)
plt.title('Strategy 2 - Implied Volatility', fontsize=16)
plt.legend(fontsize=12)
plt.tight_layout()

# Save the plot with higher DPI for improved figure quality
plt.savefig('Strategy 2 - Implied Volatility.png', dpi=300)

plt.show()


#################################################################################################
#################################################################################################

##Q6##
 
    
    
    
    
data_1 = {
    'Date': date_price,
    'Y1': MTM,
    'Y2': MTM_2
    
}
df_2 = pd.DataFrame(data_1)

# Convert dates to pandas datetime objects
df_2['Date'] = pd.to_datetime(df_2['Date'])


# Plot the polynomial function using matplotlib
plt.figure(figsize=(10, 6))

# Plot data points
plt.plot(df_2['Date'], df_2['Y1'], '-', color='blue', markersize=8)
plt.plot(df_2['Date'], df_2['Y2'], '-', color='green', markersize=8)

#, label='Data Points'

plt.xlabel('Date', fontsize=14)
plt.ylabel('Mark to Market', fontsize=14)
plt.title('Strategy 1 and 2 MTM Comparison', fontsize=16)
plt.legend(fontsize=12)
plt.tight_layout()

# Save the plot with higher DPI for improved figure quality
plt.savefig('Strategy 1 and 2 MTM Comparison.png', dpi=300)

plt.show()








# Sample data for the two curves
data_1 = {
    'Date': date_price,
    'Y1': MTM,
    'Y2': price_all_2
    
}
df_2 = pd.DataFrame(data_1)

# Convert dates to pandas datetime objects
df_2['Date'] = pd.to_datetime(df_2['Date'])

# Create the first plot
fig, ax1 = plt.subplots()
ax1.plot(df_2['Date'], df_2['Y1'], 'b-', label='MTM', markersize=8)
ax1.set_xlabel('Date', fontsize=14)
ax1.set_ylabel('MTM', color='b', fontsize=14)
ax1.tick_params('y', colors='b')

# Create the second plot with a new y-axis
ax2 = ax1.twinx()
ax2.plot(df_2['Date'], df_2['Y2'], 'r-', label='Index Price', markersize=8)
ax2.set_ylabel('Index Price', color='r', fontsize=14)
ax2.tick_params('y', colors='r')

# Add legends and title
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.title('Strategy 3')


plt.savefig('Strategy 3.png', dpi=300)

# Show the plot
plt.show()

#In this code, we first create the primary plot with ax1.plot() and then create a second y-axis using ax1.twinx(). The second plot is then added using ax2.plot(). We set the color and label for each curve's y-axis using set_ylabel() and specify the tick colors with tick_params().

#Make sure to adjust the data and labels according to your specific use case. This example will plot two curves, a sine curve (y1) and an exponential curve (y2), on the same plot with separate y-axes.





