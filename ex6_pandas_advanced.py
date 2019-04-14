# -*- coding: utf-8 -*-
"""
Created on Sun April 1 14:47:15 2019

@author: Leora Betesh
"""
#1. APPLY
#Special thanks to: https://github.com/justmarkham for sharing the dataset and materials
#United States - Crime Rates - 1960 - 2014
# Step 1. Import the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 6)
pd.set_option('display.width', 500)
# Step 2. Import the dataset from this
#https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/04_Apply/US_Crime_Rate
#s/US_Crime_Rates_1960_2014.csv
# Step 3. Assign it to a variable called crime.
crime = pd.read_csv("https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/04_Apply/US_Crime_Rates/US_Crime_Rates_1960_2014.csv")
# Step 4. What is the type of the columns?
print(crime.dtypes)
# Have you noticed that the type of Year is int64. But pandas has a different type to work with
#Time Series. Let's see it now.
# Step 5. Convert the type of the column Year to datetime64
crime['Year'] = pd.to_datetime(crime['Year'])
# Step 6. Set the Year column as the index of the dataframe
crime.set_index('Year')
# Step 7. Delete the Total column
crime = crime.drop(columns='Total')
# Step 8. Group the year by decades and sum the values
crime_decade = crime.groupby(np.floor(crime['Year']/10) * 10)[['Violent', 'Property', 'Murder',
       'Forcible_Rape', 'Robbery', 'Aggravated_assault', 'Burglary',
       'Larceny_Theft', 'Vehicle_Theft']].sum()

print(crime_decade)

# Pay attention to the Population column number, summing this column is a mistake
# Step 9. What is the mos dangerous decade to live in the US?
print(crime_decade.sum(axis = 1).sort_values())

#2. STATS -We are going to use a subset of [US Baby Names](https://www.kaggle.com/kaggle/us-baby-names)
#In the file it will be names from 2004 until 2014
# Step 1. Import the necessary libraries
# Step 2. Download and extract the data from this link
# https://drive.google.com/open?id=1At_YCsguCIEqe3l-gSPPquaOORbGLKmj
#Step 3. Assign it to a variable called baby_names.
baby_names = pd.read_csv("NationalNames.csv")
# Step 4. See the first 10 entries
baby_names.head(10)
#0' and 'Id'
# Step 6. Are there more male or female names in the dataset?
print(baby_names.groupby(['Gender'])['Name'].count())
# Step 7. Group the dataset by name and assign to names
names = baby_names.groupby('Name')
# Step 8. How many different names exist in the dataset?
print(names.count())
# Step 9. What is the name with most occurrences?
names = baby_names.groupby('Name')['Count'].sum().to_frame()
print(names.idxmax())
# Step 10. How many different names have the least occurrences?
names_min = names[names.Count ==names.Count.min()]
names_min.count()
# Step 11. What is the median name occurrence?
names[names.Count ==names.Count.median()].count()
# Step 12. What is the standard deviation of names?
names.std()
# Step 13. Get a summary with the mean, min, max, std and quartiles.
names.describe()
#3. Visualization
# Step 1. Import the necessary libraries.  set this so the graphs open internally
# Step 2. Import the dataset from this https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv
# Step 3. Assign it to a variable called chipo.
chipo = pd.read_csv('https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv', sep='\t')
# Step 4. See the first 10 entries
chipo.head()
# Step 5. Create a histogram of the top 5 items bought
item_nums = chipo.groupby('item_name')['order_id'].count()
top5 = item_nums.sort_values(ascending=False).head(5).to_frame()
names = list(top5.index)
values = top5['order_id']
plt.bar(names,values)
plt.show()
plt.hist(top5,bins=2)
plt.show()

# Step 6. Create a scatterplot with the number of items orderered per order price\n,
#Price should be in the X-axis and Items ordered in the Y-axis
#Create a question and a graph to answer your own question.
#chipo['item_price'] = chipo['item_price'].str.replace(r'\$', '').astype(float)
items_per_price = chipo.groupby('item_price')['quantity'].sum().to_frame()
plt.scatter(items_per_price.index, items_per_price['quantity'])
plt.title('Scatter plot pythonspot.com')
plt.xlabel('Price')
plt.ylabel('Quantity purchased')
plt.show()


#4. Creating Series and DataFrames
#This time you will create the data
# Step 1. Import the necessary libraries
# Step 2. Create a data dictionary that looks like the DataFrame below
dict1 = {"name": ['Bulbasaur', 'Charmander','Squirtle','Caterpie'],
            "evolution": ['Ivysaur','Charmeleon','Wartortle','Metapod'],
            "type": ['grass', 'fire', 'water', 'bug'],
            "hp": [45, 39, 44, 45],
            "pokedex": ['yes', 'no','yes','no']}
# Step 3. Assign it to a variable called
# Step 4. Ops...it seems the DataFrame columns are in alphabetical order. Place the order of the
#columns as name, type, hp, evolution, pokedex
ed = pd.DataFrame.from_dict(dict1)
ed = ed[['name','type','hp','evolution','pokedex']]
# Step 5. Add another column called place, and insert what you have in mind.
ed['place'] = ['Africa','Asia','Europe','Antarctica']
# Step 6. Present the type of each column
print(ed.dtypes)
#Create your own question and answer it.

#5 Time_Series
# Step 1. Import the necessary libraries
# Step 2. Import the dataset from this
#https://raw.githubusercontent.com/datasets/investor-flow-of-funds-us/master/data/weekly.csv
# Step 3. Assign it to a variable called
timeSeries = pd.read_csv("https://raw.githubusercontent.com/datasets/investor-flow-of-funds-us/master/data/weekly.csv")
# Step 4. What is the frequency of the dataset?
# Step 5. Set the column Date as the index.
timeSeries.set_index('Date')
# Step 6. What is the type of the index?
print(timeSeries.index.dtype)
# Step 7. Set the index to a DatetimeIndex type
timeSeries['DateTime'] = pd.to_datetime(timeSeries['Date'])
timeSeries.index = timeSeries['DateTime']
# Step 8. Change the frequency to monthly, sum the values and assign it to monthly.
ts_monthly = timeSeries.resample('M').sum()
# Step 9. You will notice that it filled the dataFrame with months that don't have any data with
#NaN. Let's drop these rows.
ts_concise = ts_monthly[(ts_monthly.T != 0).any()]
# Step 10. Good, now we have the monthly data. Now change the frequency to year.
ts_annual = ts_monthly.resample('Y').sum()
#Create your own question and answer it.
