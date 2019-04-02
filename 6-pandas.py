# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 14:47:15 2019

@author: Leora Betesh
"""
#1. Getting & Knowing Your Data
#Step 1. Import the necessary libraries
import pandas as pd
import numpy as np

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 6)
pd.set_option('display.width', 500)

#Step 2. Import the dataset from this https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv
#Step 3. Assign it to a variable called chipo.
chipo = pd.read_csv('https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv', sep='\t')
#Step 4. See the first 10 entries
print(chipo.head(10))
#Step 5. What is the number of observations in the dataset?
print("observations", chipo.shape[0])
#Step 6. What is the number of columns in the dataset?
print("cols: ", len(chipo.columns))
#Step 7. Print the name of all the columns.
print(chipo.columns.values)
#Step 8. How is the dataset indexed?
print("Dataset index info:",chipo.index)
#Step 9. Which was the most ordered item?
print(chipo.groupby('item_name').count()['quantity'].idxmax())
#Step 10. How many items were ordered?
print("number of orders\n",chipo.groupby('order_id').count()['quantity'])
#Step 11. What was the most ordered item in the choice_description column?
print(chipo.groupby('choice_description').count()['quantity'].sort_values(ascending=False).idxmax())
#Step 12. How many items were orderd in total?
print("number of items ordered", chipo['quantity'].sum())
#Step 13. Turn the item price into a float
chipo['item_price'] = chipo['item_price'].str.replace(r'\$', '').astype(float)
#Step 14. How much was the revenue for the period in the dataset?
revenue = chipo['item_price'] * chipo['quantity']
print("total revenue ${0}".format(revenue.sum()))
#Step 15. How many orders were made in the period?
print("num of orders",chipo['order_id'].nunique())
#Step 16. What is the average amount per order?
chipo['totalOrderAmount'] = chipo['item_price'] * chipo['quantity']
print("average amount per order",chipo.groupby('order_id')['totalOrderAmount'].sum().mean())
#Step 17. How many different items are sold?
print("num of item choices",chipo['item_price'].nunique())
#2. Filtering & Sorting
#Step 1. Import the necessary libraries
#Step 2. Import the dataset from this https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv
#Step 3. Assign it to a variable called chipo.
#Step 4. How many products cost more than $10.00?
expensive = chipo.query('item_price > 10').groupby('item_name')
print("products with price > 10",len(expensive))
#Step 5. What is the price of each item?
print("price per item",chipo.groupby('item_name')['item_price'].mean())
#print a data frame with only two columns item_name and item_price
print(chipo[['item_name','item_price']])
#Step 6. Sort by the name of the item
print(chipo['item_name'].sort_values())
print(chipo['item_price'].sort_values())
#Step 7. What was the quantity of the most expensive item ordered?
max_price = chipo['item_price'].max()
chipo_max_item = chipo.loc[(chipo["item_price"]==max_price),:]
print("quantity ordered of most expensive item",chipo_max_item["quantity"])
#Step 8. How many times were a Veggie Salad Bowl ordered?
print("Number of Veggie Salad Bowls ordered:",chipo.loc[chipo['item_name']=='Veggie Salad Bowl',:]['quantity'].count())
#Step 9. How many times people orderd more than one Canned Soda?
canned_soda = chipo.loc[chipo['item_name']=='Canned Soda',:]
print(canned_soda.loc[canned_soda['quantity'] > 1,'quantity'].count())
#3. Grouping
#Step 1. Import the necessary libraries
#Step 2. Import the dataset from this https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user
#Step 3. Assign it to a variable called users.
users = pd.read_csv('https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user', sep='|')
#Step 4. Discover what is the mean age per occupation
print(users.groupby('occupation').mean()['age'])
#Step 5. Discover the Male ratio per occupation and sort it from the most to the least
users_by_occ = users.groupby(['occupation','gender'])['user_id'].count()
occ_ratios = users_by_occ / users_by_occ.groupby(level=0).sum() * 100
print(occ_ratios.iloc[occ_ratios.index.get_level_values('gender') == '''M'''].sort_values(ascending=False))
#Step 6. For each occupation, calculate the minimum and maximum ages
print(users.groupby(['occupation'])['age'].max())
print(users.groupby(['occupation'])['age'].min())
#Step 7. For each combination of occupation and gender, calculate the mean age
print(users.groupby(['occupation','gender'])['age'].mean())
#Step 8. For each occupation present the percentage of women and men
print(users_by_occ / users_by_occ.groupby(level=0).sum() * 100)
#4. Merge
#Step 1. Import the necessary libraries
#Step 2. Create the 3 DataFrames based on the followin raw data
raw_data_1 = {
'subject_id': ['1', '2', '3', '4', '5'],
'first_name': ['Alex', 'Amy', 'Allen', 'Alice', 'Ayoung'],
'last_name': ['Anderson', 'Ackerman', 'Ali', 'Aoni', 'Atiches']}
raw_data_2 = {
'subject_id': ['4', '5', '6', '7', '8'],
'first_name': ['Billy', 'Brian', 'Bran', 'Bryce', 'Betty'],
'last_name': ['Bonder', 'Black', 'Balwner', 'Brice', 'Btisan']}
raw_data_3 = {
'subject_id': ['1', '2', '3', '4', '5', '7', '8', '9', '10', '11'],
'test_id': [51, 15, 15, 61, 16, 14, 15, 1, 61, 16]}
#Step 3. Assign each to a variable called data1, data2, data3
data1 = pd.DataFrame.from_dict(raw_data_1)
data2 = pd.DataFrame.from_dict(raw_data_2)
data3 = pd.DataFrame.from_dict(raw_data_3)
#Step 4. Join the two dataframes along rows and assign all_data
all_data = pd.concat([data1,data2])
#Step 5. Join the two dataframes along columns and assing to all_data_col
all_data_col = pd.concat([data1,data2],axis = 1)
#Step 6. Print data3
print(data3)
#Step 7. Merge all_data and data3 along the subject_id value
merged7 = data3.set_index('subject_id').join(all_data.set_index('subject_id'))
#Step 8. Merge only the data that has the same 'subject_id' on both data1 and data2
merged8 = data1.merge(data2, how = 'inner', on = ['subject_id'])
#Step 9. Merge all values in data1 and data2, with matching records from both sides where available.
merged9 = data1.merge(data2, how = 'outer', on = ['subject_id'])
#5. Deleting
#This exercise may seem a little bit strange, but keep doing it
#Step 1. Import the necessary libraries
#Step 2. Import the dataset from this
#https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
iris = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)
#Step 3. Assign it to a variable called iris
#Step 4. Create columns for the dataset
#1. sepal_length (in cm)
#2. sepal_width (in cm)
#3. petal_length (in cm)
#4. petal_width (in cm)
#5. class
iris.columns = ['sepal_length','sepal_width','petal_length','petal_width','class']
#Step 5. Is there any missing value in the dataframe?
iris.isna().sum()
#Step 6. Lets set the values of the rows 10 to 29 of the column 'petal_length' to NaN
iris.iloc[10:30,2] = np.nan
#Step 7. Good, now lets substitute the NaN values to 1.0
iris = iris.fillna(1)
#Step 8. Now let's delete the column class
iris = iris.drop(columns='class')
#Step 9. Set the first 3 rows as NaN
iris.iloc[0:3,:] = np.nan
#Step 10. Delete the rows that have NaN
iris = iris.dropna()
#Step 11. Reset the index so it begins with 0 again
iris.reset_index()
#Create your own question and answer it.