"""
Created on Mon May 13 10:18:34 2019

@author: Leora Betesh
"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import home_price_pca as hppca

#train = pd.read_csv('train.csv')
#test = pd.read_csv('test.csv')

def clean_data(train_input,test_input):
    train = train_input
    test = test_input
    train.drop(['Id'], axis=1, inplace=True)
    test.drop(['Id'], axis=1, inplace=True)
   
    train.drop(['SalePrice'], axis=1, inplace=True)
    train.reset_index(drop=True, inplace=True)
    
    frames = [train,test]
    combined = pd.concat(frames)    
    combined.drop(['FireplaceQu','Alley','Fence','MiscFeature','Utilities', 'Street', 'PoolQC',], axis=1, inplace=True)

    for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond','BsmtQual', 'BsmtCond', 
                'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2','MasVnrType']:
        combined[col] = combined[col].fillna('None')
    
    for col in ['GarageYrBlt', 'GarageArea', 'GarageCars','LotFrontage','MasVnrArea','TotalBsmtSF']:
        combined[col] = combined[col].fillna(0)        
    
    ## Filling these columns With most suitable value for these columns 
    combined['Functional'] = combined['Functional'].fillna('Typ') 
    combined['Electrical'] = combined['Electrical'].fillna("SBrkr") 
    combined['KitchenQual'] = combined['KitchenQual'].fillna("TA") 
    
    ## Filling these with MODE , i.e. , the most frequent value in these columns .
    combined['Exterior1st'] = combined['Exterior1st'].fillna(combined['Exterior1st'].mode()[0]) 
    combined['Exterior2nd'] = combined['Exterior2nd'].fillna(combined['Exterior2nd'].mode()[0])
    combined['SaleType'] = combined['SaleType'].fillna(combined['SaleType'].mode()[0])  
    combined['MSZoning'] = combined.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
    
    combined['YrSold'] = 2018 - combined['YrSold']
    #missing data
#    total = combined.isnull().sum().sort_values(ascending=False)
#    percent = (combined.isnull().sum()/combined.isnull().count()).sort_values(ascending=False)
#    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        
#    combined['neighborhood_onehot'] = pd.Categorical(combined['Neighborhood'])
#    dfDummies1 = pd.get_dummies(combined['neighborhood_onehot'], prefix = 'onehot')
#    combined['mszoning_onehot'] = pd.Categorical(combined['MSZoning'])
#    dfDummies2 = pd.get_dummies(combined['MSZoning'], prefix = 'onehot')
#    combined['basement_qual_onehot'] = pd.Categorical(combined['BsmtQual'])
#    dfDummies3 = pd.get_dummies(combined['basement_qual_onehot'], prefix = 'onehot')
#    combined = pd.concat([combined, dfDummies1], axis=1)
#    combined = pd.concat([combined, dfDummies2], axis=1)
#    combined = pd.concat([combined, dfDummies3], axis=1)
    
    combinedPCA = pd.DataFrame(hppca.pca_reduce(combined))
    
    train_edited = combinedPCA.iloc[0:1458,:]
    test_edited = combinedPCA.iloc[1458:2917,:]
    
#    train_edited = train_edited[train_edited.GrLivArea < 4500]  
#    test_edited = test_edited[test_edited.GrLivArea < 4500]  
#        
    return train_edited, test_edited


