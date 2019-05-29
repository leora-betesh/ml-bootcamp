import pandas as pd 
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold, train_test_split, cross_val_score
from mlxtend.classifier import StackingCVClassifier
from xgboost import XGBRegressor
#My data cleanup code
import home_price_cleaning as hpc


train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
test_ids = test_data["Id"]

#Many kernels reccommend dropping this outlier
train_data.drop(train_data[(train_data['GrLivArea'] > 4500)].index,inplace=True)

y_train = train_data["SalePrice"]

X_train, X_test = hpc.clean_data(train_data,test_data)

y_train = np.log1p(y_train)

filter_col = [col for col in X_train if col.startswith('onehot')]

X_train_feat = X_train[['YrSold','MoSold','OverallQual','OverallCond','TotalBsmtSF','1stFlrSF','GrLivArea','FullBath','TotRmsAbvGrd','GarageArea','YearRemodAdd','YearBuilt']]
X_test_feat = X_test[['YrSold','MoSold','OverallQual','OverallCond','TotalBsmtSF','1stFlrSF','GrLivArea','FullBath','TotRmsAbvGrd','GarageArea','YearRemodAdd','YearBuilt']]

X_train_feat = pd.concat([X_train_feat, X_train[filter_col]], axis=1)
X_test_feat = pd.concat([X_test_feat, X_test[filter_col]], axis=1)

scaler = StandardScaler()
scaler.fit(X_train_feat)
X_train_scaled = scaler.transform(X_train_feat)
X_test_scaled = scaler.transform(X_test_feat)

#Just for debugging
#X_train_feat['SalePrice'] = y_train

RANDOM_SEED = 42

#clf1 = SVR(C=1,degree=2,epsilon=.03)
clf1 = XGBRegressor(colsample_bytree=.7,gamma=0,learning_rate=.04,max_depth=3,min_child_weight=.2,n_estimators=870,subsample=.8)
clf2 = RandomForestRegressor(bootstrap=True,n_estimators=1000, max_features=0.2,min_samples_split=2)
clf3 = KernelRidge(alpha=1.0)
lr = LinearRegression()

clf1.fit(X_train_feat, y_train)
clf2.fit(X_train_feat, y_train)
clf3.fit(X_train_feat, y_train)

clf1_pred = clf1.predict(X_test_feat)
clf2_pred = clf2.predict(X_test_feat)
clf3_pred = clf3.predict(X_test_feat)
clf_avg_pred = [0]*len(X_test_feat)

#Take average prediction of three classifiers
for i in range(len(X_test_feat)):
    clf_avg_pred[i] = (clf1_pred[i] + clf2_pred[i] + clf3_pred[i]) / 3

#Convert back to sales price, after normalized 
clf_avg_pred = np.expm1(clf_avg_pred)
    
submission_data = pd.DataFrame({'Id': test_ids, 'SalePrice': clf_avg_pred})
submission_data.to_csv('price_prediction_submission.csv',index=False)

#sclf = StackingCVClassifier(classifiers=[clf1, clf2, clf3],meta_classifier=lr, random_state=RANDOM_SEED)
#
#print('10-fold cross validation:\n')
#
#for clf, label in zip([clf1, clf2, clf3, sclf], 
#                      ['XGboost regression', #SVM Regression
#                       'Random Forest Regression', 
#                       'Kernel Ridge Regression',
#                       'StackingClassifier']):
#
#    scores = cross_val_score(clf, X_train_feat, y_train, cv=10, scoring='r2')
#    print("Accuracy: " ,scores.mean(), label)
