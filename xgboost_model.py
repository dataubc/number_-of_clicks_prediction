import pandas as pd
import numpy as np
import requests
from io import StringIO

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostRegressor,  BaggingRegressor, ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MaxAbsScaler,MinMaxScaler,PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.base import clone, TransformerMixin
from sklearn.model_selection import KFold
import time
import pickle
from pandas.plotting import scatter_matrix
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import SGDRegressor
import xgboost
from xgboost import XGBRegressor
from sklearn.model_selection import RepeatedStratifiedKFold


hotel_data = pd.read_csv('cleaned_train.csv')
X = hotel_data.drop(columns = ['n_clicks','hotel_id'])
# let's also add the new featture avg_saving_cash
X['avg_saving_cash'] =X['avg_price']*X['avg_saving_percent']
y = hotel_data['n_clicks']

    

dtrain = xgboost.DMatrix(X, label=y)

params = {
    # These are the hyper-parameters that we are going to tune.
    'max_depth':6,
    'min_child_weight': 3,
    'eta':.1,
    'subsample': 1,
    'colsample_bytree': 0.7,
    # Other parameters
    'objective':'reg:squarederror',
}
params['eval_metric'] = "rmse"
num_boost_round = 999
print('Training phase has started')

best_model = xgboost.train(
    params,
    dtrain,
    num_boost_round=num_boost_round,
)
print('Saving the model as best_model1.model')
best_model.save_model("best_model1.model")
print('Reading test data')
X_test = pd.read_csv('cleaned_test.csv')
dtest = xgboost.DMatrix(X_test.drop(columns = ['hotel_id']))
predicted_y = best_model.predict(dtest)
X_test['n_clicks'] = predicted_y
# getting all negative prediction to 0
X_test['n_clicks'] = np.where(X_test['n_clicks'] < 0,0,X_test['n_clicks'])
final_result = X_test[['hotel_id','n_clicks']]
print('Saving the prediction as predictions1.csv')
final_result.to_csv('predictions1.csv')






