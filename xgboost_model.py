import pandas as pd
import numpy as np
import xgboost

# reading data
hotel_data = pd.read_csv('cleaned_train.csv')
X = hotel_data.drop(columns=['n_clicks', 'hotel_id'])
# let's also add the new feature avg_saving_cash
X['avg_saving_cash'] = X['avg_price'] * X['avg_saving_percent']
y = hotel_data['n_clicks']

# let's create trained data for xgboost
dtrain = xgboost.DMatrix(X, label=y)

params = {'max_depth': 6, 'min_child_weight': 3, 'eta': .1, 'subsample': 1, 'colsample_bytree': 0.7,
          'objective': 'reg:squarederror', 'eval_metric': "rmse"}
num_boost_round = 999
print('Training phase has started')

# training best model on the optimized hyper-parameters.
best_model = xgboost.train(
    params,
    dtrain,
    num_boost_round=num_boost_round,
)
print('Saving the model as best_model.model')
best_model.save_model("best_model.model")
print('Reading test data')

# reading test data
X_test = pd.read_csv('cleaned_test.csv')
dtest = xgboost.DMatrix(X_test.drop(columns=['hotel_id']))
predicted_y = best_model.predict(dtest)
X_test['n_clicks'] = predicted_y
# getting all negative prediction to 0
X_test['n_clicks'] = np.where(X_test['n_clicks'] < 0, 0, X_test['n_clicks'])
final_result = X_test[['hotel_id', 'n_clicks']]
print('Saving the prediction as predictions.csv')
# saving the result
final_result.to_csv('predictions.csv')
