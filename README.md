# Number  of clicks prediction

Predicting the number of clicks for a hotel listing site :

# Task:

Given a training set for a hotel listing site that includes a list of hotels and hotel parameters such as the number of reviews, stars, avg rating, etc, could we predict the number of clicks that a hotel will receive?

# EDA:

EDA has revealed that the raining set has many missing data. Each feature was explored and missing data were filled according to the best suitable method. Furthermore, Power Transformation seems to be essential to process features that have heavy tail distribution. One additional issue that was noted during the eda is big variability among features in terms of scale, therefore, a minmax scaler will be needed as a part of the preprocessing phase.

# Models:

Preprocessing and Model Evaluation: Since the target variable also appears to be mostly zeros, there was a need to split the data in a stratified way. 20% of the data was used for validation, and the models were trained in the remaining 80%. To evaluate the performance of the models, we use R2 and mean squared error.

Baseline: Linear models are fast and easy to build and interpret, if we can fit a linear model with good performance then it will be fast when deployed in production. Also, we would understand exactly how each predictor contributes to our target variable n_clicks. Hence we started with Ridge models which apply linear regression with regularization, MinMax scaling and cross-validation was applied as part of a pipeline.

Base model + Power Transformation:
EDA has revealed that some features have heavy tail distribution issues. As we know linear models such as Ridge assumes that features have Gaussian probability distribution. Power transformation was found to help improve the accuracy of the result.

Base model + Power Transformation + Feature Engineering:
One fast way to improve performance is to create a more meaningful feature. For example, we have the avg_price and the avg_saving_percent, but often people think in terms of how much money the save. Therefore, we created a new feature, called avg_saving_cash by multiplying avg_price and avg_saving_percent. This has reduced the MSE of the model.

Testing other Models:
Several models including SGD, Gradient Boost, Random Forest, and XGB and evaluated based on mean squared error and performance.

- linear Regression SGDRegressor has the fastest models but also have the lowest accuracy.

- Gradient Boost model has bias issues, while random forest suffers from variance issues. We can reduce the overfitting of the random forest by reducing max depth, n_estimotros, or max_features.

- XGBRegressor seems to be the most promising in terms of MSE. We adopted this model fine-tuned the parameters to improve the mse.

XGBOOST : Model Improvements Using Hyperparameters optimization

Hyperparameters optimization was used to find the best values for `max_depth`, `min_child_weight`, `subsample` , `colsample_bytree` and `ETA`. This was a time consuming especially when the `city_id` was encoded using One-Hot-Encoding. This was mainly because `city_id` has ~ 300,000 labels. Therefore, I replaced each `city_id` with it's count. Comparing the results for models with One-Hot-Encoding indicates that using `count` is more effective in terms of the increased predictive power of the model.

Following are the best paramters for XGBOOST :

{'max_depth': 6,
 'min_child_weight': 3,
 'eta': 0.1,
 'subsample': 1,
 'colsample_bytree': 0.7,
 'objective': 'reg:squarederror',
 'eval_metric': 'rmse'}
 
 A new XGBOOST model was trained on the optimized parameters and the whole dataset and saved for future use. The new model was then used for producing predictions for the testing data set.
 
# Scripts:
To reproduce the result of this work :

- install requirements : 
```
pip install -r requirements.txt

```

Then to run the model and export the results
```
#cleaning train data
python clean.py 'https://drive.google.com/file/d/1c85h1hzgzLvAeYSh-EVpY6Gz3dYLsd6R/view?usp=sharing' 'cleaned_train.csv' train
#cleaning test data
python clean.py 'https://drive.google.com/file/d/13zo8AnjBsXBGHG-KKybxPG5D0Kc5p3-s/view?usp=sharing' 'cleaned_test.csv' test
python xgboost_model.py
```

To run the EDA and Model development notebooks

- First, run the `1.eda_preprocessing.ipynb`. When all cells are run the cleaned data will be saved as `cleaned.csv` which will be needed for the seecond notebook. This notebook includes all the details about the eda, data preprocessing, filling missing values, and visualization.

- Run the `2.creating_models.ipynb`. This notebook goes over the overflow of developing the ML models from linear models, Random Forest, etc. Different techniques were implemented for cross-validation, hyper-parameters optimization, feature engineering, regularization,etc. The best model is then tested on the test data.