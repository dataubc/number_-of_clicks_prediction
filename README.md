### number_-of_clicks_prediction
Predicting the number of clicks for a hotel listing site :


To reproduce th result of this work :

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
python python xgboost_model.py
```

To run the EDA and Model development notebooks

- First run the `1.eda_preprocessing.ipynb`. When all cells are run the cleaned data will be saved as `cleaned.csv` which will be needed for the seecond notebook. This notebook include all the deatails about the eda, data preprocessing, filling missing values and visualization.

- Run the `2.creating_models.ipynb`. This notebook goes over the oveflow of developing the ML models frorm linear models, Random Forest, etc. Different techniques were implemented for cross-validation, hyper-paramters optimization, feature engineering, regularization,etc. The best model is then testd on the test data.
