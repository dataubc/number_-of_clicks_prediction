# number_-of_clicks_prediction
Predicting the number of clicks for a hotel listing site


To install requirements : 

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