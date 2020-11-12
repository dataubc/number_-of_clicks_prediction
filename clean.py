import pandas as pd
import numpy as np
import requests
from io import StringIO

import matplotlib.pyplot as plt
import time
import pickle
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
import sys

def main(hotel_url ,output_path, type = 'train' ):
    
  

    def loading_data(file_url):
        """
        Reading data from a given url

        inputs:
        ------
        url : link to the data
        returns
        -----:
        data : data frame  
        """

        file_id = file_url.split('/')[- 2]
        dwn_url='https://drive.google.com/uc?export=download&id=' + file_id
        url = requests.get(dwn_url).text
        csv_raw = StringIO(url)
        df = pd.read_csv(csv_raw)
        return df
 
    hotel_data = loading_data(hotel_url)
    hotel_data = hotel_data[hotel_data['content_score'].notnull()]
    hotel_data = hotel_data[hotel_data['n_images'] >= 0]
    hotel_data = hotel_data[hotel_data['distance_to_center'].notnull()]
    hotel_data['avg_rating'] = hotel_data['avg_rating'].fillna(0)
    hotel_data = hotel_data[hotel_data['stars'].notnull()]
    hotel_data = hotel_data[hotel_data['avg_price'].notnull()]
    
    city_id_count = hotel_data['city_id'].value_counts().reset_index(name='count').rename(columns={'index': 'city_id'})
    hotel_data = pd.merge(hotel_data,city_id_count ,on=['city_id'], how='left')
    hotel_data = hotel_data.drop(columns = ['city_id'])
    hotel_data['avg_saving_cash'] =hotel_data['avg_price']*hotel_data['avg_saving_percent']
    
    if type =='train':
    
        y = hotel_data['n_clicks']
        removed_outliers = y.between(y.quantile(.05), y.quantile(.95))
        index_names = hotel_data[~removed_outliers].index 
        hotel_data.drop(index_names, inplace=True)
        
        hotel_data['n_clicks'] = hotel_data['n_clicks'].astype(float)
        
    hotel_data['count'] = hotel_data['count'].astype(float)
    hotel_data.to_csv(output_path,index =False)
                                                           

if __name__ == "__main__":
    import sys
    
    main(sys.argv[1] ,sys.argv[2], sys.argv[3])