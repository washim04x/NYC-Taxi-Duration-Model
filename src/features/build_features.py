import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import timedelta
from sklearn.decomposition import PCA
from pathlib import Path
import datetime as dt
from sklearn.cluster import MiniBatchKMeans
from feature_definiation import feature_build
import yaml
def load_data(input_path):
    df=pd.read_csv(input_path)
    return df

def save_data(train,test, output_path):
    Path(output_path).mkdir(parents=True, exist_ok=True)
    train.to_csv(output_path+'/train.csv', index=False)
    test.to_csv(output_path+'/test.csv', index=False)

if __name__ == "__main__":
    curr_dir=Path(__file__)
    home_dir=curr_dir.parent.parent.parent
    params_path=home_dir.as_posix()+'/params.yaml'
    params = yaml.safe_load(open(params_path))['feature_definiation']
    train_path=home_dir.as_posix()+'/data/raw/train.csv'
    test_path=home_dir.as_posix()+'/data/raw/test.csv'

    train_data=load_data(train_path)
    test_data=load_data(test_path)

    output_path=home_dir.as_posix()+'/data/processed/'



    train_data=feature_build(train_data,params)
    test_data=feature_build(test_data,params)



    do_not_use_for_training = ['id', 'log_trip_duration', 'pickup_datetime', 'dropoff_datetime',
                            'trip_duration', 'check_trip_duration',
                            'pickup_date', 'avg_speed_h', 'avg_speed_m',
                            'pickup_lat_bin', 'pickup_long_bin',
                            'center_lat_bin', 'center_long_bin',
                            'pickup_dt_bin', 'pickup_datetime_group']
    feature_names = [f for f in train_data.columns if f not in do_not_use_for_training]
    test_data=test_data[feature_names]
    

    
    feature_names.append('trip_duration')
    train_data=train_data[feature_names]
  
    
    
    save_data(train_data,test_data,output_path)
        

