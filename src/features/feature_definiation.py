import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import timedelta
from sklearn.decomposition import PCA
from pathlib import Path
import datetime as dt
from sklearn.cluster import MiniBatchKMeans
import yaml

def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h

def dummy_manhattan_distance(lat1, lng1, lat2, lng2):
    a = haversine_array(lat1, lng1, lat1, lng2)
    b = haversine_array(lat1, lng1, lat2, lng1)
    return a + b

def bearing_array(lat1, lng1, lat2, lng2):
    AVG_EARTH_RADIUS = 6371  # in km
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))

def datetime_feature_fix(df):
    df['pickup_datetime'] = pd.to_datetime(df.pickup_datetime)
    df.loc[:,'pickup_date'] = df['pickup_datetime'].dt.date
    try:
       df['dropoff_datetime'] = pd.to_datetime(df.dropoff_datetime)
    except:
       df['dropoff_datetime'] = np.nan
    df['store_and_fwd_flag'] = 1 * (df.store_and_fwd_flag.values == 'Y')
    try:
        df['check_trip_duration'] = (df['dropoff_datetime'] - df['pickup_datetime']).map(lambda x: x.total_seconds())
        df['log_trip_duration'] = np.log(df['trip_duration'].values + 1)
    except:
        df['check_trip_duration'] = np.nan
        df['log_trip_duration'] = np.nan
    df.loc[:, 'pickup_weekday'] = df['pickup_datetime'].dt.weekday
    df.loc[:, 'pickup_hour_weekofyear'] = df['pickup_datetime'].dt.isocalendar().week
    df.loc[:, 'pickup_hour'] = df['pickup_datetime'].dt.hour
    df.loc[:, 'pickup_minute'] = df['pickup_datetime'].dt.minute
    df.loc[:, 'pickup_dt'] = (df['pickup_datetime'] - df['pickup_datetime'].min()).dt.total_seconds()
    df.loc[:, 'pickup_week_hour'] = df['pickup_weekday'] * 24 + df['pickup_hour']

def PCA_feature(df):
    coords = np.vstack((df[['pickup_latitude', 'pickup_longitude']].values,
                    df[['dropoff_latitude', 'dropoff_longitude']].values))

    pca = PCA().fit(coords)
    df['pickup_pca0'] = pca.transform(df[['pickup_latitude', 'pickup_longitude']])[:, 0]
    df['pickup_pca1'] = pca.transform(df[['pickup_latitude', 'pickup_longitude']])[:, 1]
    df['dropoff_pca0'] = pca.transform(df[['dropoff_latitude', 'dropoff_longitude']])[:, 0]
    df['dropoff_pca1'] = pca.transform(df[['dropoff_latitude', 'dropoff_longitude']])[:, 1]

    return coords
    


    
def create_dist_features(df):
    df.loc[:, 'distance_haversine'] = haversine_array(df['pickup_latitude'].values, df['pickup_longitude'].values, df['dropoff_latitude'].values, df['dropoff_longitude'].values)
    df.loc[:, 'distance_dummy_manhattan'] = dummy_manhattan_distance(df['pickup_latitude'].values, df['pickup_longitude'].values, df['dropoff_latitude'].values, df['dropoff_longitude'].values)
    df.loc[:, 'direction'] = bearing_array(df['pickup_latitude'].values, df['pickup_longitude'].values, df['dropoff_latitude'].values, df['dropoff_longitude'].values)
    df.loc[:, 'pca_manhattan'] = np.abs(df['dropoff_pca1'] - df['pickup_pca1']) + np.abs(df['dropoff_pca0'] - df['pickup_pca0'])
    df.loc[:, 'center_latitude'] = (df['pickup_latitude'].values + df['dropoff_latitude'].values) / 2
    df.loc[:, 'center_longitude'] = (df['pickup_longitude'].values + df['dropoff_longitude'].values) / 2
    df.loc[:, 'pickup_lat_bin'] = np.round(df['pickup_latitude'], 3)
    df.loc[:, 'pickup_long_bin'] = np.round(df['pickup_longitude'], 3)
    df.loc[:, 'center_lat_bin'] = np.round(df['center_latitude'], 2)
    df.loc[:, 'center_long_bin'] = np.round(df['center_longitude'], 2)
    df.loc[:, 'pickup_dt_bin'] = (df['pickup_dt'] // (3 * 3600))


def speed_estimate(df):
    try:
        df.loc[:, 'avg_speed_h'] = 1000 * df['distance_haversine'] / df['trip_duration']
        df.loc[:, 'avg_speed_m'] = 1000 * df['distance_dummy_manhattan'] / df['trip_duration']
    except:
        df.loc[:, 'avg_speed_h'] = np.nan
        df.loc[:, 'avg_speed_m'] = np.nan



def cluster_features(df, coords,params):
    sample_ind = np.random.permutation(len(coords))[:500000]
    kmeans = MiniBatchKMeans(n_clusters=params['n_clusters'], batch_size=params['min_batch_size']).fit(coords[sample_ind])
    df.loc[:, 'pickup_cluster'] = kmeans.predict(df[['pickup_latitude', 'pickup_longitude']])
    df.loc[:, 'dropoff_cluster'] = kmeans.predict(df[['dropoff_latitude', 'dropoff_longitude']])


def Temporal_and_geospatial_aggregation(df):
    for gby_col in ['pickup_hour', 'pickup_date', 'pickup_dt_bin',
                'pickup_week_hour', 'pickup_cluster', 'dropoff_cluster']:
        gby = df.groupby(gby_col)[['avg_speed_h', 'avg_speed_m', 'log_trip_duration']].mean()
        gby.columns = ['%s_gby_%s' % (col, gby_col) for col in gby.columns]
        df = pd.merge(df, gby, how='left', left_on=gby_col, right_index=True)
        

    for gby_cols in [['center_lat_bin', 'center_long_bin'],
                    ['pickup_hour', 'center_lat_bin', 'center_long_bin'],
                    ['pickup_hour', 'pickup_cluster'],  ['pickup_hour', 'dropoff_cluster'],
                    ['pickup_cluster', 'dropoff_cluster']]:
        coord_speed = df.groupby(gby_cols)['avg_speed_h'].mean().reset_index()
        coord_count = df.groupby(gby_cols)['id'].count().reset_index()
        coord_stats = pd.merge(coord_speed, coord_count, on=gby_cols)
        coord_stats = coord_stats[coord_stats['id'] > 100]
        coord_stats.columns = gby_cols + ['avg_speed_h_%s' % '_'.join(gby_cols), 'cnt_%s' %  '_'.join(gby_cols)]
        df = pd.merge(df, coord_stats, how='left', on=gby_cols)
    group_freq = '60min'
    df_all = df[['id', 'pickup_datetime', 'pickup_cluster', 'dropoff_cluster']]
    df.loc[:,'pickup_datetime_group'] = df['pickup_datetime'].dt.round(group_freq)

    # Count trips over 60min
    df_counts = df_all.set_index('pickup_datetime')[['id']].sort_index()
    df_counts['count_60min'] = df_counts.isnull().rolling(group_freq).count()['id']
    df = df.merge(df_counts, on='id', how='left')


    # Count how many trips are going to each cluster over time
    dropoff_counts = df_all \
        .set_index('pickup_datetime') \
        .groupby([pd.Grouper(freq=group_freq), 'dropoff_cluster']) \
        .agg({'id': 'count'}) \
        .reset_index().set_index('pickup_datetime') \
        .groupby('dropoff_cluster').rolling('240min').mean() \
        .reset_index(level='dropoff_cluster') \
        .reset_index().set_index('pickup_datetime').shift(freq='-120min').reset_index() \
        .rename(columns={'pickup_datetime': 'pickup_datetime_group', 'id': 'dropoff_cluster_count'})

    df['dropoff_cluster_count'] = df[['pickup_datetime_group', 'dropoff_cluster']].merge(dropoff_counts, on=['pickup_datetime_group', 'dropoff_cluster'], how='left')['dropoff_cluster_count'].fillna(0)
    
    df_all = df[['id', 'pickup_datetime', 'pickup_cluster', 'dropoff_cluster']]
    pickup_counts = df_all \
        .set_index('pickup_datetime') \
        .groupby([pd.Grouper(freq=group_freq), 'pickup_cluster']) \
        .agg({'id': 'count'}) \
        .reset_index().set_index('pickup_datetime') \
        .groupby('pickup_cluster').rolling('240min').mean() \
        .reset_index().set_index('pickup_datetime').shift(freq='-120min').reset_index() \
        .rename(columns={'pickup_datetime': 'pickup_datetime_group', 'id': 'pickup_cluster_count'})

    df['pickup_cluster_count'] = df[['pickup_datetime_group', 'pickup_cluster']].merge(pickup_counts, on=['pickup_datetime_group', 'pickup_cluster'], how='left')['pickup_cluster_count'].fillna(0)


    
def test_feature_build(df,params):
    datetime_feature_fix(df)
    coords=PCA_feature(df)
    create_dist_features(df)
    speed_estimate(df)
    cluster_features(df, coords)
    Temporal_and_geospatial_aggregation(df)
    print(df.head())
    

def feature_build(df,params):
    datetime_feature_fix(df)
    coords=PCA_feature(df)
    create_dist_features(df)
    speed_estimate(df)
    cluster_features(df, coords,params)
    Temporal_and_geospatial_aggregation(df)
    return df



if __name__ == "__main__":
    curr_dir=Path(__file__)
    home_dir=curr_dir.parent.parent.parent
    test_path=home_dir.as_posix()+'/data/raw/test.csv'
    data=pd.read_csv(test_path,nrows=10)
    params_path=home_dir.as_posix()+'/params.yaml'
    params = yaml.safe_load(open(params_path))['feature_definiation']

    test_feature_build(data,params)
   
