import datetime
import pandas as pd
import numpy as np
from haversine import haversine
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.decomposition import PCA

df_train = pd.read_csv("data/train.csv", low_memory = False)
df_train = df_train.drop(columns = ['passenger_count'])
df_train = df_train.dropna()
df_train = df_train[df_train.fare_amount != 'fare_amount']

df_train['pickup_latitude'] = pd.to_numeric(df_train['pickup_latitude'])
df_train['pickup_longitude'] = pd.to_numeric(df_train['pickup_longitude'])
df_train['dropoff_latitude'] = pd.to_numeric(df_train['dropoff_latitude'])
df_train['dropoff_longitude'] = pd.to_numeric(df_train['dropoff_longitude'])
df_train['fare_amount'] = pd.to_numeric(df_train['fare_amount'])
df_train['passenger_count'] = pd.to_numeric(df_train['passenger_count'])

df_train = df_train[df_train['passenger_count'] <= 6]

def hours_after_midnight(date_time):
    time = date_time.split(' ')
    if len(time) == 3:
        time_arr = time[1].split(':')
        hours_after_midnight = int(time_arr[0]) 
        return hours_after_midnight
    else:
        return np.nan

def day_of_week(date_time):
    time = date_time.split(' ')
    if len(time) == 3:
        date_arr = time[0].split('-')
        date = datetime.date(int(date_arr[0]), int(date_arr[1]), int(date_arr[2]))
        return (date.weekday() + 1)
    else:
        return np.nan

def month(date_time):
    time = date_time.split(' ')
    if len(time) == 3:
        return int(time[0].split('-')[1])
    else:
        return np.nan

def year(date_time):
    time = date_time.split(' ')
    if len(time) == 3:
        return float(time[0].split('-')[0])
    else:
        return np.nan

hour_of_day_vec = np.vectorize(hours_after_midnight)
day_of_week_vec = np.vectorize(day_of_week)
month_vec = np.vectorize(month)
year_vec = np.vectorize(year)

time_column = df_train['pickup_datetime'].to_numpy()
df_train.drop(columns = ['pickup_datetime', 'key'], inplace = True)

df_train['year'] = year_vec(time_column)
df_train['month'] = month_vec(time_column)
df_train['weekday'] = day_of_week_vec(time_column)
df_train['hour'] = hour_of_day_vec(time_column)
df_train['distance'] = haversine(df_train['pickup_latitude'], df_train['pickup_longitude'], df_train['dropoff_latitude'] , df_train['dropoff_longitude'])

df_train = df_train[df_train['distance'] > 0]

nyc_bounds = (-74.5, -72.8, 40.5, 41.8)

def select_within_bounds(df, bounds):
    pickup_indices = (df.pickup_longitude >= bounds[0]) & (df.pickup_longitude <= bounds[1]) & \
        (df.pickup_latitude >= bounds[2]) & (df.pickup_latitude <= bounds[3])

    dropoff_indices = (df.dropoff_longitude >= bounds[0]) & (df.dropoff_longitude <= bounds[1]) & \
        (df.dropoff_latitude >= bounds[2]) & (df.dropoff_latitude <= bounds[3])

    return pickup_indices & dropoff_indices

def map_to_nyc_mask(longitude, latitude, points_x, points_y, bounds):
    x = (points_x * (longitude - bounds[0]) / (bounds[1] - bounds[0])).astype('int')
    y = (points_y - points_y * (latitude - bounds[2]) / (bounds[3] - bounds[2])).astype('int')
    return x,y

def remove_points_in_water(df):
    # Create a mask of the New York City with 1 as land and 0 as water
    nyc_mask = plt.imread('img/nyc_water_mask.png')[:,:,0] > 0.9

    # Remove points outside New York
    df = df[select_within_bounds(df, nyc_bounds)]
    print("After Bounds:", df.shape[0])

    # Map the latitudes and longitudes to the points in the map
    pickup_x, pickup_y = map_to_nyc_mask(df.pickup_longitude, df.pickup_latitude, nyc_mask.shape[1],
                                        nyc_mask.shape[0], nyc_bounds)
    dropoff_x, dropoff_y = map_to_nyc_mask(df.dropoff_longitude, df.dropoff_latitude, nyc_mask.shape[1],
                                        nyc_mask.shape[0], nyc_bounds)
    
    pickup_y[pickup_y == 1262] = 1261
    dropoff_y[dropoff_y == 1262] = 1261
    pickup_x[pickup_x == 1242] = 1241
    dropoff_x[dropoff_x == 1242] = 1241

    # Compute the indices where pickup and dropoff locations are on land
    indices = nyc_mask[pickup_y, pickup_x] & nyc_mask[dropoff_y, dropoff_x]

    df = df[indices]
    print("Number of trips in water: ", np.sum(~indices))
    return df

df_train = remove_points_in_water(df_train)

df_train.drop(columns = ['index'], inplace = True)

y = df_train['fare_amount']
X = df_train.drop(columns=['fare_amount'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

params = {
        'learning_rate': 0.65,
        'application': 'regression',
        'max_depth': 3,
        'num_leaves': 200,
        'verbosity': -1,
        'metric': 'RMSE',
    }

lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
gbm = lgb.train(params,
                lgb_train,
                valid_sets=lgb_eval,
                num_boost_round=300)
gbm.save_model('model_v2.txt')


df_test = pd.read_csv('data/test.csv', low_memory = False)

time_column = df_test['pickup_datetime'].to_numpy()
df_test.drop(columns = ['pickup_datetime'], inplace = True)
df_test['year'] = year_vec(time_column)
df_test['month'] = month_vec(time_column)
df_test['weekday'] = day_of_week_vec(time_column)
df_test['hour'] = hour_of_day_vec(time_column)

df_test['distance'] = haversine(df_test['pickup_latitude'], df_test['pickup_longitude'], df_test['dropoff_latitude'] , df_test['dropoff_longitude'])

keys = df_test['key'] 
df_test.drop(columns = ['key'], inplace = True)

df_test.fillna(df_test.mean())
df_test.drop(columns = ['index'], inplace = True)

pred_fares = gbm.predict(df_test, num_iteration=gbm.best_iteration)
df_final = pd.DataFrame({'key':keys, 'fare_amount':pred_fares})

df_final.to_csv(r'predictions/model_v2.csv', index = False)
