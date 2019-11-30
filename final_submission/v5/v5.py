import datetime
import pandas as pd
import numpy as np
from haversine import haversine
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import xgboost as xgb
import joblib

df_train = pd.read_hdf("data/train.csv", low_memory = False)
df_train = df_train.drop(columns = ['passenger_count'])
df_train.dropna(inplace = True)
df_train = df_train[df_train.fare_amount != 'fare_amount']

df_train['pickup_latitude'] = pd.to_numeric(df_train['pickup_latitude'])
df_train['pickup_longitude'] = pd.to_numeric(df_train['pickup_longitude'])
df_train['dropoff_latitude'] = pd.to_numeric(df_train['dropoff_latitude'])
df_train['dropoff_longitude'] = pd.to_numeric(df_train['dropoff_longitude'])
df_train['fare_amount'] = pd.to_numeric(df_train['fare_amount'])

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

def add_airport_dist(df):
    """
    Return minumum distance from pickup or dropoff coordinates to each airport.
    JFK: John F. Kennedy International Airport
    EWR: Newark Liberty International Airport
    LGA: LaGuardia Airport
    """
    jfk_coord = (40.639722, -73.778889)
    ewr_coord = (40.6925, -74.168611)
    lga_coord = (40.77725, -73.872611)
    
    pickup_lat = df['pickup_latitude']
    dropoff_lat = df['dropoff_latitude']
    pickup_lon = df['pickup_longitude']
    dropoff_lon = df['dropoff_longitude']
    
    pickup_jfk = haversine(pickup_lat, pickup_lon, jfk_coord[0], jfk_coord[1]) 
    dropoff_jfk = haversine(jfk_coord[0], jfk_coord[1], dropoff_lat, dropoff_lon) 
    pickup_ewr = haversine(pickup_lat, pickup_lon, ewr_coord[0], ewr_coord[1])
    dropoff_ewr = haversine(ewr_coord[0], ewr_coord[1], dropoff_lat, dropoff_lon) 
    pickup_lga = haversine(pickup_lat, pickup_lon, lga_coord[0], lga_coord[1]) 
    dropoff_lga = haversine(lga_coord[0], lga_coord[1], dropoff_lat, dropoff_lon) 
    
    df['jfk_dist'] = pd.concat([pickup_jfk, dropoff_jfk], axis=1).min(axis=1)
    df['ewr_dist'] = pd.concat([pickup_ewr, dropoff_ewr], axis=1).min(axis=1)
    df['lga_dist'] = pd.concat([pickup_lga, dropoff_lga], axis=1).min(axis=1)
    
    return df
    
df_train = add_airport_dist(df_train)

def is_valid(p_lat, p_long, d_lat, d_long):
    bounds = (-74.5, -72.8, 40.5, 41.8)
    if ((p_long >= bounds[0]) & (p_long <= bounds[1]) & (p_lat >= bounds[2]) & (p_lat <= bounds[3])):
        if (d_long >= bounds[0]) & (d_long <= bounds[1]) & (d_lat >= bounds[2]) & (d_lat <= bounds[3]):
            return 0
    return 1

valid_vec = np.vectorize(is_valid)
df_train['invalid'] = valid_vec(df_train['pickup_latitude'], df_train['pickup_longitude'], df_train['dropoff_latitude'], df_train['dropoff_longitude'])
df_train.head()

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
    nyc_mask = plt.imread('../img/nyc_water_mask.png')[:,:,0] > 0.9

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


def make_invalid_water(invalid_col):
    if (invalid_col == 1):
        return 1
    else:
        return 2
inv_vec = np.vectorize(make_invalid_water)

def get_water_invalid(df):
    df2 = remove_points_in_water(df)
    df_diff = pd.concat([df, df2])
    print("Concatenated dataframes")
    df_diff = df_diff.drop_duplicates(keep=False)
    print("dropped duplicates")
    df_diff['invalid'] = inv_vec(df_diff.invalid)
    df = pd.concat([df2, df_diff])
    df.reset_index(inplace = True)
    return df

df_train = get_water_invalid(df_train)
df_train.drop(columns = ['index'], inplace = True)


params = {
    'max_depth': 8,
    'eta':.03,
    'subsample': 1, 
    'colsample_bytree': 0.8,
    'objective':'reg:linear',
    'eval_metric':'rmse',
    'silent': 0
}

for i in range(20):
    X = df_train.sample(frac = 1, replace = True)
    y = X['fare_amount']
    X = X.drop(columns = ['fare_amount'])
    print ("Iteration Number = "+ str(i))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    gbm = lgb.train(params,
                    lgb_train,
                    valid_sets=lgb_eval,
                    num_boost_round=300,
                    early_stopping_rounds=10)
    gbm.save_model('bootstrap_model_new_v'+ str(i) + '.txt')


y = df_train['fare_amount']
train = df_train.drop(columns=['fare_amount'])
x_train,x_test,y_train,y_test = train_test_split(train,y,random_state=0,test_size=0.01)

# Trained an XGBoost model for experimentation
def XGBmodel(x_train,x_test,y_train,y_test,params):
    matrix_train = xgb.DMatrix(x_train,label=y_train)
    matrix_test = xgb.DMatrix(x_test,label=y_test)
    model=xgb.train(params=params,
                    dtrain=matrix_train,num_boost_round=5000, 
                    early_stopping_rounds=10,evals=[(matrix_test,'test')])
    return model

model = XGBmodel(x_train,x_test,y_train,y_test,params)
joblib.dump(model, 'models/xgboost/xgboost_model.dat')

df_test = pd.read_csv("data/test.csv", low_memory = False)
df_test = df_test.drop(columns = ['passenger_count'])
time_column = df_test['pickup_datetime'].to_numpy()
keys = df_test['key']
df_test.drop(columns = ['pickup_datetime', 'key'], inplace = True)
df_test = df_test.fillna(df_test.mean())

df_test['year'] = year_vec(time_column)
df_test['month'] = month_vec(time_column)
df_test['weekday'] = day_of_week_vec(time_column)
df_test['hour'] = hour_of_day_vec(time_column)

df_test['distance'] = haversine(df_test['pickup_latitude'], df_test['pickup_longitude'], df_test['dropoff_latitude'] , df_test['dropoff_longitude'])
df_test = add_airport_dist(df_test)
df_test['invalid'] = valid_vec(df_test['pickup_latitude'], df_test['pickup_longitude'], df_test['dropoff_latitude'], df_test['dropoff_longitude'])

df_test['key'] = keys
df_test = get_water_invalid(df_test)
df_test.drop(columns = ['index', 'key'], inplace = True)


dtest = xgb.DMatrix(df_test)
ypred = model.predict(dtest, ntree_limit=model.best_ntree_limit)
df_final = pd.DataFrame({'key':keys, 'fare_amount':ypred})
df_final.to_csv(r'/predictions/xgboost.csv', index = False)


def get_sum(v1, v2):
    return v1 + v2

vect = np.vectorize(get_sum)
def get_avg(v1):
    return v1/10

vect = np.vectorize(get_sum)
vect2 = np.vectorize(get_avg)
fare = [0]*11084772
fare = np.asarray(fare)
for i in range(20):
    df = pd.read_csv('predictions/bootstrap/model_new_v' + str(i) + '.csv')
    key = df['key']
    pred_fare = df['fare_amount']
    fare = vect(fare, pred_fare)
    
fare = vect2(fare)
df_final = pd.DataFrame({'key':key, 'fare_amount':fare})
df_final.to_csv(r'predictions/bootstrap/merged_preds.csv', index = False)