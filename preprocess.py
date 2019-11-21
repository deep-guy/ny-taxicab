import datetime
import pandas as pd
import numpy as np

def minutes_after_midnight(date_time):
    time = date_time.split(' ')
    if len(time) == 3:
        time_arr = time[1].split(':')
        minutes_after_midnight = 60 * int(time_arr[0]) + int(time_arr[1])
        return 2 * np.pi * minutes_after_midnight / 1440
    else:
        return np.nan

def day_of_week(date_time):
    time = date_time.split(' ')
    if len(time) == 3:
        date_arr = time[0].split('-')
        date = datetime.date(int(date_arr[0]), int(date_arr[1]), int(date_arr[2]))
        return 2 * np.pi * (date.weekday() + 1) / 7
    else:
        return np.nan

def month(date_time):
    time = date_time.split(' ')
    if len(time) == 3:
        return 2 * np.pi * int(time[0].split('-')[1]) / 12
    else:
        return np.nan

def year(date_time):
    time = date_time.split(' ')
    if len(time) == 3:
        return float(time[0].split('-')[0])
    else:
        return np.nan

time_of_day_vec = np.vectorize(minutes_after_midnight)
day_of_week_vec = np.vectorize(day_of_week)
month_vec = np.vectorize(month)
year_vec = np.vectorize(year)

def preprocess(df):
    # Drop all null values
    df = df.dropna()

    # Cyclise time and remove key column
    time_column = df['pickup_datetime'].to_numpy()
    df = df.drop(columns=['pickup_datetime', 'key'])

    df['time_of_day'] = time_of_day_vec(time_column)
    df['day_of_week'] = day_of_week_vec(time_column)
    df['month'] = month_vec(time_column)
    df['year'] = year_vec(time_column)

    # Remove 0 passenger count
    df = df[df['passenger_count'] > 0]
    return df


# Function to calculate distance between two points and add it as a feature
import haversine
def distance(df):
    rows = df.shape[0]
    dist_arr = [0]*rows
    for i in range(rows):
        pickup = (df.pickup_latitude[i], df.pickup_longitude[i]
        dropoff = (df.dropoff_latitude[i], df.dropoff_longitude[i])
        dist_arr[i] = haversine(pickup, dropoff)
    
    df['distance'] = dist_arr
    return df



# Points in wata are bad..
import matplotlib.pyplot as plt
nyc_bounds = (-74.5, -72.8, 40.5, 41.8)

def select_within_bounds(df, bounds):
    pickup_indices = (df.pickup_longitude >= bounds[0]) & (df.pickup_longitude <= bounds[1]) & \
        (df.pickup_latitude >= bounds[2]) & (df.pickup_longitude <= bounds[3])

    dropoff_indices = (df.dropoff_longitude >= bounds[0]) & (df.dropoff_longitude <= bounds[1]) & \
        (df.dropoff_latitude >= bounds[2]) & (df.dropoff_longitude <= bounds[3])

    return pickup_indices & dropoff_indices

def map_to_nyc_mask(longitude, latitude, points_x, points_y, bounds):
    x = (points_x * (longitude - bounds[0]) / (bounds[1] - bounds[0])).astype('int')
    y = (points_y - points_y * (latitude - bounds[2]) / (bounds[3] - bounds[2])).astype('int')
    return x,y

def remove_points_in_water(df):
    # Create a mask of the New York City with 1 as land and 0 as water
    nyc_mask = plt.imread('img/nyc_map.png')[:,:,0] > 0.9

    # Remove points outside New York
    df = df[select_within_bounds(df, nyc_bounds)]

    # Map the latitudes and longitudes to the points in the map
    pickup_x, pickup_y = map_to_nyc_mask(df.pickup_longitude, df.pickup_latitude, nyc_mask.shape[1], nyc_mask.shape[0], nyc_bounds)
    dropoff_x, dropoff_y = map_to_nyc_mask(df.dropoff_longitude, df.dropoff_latitude, nyc_mask.shape[1], nyc_mask.shape[0], nyc_bounds)

    # Compute the indices where pickup and dropoff locations are on land
    indices = nyc_mask[pickup_x, pickup_y] & nyc_mask[dropoff_x, dropoff_y]

    return df[indices]

if __name__=="__main__":
    import sys
    path = sys.argv[1]
    target_path = sys.argv[2]
    df = pd.read_csv(path, low_memory=False)
    preprocessed_df = preprocess(df)
    preprocessed_df.to_csv(target_path, index=False)
