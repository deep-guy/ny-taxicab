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

    time_of_day = time_of_day_vec(time_column)
    day_of_week = day_of_week_vec(time_column)
    month = month_vec(time_column)
    year = year_vec(time_column)

    df['sin_time_of_day'] = np.sin(time_of_day)
    df['cos_time_of_day'] = np.cos(time_of_day)
    df['sin_day_of_week'] = np.sin(day_of_week)
    df['cos_day_of_week'] = np.cos(day_of_week)
    df['sin_month'] = np.sin(month)
    df['cos_month'] = np.cos(month)
    df['year'] = year

    # Remove 0 passenger count
    df = df[df['passenger_count'] > 0]
    return df

if __name__=="__main__":
    import sys
    path = sys.argv[1]
    target_path = sys.argv[2]
    df = pd.read_csv(path, low_memory=False)
    preprocessed_df = preprocess(df)
    preprocessed_df.to_csv(target_path, index=False)
