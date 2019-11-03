# key,pickup_datetime,pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude,passenger_count
# 2009-01-01 00:01:04.0000003,2009-01-01 00:01:04 UTC,-73.972484,40.742743,-73.918937,40.764496,1
import datetime
import numpy as np

def getNewString(s):
    # Extracting the values from the string
    s = s.strip()
    lst = s.split(',')
    pickup_lst = lst[1].strip().split(' ')
    year_lst = pickup_lst[0].strip().split('-')
    time_lst = pickup_lst[1].strip().split(':')
    mins_after_midnight = (60 * int(time_lst[0])) + int(time_lst[1])
    month = int(year_lst[1])
    date = datetime.date(int(year_lst[0]), int(year_lst[1]), int(year_lst[2]))
    day_of_week = date.weekday() + 1

    #Making Values cyclic
    minutes_sin = np.sin(2*np.pi*mins_after_midnight/1400)
    minutes_cos = np.cos(2*np.pi*mins_after_midnight/1400)
    month_sin = np.sin(2*np.pi*month/12)
    month_cos = np.cos(2*np.pi*month/12)
    weekday_sin = np.sin(2*np.pi*day_of_week/7)
    weekday_cos = np.cos(2*np.pi*day_of_week/7)

    #Appending to create new string
    ans = lst[0] + ',' + "{0:.5f}".format(minutes_sin) + ',' + "{0:.5f}".format(minutes_cos) + ',' + "{0:.3f}".format(weekday_sin) + ',' + "{0:.3f}".format(weekday_cos) + ',' + "{0:.4f}".format(month_sin) + ',' + "{0:.4f}".format(month_cos) + ',' + lst[2] + ',' + lst[3] + ',' + lst[4] + ',' + lst[5] 
    return ans

# I had to remove the header line (feature names) in the files for this to work

fp1 = open('data/test.csv', 'r')
fp2 = open('data/test_mod.csv', 'w')
fp1.readline
fp2.write("key,minutes_sin,minutes_cos,weekday_sin,weekday_cos,month_sin,month_cos,pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude,passenger_count")
for line in fp1:
    fp2.write(getNewString(line) + '\n')
fp1.close()
fp2.close()
