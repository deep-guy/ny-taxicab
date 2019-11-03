import datetime
import numpy as np

def getNewString(s):
    # Extracting the values from the string
    s = s.strip()
    lst = s.split(',')
    if (len(lst) != 8):
        return str(lst[0]) + "0,0,0,0,0,0,0"
    pickup_lst = lst[2].strip().split(' ')
    if (len(pickup_lst) != 3):
        return 0

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
    ans = lst[0] + ',' + lst[1] + ',' + "{0:.5f}".format(minutes_sin) + ',' + "{0:.5f}".format(minutes_cos) + ',' + "{0:.3f}".format(weekday_sin) + ',' + "{0:.3f}".format(weekday_cos) + ',' + "{0:.4f}".format(month_sin) + ',' + "{0:.4f}".format(month_cos) + ',' + lst[3] + ',' + lst[4] + ',' + lst[5] + ',' + lst[6] + ',' + lst[7]
    return ans

fname = "data/train.csv"
newfname = "data/train_mod.csv"

with open(fname, 'r') as fp1:
    fp2 = open(newfname, 'w')
    line = fp1.readline()
    line = fp1.readline()
    fp2.write("key,fare_amount,minutes_sin,minutes_cos,weekday_sin,weekday_cos,month_sin,month_cos,pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude,passenger_count\n")
    while line:
        ss = getNewString(line)
        if (ss != 0):
            fp2.write(getNewString(line) + '\n')
        line = fp1.readline()
    fp2.close()
