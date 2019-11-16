# What is the fare boys?
## Pre-processing
`preprocess.py` takes in a path to a dataframe (csv) and a target path to save the pre-processed dataframe
``` shell
python3 preprocess.py RAW_DATA_PATH TARGET_DATA_PATH
```
Currently, `preprocess(df)` chucks the null values, cyclises time and chucks all cabs with 0 passengers.
Add outlier detection here..
