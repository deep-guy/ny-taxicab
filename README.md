# What is the fare boys?
## Pre-processing
`preprocess.py` takes in a path to a dataframe (csv) and a target path to save the pre-processed dataframe
``` shell
python3 preprocess.py train/test RAW_DATA_PATH TARGET_DATA_PATH
```
Currently, `preprocess(df)` preprocesses the data and `scale(df)` scales the data
