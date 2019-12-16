# New York Taxicab: Predict the fare

### Structure
- Notebooks: All the notebooks for different approaches and data preprocessing
- Code: Code for every submission compiled into a single python file
- final_submission: Folder containing the codes for every approach + the corresponding trained model
- report_plan.md: Brief overview of things to cover in the report
- report.pdf: Final report of the project

### Pre-processing
`preprocess.py` takes in a path to a dataframe (csv) and a target path to save the pre-processed dataframe
``` shell
python3 preprocess.py train/test RAW_DATA_PATH TARGET_DATA_PATH
```
Currently, `preprocess(df)` preprocesses the data and `scale(df)` scales the data
