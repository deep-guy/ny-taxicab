### Initial approach: Some basic data visualisations
- Deciding features: cyclising the time data, why we did it.
- Some basic noise removal from data: removing passengers>6, null values
- More noise removal: Putting latitude and longitude values inside a bounding box for New york
- Even better noise removal: Removing trip points that fall in the hudson
- Scaling the entire dataset.
- Training LightGBM boosting regression model: why we chose that
- Results: All of them fall in the same range. Why is this happening? (scaling + we haven't taken distance either)

### Approach 2: Simpler features.
- Taking non-cyclised, non-scaled time features.
- Creating a distance feature using the haversine formula
- Removing 0 distance values
- Training a model with our new features, and submitting on kaggle
- Very bad rmse on test data (>9). Why is this happening?
- Only possible explanation: too many noisy values in test data

### Approach 3: Keeping Noisy data
- Created a new column labelled 'invalid'
    - If a point in valid: Invalid value = 0
    - If a point is outside New York: Invalid value = 1
    - If a point lies in a water body: Invalid value = 2
- Some additional research into New York taxis shows that the rides to airports have fixed fare
- Therefore, added a new feature 'distance_to_JFK'
- Training on LightGBM: Best score yet

### Approach 4: k-means clustering
- Did k means clustering taking 20% of the total data for training for each models
- no. of folds = 10
- 10 models created. Model with best result on test data chosen for submission.
- Did not perform too well on unseen test data. Possible explanation: Overfitting

### Approach 4: Boosting results 
- There are 3 airports in NY: Distance of pickup to each added as a feature.
- Trained model on XGBoost

### Final Improvements
- Took a bootstrapping approach to making predictions
- Analysis of what causes error (Noise, Bias and Variance), and how bootstrapping reduces Variance
- Final results after training 10 LightGBM models

        ---- FIN ----
