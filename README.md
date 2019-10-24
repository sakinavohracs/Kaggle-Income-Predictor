# Kaggle-Income-Predictor



# Import all below libaries: 
```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from math import sqrt
```
# Reading Training and Test datasets using Dataframes. 
```
X = pd.read_csv('/training.csv', index_col='Instance', engine='c') #X has training data set 
test_data = pd.read_csv('/testing.csv', index_col='Instance', engine='c') #we have to predict values for this data set
y = X.Income #y is target
```

# Drop columns that has least effect on target
```
#Analysed from Heatmaps, plots and correlation matrix
X.drop(['WearsGlasses', 'HairColor'], inplace=True, axis=1)
test_data.drop(['WearsGlasses', 'HairColor'], inplace=True, axis=1)
```
# Split the train and test sets
```
# using sklearn.model_selection library
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=1)
```
# For Missing Values : 
- Converted all unknown values, (0, unknown, no) to #N/A. Perform for Train, test and validation sets.
```
X_train_plus.UniversityDegree.replace('No', np.nan, inplace=True)
X_train_plus.UniversityDegree.replace('0', np.nan, inplace=True)
X_train_plus.UniversityDegree.replace('unknown', np.nan, inplace=True)
```
- Performed Simple Imputer from Sklearn to fill missing values. 
Please refer [Simple Imputer](https://www.kaggle.com/alexisbcook/missing-values) to better understand Imputation approaches. 



# For Pre-processing : 
- Target Encoding on Categorical Columns (using smooth_cal_mean function). Please refer [Target Encoding](https://maxhalford.github.io/blog/target-encoding-done-the-right-way/) to undertsand target encoding with regularization to avoid over fitting.
- MinMaxScalar mean on Numerical Columns


# Prediction Model: 
LGBMRegressor
Please refer [LGBMRegressor] (https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html) for LGBMRegressor Documentation

```
model = LGBMRegressor(random_state=40, n_estimators=2100, boosting_type='gbdt', learning_rate=0.0072,
                      num_leaves=30,
                      max_depth=-10, seed=400)

```



