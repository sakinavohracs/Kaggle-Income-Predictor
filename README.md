# Kaggle-Income-Predictor



#Import : 
```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
rom sklearn import preprocessing
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from math import sqrt
```

For Missing Values : 
- Converted all unknown values, (0, unknown, no) to #N/A
- Performed Simple Imputer from Sklearn to fill missing values. 

For Pre-processing : 
- Target Encoding on Categorical Columns (using smooth_cal_mean function)[1] 
- MinMaxScalar mean on Numerical Columns 

Prediction : 
LGBMRegressor

Best fit parameters : [2]
random_state=40, 
n_estimators=2100, 
boosting_type='gbdt', 
learning_rate=0.0072,
num_leaves=30,
max_depth=-10,
seed=400


References :
[1] https://maxhalford.github.io/blog/target-encoding-done-the-right-way/
[2] https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html
