import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# %matplotlib inline
X = pd.read_csv('/training.csv', index_col='Instance', engine='c')
test_data = pd.read_csv('/testing.csv', index_col='Instance', engine='c')
y = X.Income

# Dropping useless features
X.drop(['WearsGlasses', 'HairColor'], inplace=True, axis=1)
test_data.drop(['WearsGlasses', 'HairColor'], inplace=True, axis=1)

# Split data into 2 parts
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=1)

# Imputating in missing values
# Make copy to avoid changing original data (when imputing)
X_train_plus = X_train.copy()
X_valid_plus = X_valid.copy()

# relace creates new temporary object, doesn't modify itself
# conversting all No and 0 from University degree to NaN and then do simple imputing on all
X_train_plus.UniversityDegree.replace('No', np.nan, inplace=True)
X_valid_plus.UniversityDegree.replace('No', np.nan, inplace=True)
test_data.UniversityDegree.replace('No', np.nan, inplace=True)

X_train_plus.UniversityDegree.replace('0', np.nan, inplace=True)
X_valid_plus.UniversityDegree.replace('0', np.nan, inplace=True)
test_data.UniversityDegree.replace('0', np.nan, inplace=True)
# converting all unknown and 0 from Gender to Nan and then do simple imputing on all
X_train_plus.Gender.replace('unknown', np.nan, inplace=True)
X_valid_plus.Gender.replace('unknown', np.nan, inplace=True)
test_data.Gender.replace('unknown', np.nan, inplace=True)

X_train_plus.Gender.replace('0', np.nan, inplace=True)
X_valid_plus.Gender.replace('0', np.nan, inplace=True)
test_data.Gender.replace('0', np.nan, inplace=True)
# print('Profession',X_train_plus.Profession.value_counts())
# print('SizeOfCity',X_train_plus.SizeOfCity.value_counts())
# print('UniversityDegree',X_train_plus.UniversityDegree.value_counts())
# print('Country',X_train_plus.Country.value_counts())
# print('Gender',X_train_plus.Gender.value_counts())
# print('Age',X_train_plus.Age.value_counts())
# print('*********',X_train_plus.isnull().sum())
# print('*********',test_data.isnull().sum())


# Make new columns indicating what will be imputed
cols_with_missing = [col for col in X.columns if X[col].isnull().any()]
for col in cols_with_missing:
    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
    X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()
    test_data[col + '_was_missing'] = test_data[col].isnull()

# s = (imputed_X_train_plus.dtypes == 'object')
# object_cols = list(s[s].index)
num = (X_train_plus.dtypes != 'object')
object_cols_num = list(num[num].index)
# Imputation
my_imputer = SimpleImputer(strategy='mean')
imputed_X_train_plus_num = pd.DataFrame(my_imputer.fit_transform(X_train_plus[object_cols_num]))
imputed_X_valid_plus_num = pd.DataFrame(my_imputer.transform(X_valid_plus[object_cols_num]))
imputed_test_data_plus_num = pd.DataFrame(my_imputer.transform(test_data[object_cols_num]))

ch = (X_train_plus.dtypes == 'object')
object_cols_ch = list(ch[ch].index)
my_imputer_c = SimpleImputer(strategy='constant')
imputed_X_train_plus_ch = pd.DataFrame(my_imputer_c.fit_transform(X_train_plus[object_cols_ch]))
imputed_X_valid_plus_ch = pd.DataFrame(my_imputer_c.transform(X_valid_plus[object_cols_ch]))
imputed_test_data_plus_ch = pd.DataFrame(my_imputer_c.transform(test_data[object_cols_ch]))

imputed_X_train_plus = pd.concat([imputed_X_train_plus_num, imputed_X_train_plus_ch], axis=1)
imputed_X_valid_plus = pd.concat([imputed_X_valid_plus_num, imputed_X_valid_plus_ch], axis=1)
imputed_test_data_plus = pd.concat([imputed_test_data_plus_num, imputed_test_data_plus_ch], axis=1)

print(imputed_test_data_plus.describe())

print(object_cols_num)
print(object_cols_ch)
print(X_train_plus.columns)
print()
cols_titles = ['YearOfRecord', 'Age', 'SizeOfCity', 'BodyHeight', 'Income', 'YearOfRecord_was_missing',
               'Gender_was_missing', 'Age_was_missing', 'Profession_was_missing', 'UniversityDegree_was_missing',
               'Gender', 'Country', 'Profession', 'UniversityDegree']
# Imputation removed column names; put them back

# need to put proper columns
imputed_X_train_plus.columns = cols_titles
imputed_X_valid_plus.columns = cols_titles
imputed_test_data_plus.columns = cols_titles

imputed_X_train_plus.drop('UniversityDegree', axis=1, inplace=True)
imputed_X_valid_plus.drop('UniversityDegree', axis=1, inplace=True)
imputed_test_data_plus.drop('UniversityDegree', axis=1, inplace=True)
imputed_X_train_plus.drop('Age_was_missing', axis=1, inplace=True)
imputed_X_valid_plus.drop('Age_was_missing', axis=1, inplace=True)
imputed_test_data_plus.drop('Age_was_missing', axis=1, inplace=True)
imputed_X_train_plus.drop('Gender', axis=1, inplace=True)
imputed_X_valid_plus.drop('Gender', axis=1, inplace=True)
imputed_test_data_plus.drop('Gender', axis=1, inplace=True)
imputed_X_train_plus.drop('Profession', axis=1, inplace=True)
imputed_X_valid_plus.drop('Profession', axis=1, inplace=True)
imputed_test_data_plus.drop('Profession', axis=1, inplace=True)

# print('Profession',len(concat_df2.Profession.unique()))
# # print('SizeOfCity',len(concat_df2.SizeOfCity.unique()))
# print('UniversityDegree',len(concat_df2.UniversityDegree.unique()))
# print('Country',len(concat_df2.Country.unique()))
# print('Gender',len(concat_df2.Gender.unique()))
# print('Age',len(concat_df2.Age.unique()))

# below line to avoid : SettingWithCopyWarning: A value is trying to be set on a copy of a slice from a DataFrame
pd.options.mode.chained_assignment = None

imputed_X_train_plus['train'] = 0
imputed_X_valid_plus['train'] = 1
imputed_test_data_plus['train'] = 2

copy_store = imputed_test_data_plus.index
concat_df1 = pd.concat([imputed_X_train_plus, imputed_X_valid_plus], axis=0)
concat_df2 = pd.concat([concat_df1, imputed_test_data_plus], axis=0)

object_cols = ['Country', 'Gender_was_missing', 'Profession_was_missing', 'UniversityDegree_was_missing']



label_X_train = concat_df2.copy()





def calc_smooth_mean(df, by, on, m):
    # Compute the global mean
    mean = df[on].mean()

    # Compute the number of values and the mean of each group
    agg = df.groupby(by)[on].agg(['count', 'mean'])
    counts = agg['count']
    means = agg['mean']

    # Compute the "smoothed" means
    smooth = (counts * means + m * mean) / (counts + m)

    # Replace each value by the according smoothed mean
    return df[by].map(smooth)



for col in object_cols:
    label_X_train[col] = calc_smooth_mean(label_X_train, by=col, on='Income', m=1)




from sklearn import preprocessing

object_cols_num_scalar = ['SizeOfCity', 'BodyHeight', 'Age']
min_max_scaler = preprocessing.MinMaxScaler()

for col in object_cols_num_scalar:
    label_X_train[col] = min_max_scaler.fit_transform(label_X_train[col].values.reshape(-1, 1))

use_X = label_X_train[label_X_train['train'] == 0]
use_y = label_X_train[label_X_train['train'] == 1]
use_predict = label_X_train[label_X_train['train'] == 2]
use_X.drop(['train', 'Income'], axis=1, inplace=True)
use_y.drop(['train', 'Income'], axis=1, inplace=True)
use_predict.drop(['train', 'Income'], axis=1, inplace=True)
# print(np.shape(use_X)[0])
# print(np.shape(use_y)[0])
# train and predict
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

#random forest isn't best fit
# model = RandomForestRegressor(bootstrap= True,
#  max_depth= 40,
#  max_features= 5,
#  min_samples_leaf= 4,
#  min_samples_split= 10,
#  n_estimators= 10000)

from lightgbm import LGBMRegressor

model = LGBMRegressor(random_state=40, n_estimators=2100, boosting_type='gbdt', learning_rate=0.0072,
                      num_leaves=30,
                      max_depth=-10, seed=400)

#parameters not best fit 
# model = LGBMRegressor(random_state=40,objective = "binary",
#     boosting = "gbdt",
#     metric="auc",
#     boost_from_average=False,
#     num_threads=8,
#     learning_rate =0.0081,
#     num_leaves =13,
#     max_depth=-1,
#     feature_fraction =0.041,
#     bagging_freq =5,
#     bagging_fraction =0.331,
#     min_data_in_leaf =80,
#     min_sum_hessian_in_leaf =10.0,
#     verbosity =1,
#     n_estimators=786,
#     seed=40)



model.fit(use_X, y_train)


preds = model.predict(use_y)


score = mean_absolute_error(y_valid, preds)

from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(y_valid, preds))
print('MAE:', score)
print('RMS:', rms)

preds_test = model.predict(use_predict)

output = pd.DataFrame({'Instance': copy_store,
                       'Income': preds_test})
output.to_csv('/submission.csv', index=False)






