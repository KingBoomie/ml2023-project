import os
import sys

import pandas as pd

os.chdir("project/ml2023-project")
os.getcwd()
sys.path.append(os.getcwd())

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer

# %%
train_df = pd.read_csv('train.csv')[100000:]
# Drip high null columsn
high_null_cols = train_df.isna().sum().sort_values(ascending=False)[:2].index

print(f"high null values: {high_null_cols}")

def drop_cols(df, cols):
    df = df.drop(cols, axis=1)
    return df

df = drop_cols(train_df, high_null_cols)
df = drop_cols(df, ['row_id'])

def imputer(df_processed):
    '''
    Function that receives a dataframe and returns the median value of each missing value in a column partitioned by stock_id,
    '''
    stock_list = list(df_processed['stock_id'].unique())

    imputer_columns = ['imbalance_size', 'reference_price', 'matched_size', 'bid_price', 'ask_price',
                       'wap', 'target', 'seconds_in_bucket', 'bid_size', 'ask_size']

    # Create a single SimpleImputer instance for each column
    imputers = {col: SimpleImputer(missing_values=np.nan, strategy='median') for col in imputer_columns}

    for stock in stock_list:
        stock_df = df_processed.loc[df_processed['stock_id'] == stock]

        # Apply imputation to each column
        for col, imputer in imputers.items():
            stock_df.loc[:,col] = imputer.fit_transform(stock_df[col].values.reshape(-1,1))

        # Update the original DataFrame with imputed values
        df_processed.loc[df_processed['stock_id'] == stock, imputer_columns] = stock_df[imputer_columns].values

    return df_processed, imputers

df_processed, imputers = imputer(df)




# %%

mask = df_processed['date_id'] < 384

y = df_processed['target']
X = df_processed.drop(['target'], axis = 1)
X = X.drop(['time_id'], axis = 1)
X_cols = X.columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = False)


# # normalize mean and std by stock_id
means = X_test.groupby('stock_id').mean().target
stds = X_test.groupby('stock_id').std().target

X_train['target'] = X_train.apply(lambda row: (row['target'] - means[row['stock_id']]) / stds[row['stock_id']], axis=1)
X_test['target'] = X_test.apply(lambda row: (row['target'] - means[row['stock_id']]) / stds[row['stock_id']], axis=1)

# %%
# Train the model
clf_fst = xgb.XGBRegressor(n_estimators = 50, eval_metric = 'mae')
clf_fst.fit(X_train, y_train)

# %%
# Predict on training and test sets, and compute MAE
preds_test = clf_fst.predict(X_test)
mean_absolute_error(y_test, preds_test)

# %%
import optiver2023
env = optiver2023.make_env()
iter_test = env.iter_test()

# %%
tests = []
revealed_targetss = []
test_preds = []

counter = 0
for (test, revealed_targets, test_preds) in iter_test:
    tests.append(test)
    # test_preds['target'] = clf.predict(test[X_cols])
    print(counter, test.shape, revealed_targets.shape, test_preds.shape)
    if counter != 0 and revealed_targets.shape[0] == 11000:
        # train!!
        last_day_test = pd.concat(tests)
        print("last_day_test.shape: ", last_day_test.shape)
        last_day_test = last_day_test.merge(revealed_targets, left_on=['stock_id', 'date_id', 'seconds_in_bucket'],
                                            right_on=['stock_id', 'revealed_date_id', 'seconds_in_bucket'], how='inner')
        last_day_test.rename(columns={'revealed_target': "target"}, inplace=True)
        print("last_day_test.shape: ", last_day_test.shape)
        revealed_targetss.append(revealed_targets)

        df_processed_last_day, imputers = imputer(last_day_test)
        # merge with training data
        X_train_new = pd.concat([df_processed, df_processed_last_day])

        print(sum(X_train_new['target'].isna()))

        clf_fst.fit(X_train_new[X_cols], X_train_new['target'])

    test_preds['target'] = clf_fst.predict(test[X_cols])
    env.predict(test_preds)

    counter += 1
tests = pd.concat(tests)

#%%
targess = pd.concat(revealed_targetss)
testssss = pd.concat(tests)
m2 = testssss.merge(targess, on=['stock_id', 'date_id', 'seconds_in_bucket'], how='inner')