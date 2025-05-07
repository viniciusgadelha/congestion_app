import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARMA
from sklearn.metrics import mean_squared_error, r2_score
# from learntools.time_series.style import *
import matplotlib.pyplot as plt

df = pd.read_csv('inputs/Smart_Meters_Import_CT941.csv')
df = df.set_index(pd.to_datetime(df['time_reading'], format='%Y-%m-%d %H:%M:%S', utc=False))


plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True, figsize=(11, 5))
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)

plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
    legend=False,
)
#%%

y = df.iloc[:, 1:3]
y.fillna(0, inplace=True)

dp = DeterministicProcess(
    index=y.index,  # dates from the training data
    constant=True,  # the intercept
    order=2,        # quadratic trend
    drop=True,      # drop terms to avoid collinearity
)
X = dp.in_sample()


# Test on the years 2016-2019. It will be easier for us later if we
# split the date index instead of the dataframe directly.
idx_train, idx_test = train_test_split(y.index, test_size=24*7, shuffle=False)

X_train, X_test = X.loc[idx_train, :], X.loc[idx_test, :]
y_train, y_test = y.loc[idx_train], y.loc[idx_test]

# Fit trend model
model = LinearRegression(fit_intercept=False)
model.fit(X_train, y_train)

# Make predictions
y_fit = pd.DataFrame(model.predict(X_train), index=y_train.index, columns=y_train.columns)
y_pred = pd.DataFrame(model.predict(X_test), index=y_test.index, columns=y_test.columns)

# Plot
axs = y_train.tail(1000).plot(color='0.25', subplots=True, sharex=True)
axs = y_test.tail(1000).plot(color='0.25', subplots=True, sharex=True, ax=axs)
axs = y_fit.tail(1000).plot(color='C0', subplots=True, sharex=True, ax=axs)
axs = y_pred.tail(1000).plot(color='C3', subplots=True, sharex=True, ax=axs)
for ax in axs: ax.legend([])
_ = plt.suptitle("Trends")
plt.show()

#%%

# The `stack` method converts column labels to row labels, pivoting from wide format to long
X = df.iloc[:, 1:3].stack()  # pivot dataset wide to long
y = pd.DataFrame(X.values)  # grab target series

# Turn row labels into categorical feature columns with a label encoding
X = X.reset_index('74576')
# Label encoding for 'Industries' feature
for colname in X.select_dtypes(["object", "category"]):
    X[colname], _ = X[colname].factorize()

# Label encoding for annual seasonality
X["Month"] = X.index.month  # values are 1, 2, ..., 12

# Create splits
X_train, X_test = X.loc[idx_train, :], X.loc[idx_test, :]
y_train, y_test = y.loc[idx_train], y.loc[idx_test]

#%%
df = pd.read_csv('inputs/Smart_Meters_Import_CT941.csv')

df['time'] = np.arange(len(df.index))

X = df.loc[:, ['time']]
y = df.loc[:, '74576']
y.fillna(0, inplace=True)

model = LinearRegression()
model.fit(X, y)


y_pred = pd.Series(model.predict(X), index=X.index)


fig, ax = plt.subplots()
ax.plot('time', '74576', data=df, color='0.75')
ax = sns.regplot(x='time', y='74576', data=df, ci=None, scatter_kws=dict(color='0.25'))
ax.set_title('Time Plot of Hardcover Sales');
plt.show()

ax = y.plot(alpha=0.5)
ax = y_pred.plot(ax=ax, linewidth=3)
ax.set_title('Time Plot of Total Store Sales')
plt.show()

#%%

df = pd.read_csv('inputs/Smart_Meters_Import_CT941.csv')

df['lag_1'] = df['74576'].shift(1)


X = df.loc[:, ['lag_1']]
y = df.loc[:, '74576']
y.fillna(0, inplace=True)
X.fillna(0, inplace=True)

model = LinearRegression()
model.fit(X, y)


y_pred = pd.Series(model.predict(X), index=X.index)



fig, ax = plt.subplots()
ax.plot('lag_1', '74576', data=df, color='0.75')
ax = sns.regplot(x='lag_1', y='74576', data=df, ci=None, scatter_kws=dict(color='0.25'))
ax.set_title('Time Plot of Hardcover Sales')
plt.show()


ax = y.plot(alpha=0.5)
ax = y_pred.plot(ax=ax, linewidth=3)
ax.set_title('Time Plot of Total Store Sales')
plt.show()

#%%

df = pd.read_csv('inputs/Smart_Meters_Import_CT941.csv')
plt.figure(figsize=(10, 4))
plt.plot(df['74576'])

df['lag_1'] = df['74576'].shift(1)


X = df.loc[:, ['lag_1']]
y = df.loc[:, '74576']
y.fillna(0, inplace=True)
X.fillna(0, inplace=True)

model = LinearRegression()
model.fit(X, y)


y_pred = pd.Series(model.predict(X), index=X.index)


fig, ax = plt.subplots()
ax.plot('lag_1', '74576', data=df, color='0.75')
ax = sns.regplot(x='lag_1', y='74576', data=df, ci=None, scatter_kws=dict(color='0.25'))
ax.set_title('Time Plot of Hardcover Sales')
plt.show()


ax = y.plot(alpha=0.5)
ax = y_pred.plot(ax=ax, linewidth=3)
ax.set_title('Time Plot of Total Store Sales')
plt.show()


#%%

