import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from pandas.core.common import random_state
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr

# Get dataset
df_qual = pd.read_csv(r'E:\MS\Machine learning\paper\UML class diagrams for layout quality checking\Data\features_labels.csv')
df_qual.head()

# Describe data
df_qual.describe()

# Data distribution
plt.title('Quality of layout Plot')
sns.histplot(df_qual['Quality'])
plt.show()

plt.scatter(df_qual['RectOrth2'], df_qual['Quality'], color = 'lightcoral')
plt.title('Quality vs RectOrth2')
plt.xlabel('RectOrth2')
plt.ylabel('Quality')
plt.box(False)
plt.show()

# Splitting variables
X = df_qual.iloc[:, 1 :]  # independent
y = df_qual.iloc[:, 1:]  # dependent

# Splitting dataset into test/train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Regressor model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Prediction result
y_pred_test = regressor.predict(X_test)     # predicted value of y_test
y_pred_train = regressor.predict(X_train)   # predicted value of y_train

# Prediction on test set
plt.scatter(X_test, y_test, color = 'lightcoral')
plt.plot(X_train, y_pred_train, color = 'firebrick')
plt.title('Quality vs RectOrth2 (Test Set)')
plt.xlabel('RectOrth2')
plt.ylabel('Quality')
plt.legend(['X_train/Pred(y_test)', 'X_train/y_train'], title = 'Sal/Exp', loc='best', facecolor='white')
plt.box(False)
plt.show()

# Regressor coefficients and intercept
print(f'Coefficient: {regressor.coef_}')
print(f'Intercept: {regressor.intercept_}')

y_test_np = np.array(y_test)
y_pred_test_np = np.array(y_pred_test)

# Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test_np, y_pred_test_np)
print("Mean Absolute Error (MAE):", mae)

# Relative Absolute Error (RAE)
rae = np.sum(np.abs(y_test_np - y_pred_test_np)) / np.sum(np.abs(y_test_np - np.mean(y_test_np)))
print("Relative Absolute Error (RAE):", rae)

# Pearson Correlation Coefficient (PCC)
pcc, _ = pearsonr(y_test_np.flatten(), y_pred_test_np.flatten())
print("Pearson Correlation Coefficient (PCC):", pcc)