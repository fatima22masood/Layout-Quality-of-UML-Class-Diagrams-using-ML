import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, mean_absolute_error
from scipy.stats import pearsonr

# Get dataset
df_qual = pd.read_csv(r'E:\MS\Machine learning\paper\UML class diagrams for layout quality checking\Data\features_labels.csv')
df_qual.head()

# Assuming df_qual is your DataFrame
X = df_qual[['RectOrth2']]
y = df_qual['Quality']

# Create a linear regression model
regressor = LinearRegression()

# Define a custom scoring function for RAE
def rae_scorer(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true - np.mean(y_true)))

# Define the scoring metrics
scoring = {
    'MAE': 'neg_mean_absolute_error',
    'RAE': make_scorer(rae_scorer, greater_is_better=False),
    'PCC': 'r2'
}

# Perform 10-fold cross-validation
cv = KFold(n_splits=10, shuffle=True, random_state=0)

# Calculate scores
cv_results = cross_validate(regressor, X, y, cv=cv, scoring=scoring, return_train_score=False)

# Extract and print results
mae_mean = -cv_results['test_MAE'].mean()
rae_mean = -cv_results['test_RAE'].mean()
pcc_mean = cv_results['test_PCC'].mean()

print(f"Mean Absolute Error (MAE): {mae_mean}")
print(f"Relative Absolute Error (RAE): {rae_mean}")
print(f"Pearson Correlation Coefficient (PCC): {pcc_mean}")
