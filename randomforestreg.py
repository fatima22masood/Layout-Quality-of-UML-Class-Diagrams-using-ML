
import pandas as pd
# Read in data and display first 5 rows
features = pd.read_csv(r'E:\MS\Machine learning\paper\UML class diagrams for layout quality checking\Data\features_labels.csv')
# Remove the 'id' column
features = features.drop('id', axis=1)


features.head(5)
print('The shape of our features is:', features.shape)

# Descriptive statistics for each column
features.describe()
# One-hot encode the data using pandas get_dummies
features = pd.get_dummies(features)
# Display the first 5 rows of the last 12 columns
features.iloc[:,5:].head(5)

# Use numpy to convert to arrays
import numpy as np
# Labels are the values we want to predict
labels = np.array(features['Quality'])
# Remove the labels from the features
# axis 1 refers to the columns
features= features.drop('Quality', axis = 1)
# Saving feature names for later use
feature_list = list(features.columns)
# Convert to numpy array
features = np.array(features)

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

import matplotlib.pyplot as plt

# Shapes of your data
shapes = [(train_features.shape[0], train_features.shape[1]),
          (train_labels.shape[0],),
          (test_features.shape[0], test_features.shape[1]),
          (test_labels.shape[0],)]

# Extracting number of samples
num_samples = [shape[0] for shape in shapes]

# Extracting dataset names
datasets = ['Training Features', 'Training Labels', 'Testing Features', 'Testing Labels']

# Plotting
plt.bar(datasets, num_samples, color=['blue', 'orange', 'green', 'red'])
plt.ylabel('Number of Samples')
plt.title('Shapes of Training and Testing Data')
plt.show()

# The baseline predictions are the historical averages
baseline_preds = test_features[:, feature_list.index('RectOrth2')]
# Baseline errors, and display average baseline error
baseline_errors = abs(baseline_preds - test_labels)
print('RectOrth2 baseline error: ', round(np.mean(baseline_errors), 2))

# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(train_features, train_labels);

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)
# Calculate the absolute errors
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

import os

# Replace this with the actual path to the 'dot' executable in your Graphviz installation.
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'

# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot
# Pull out one tree from the forest
tree = rf.estimators_[5]
# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot
# Pull out one tree from the forest
tree = rf.estimators_[5]
# Export the image to a dot file
export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')
# Write graph to a png file
graph.write_png('tree.png')

# Limit depth of tree to 3 levels
rf_small = RandomForestRegressor(n_estimators=10, max_depth = 3)
rf_small.fit(train_features, train_labels)
# Extract the small tree
tree_small = rf_small.estimators_[5]
# Save the tree as a png image
export_graphviz(tree_small, out_file = 'small_tree.dot', feature_names = feature_list, rounded = True, precision = 1)
(graph, ) = pydot.graph_from_dot_file('small_tree.dot')
graph.write_png('small_tree.png');

# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

# New random forest with only the two most important variables
rf_most_important = RandomForestRegressor(n_estimators= 1000, random_state=42)
# Extract the two most important features
important_indices = [feature_list.index('LongestLine'), feature_list.index('RectOrth')]
train_important = train_features[:, important_indices]
test_important = test_features[:, important_indices]
# Train the random forest
rf_most_important.fit(train_important, train_labels)
# Make predictions and determine the error
predictions = rf_most_important.predict(test_important)
errors = abs(predictions - test_labels)
# Display the performance metrics
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
mape = np.mean(100 * (errors / test_labels))
accuracy = 100 - mape
print('Accuracy:', round(accuracy, 2), '%.')


# Set the style
plt.style.use('fivethirtyeight')
# list of x locations for plotting
x_values = list(range(len(importances)))
# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical')
# Tick labels for x axis
plt.xticks(x_values, feature_list, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances')
plt.show()


import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# Convert Quality values to integer class labels (adjust as needed)
predicted_classes = np.round(predictions).astype(int)
true_classes = np.round(test_labels).astype(int)

# Calculate confusion matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes)

# Plot the confusion matrix using seaborn
plt.figure(figsize=(8, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=[str(i) for i in range(1, 6)],
            yticklabels=[str(i) for i in range(1, 6)])
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title('Confusion Matrix')
plt.show()


accuracy = accuracy_score(true_classes, predicted_classes)
precision = precision_score(true_classes, predicted_classes, average='weighted')
recall = recall_score(true_classes, predicted_classes, average='weighted')

# Print performance metrics
print("\nAccuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

metrics = ['Accuracy', 'Precision', 'Recall']
values = [accuracy, precision, recall]

plt.bar(metrics, values, color=['blue', 'orange', 'green'])
plt.ylabel('Score')
plt.title('Model Performance Metrics')
plt.show()

