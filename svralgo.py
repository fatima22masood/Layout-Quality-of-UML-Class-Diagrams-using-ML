import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# Read in data and display first 5 rows
features = pd.read_csv(r'E:\MS\Machine learning\paper\UML class diagrams for layout quality checking\Data\features_labels.csv')

# Remove the 'id' column
features = features.drop('id', axis=1)

# One-hot encode the data using pandas get_dummies
features = pd.get_dummies(features)

# Labels are the values we want to predict
labels = np.array(features['Quality'])

# Remove the labels from the features
features = features.drop('Quality', axis=1)

# Saving feature names for later use
feature_list = list(features.columns)

# Convert to numpy array
features = np.array(features)

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.25, random_state=42)

# Create a pipeline with SVR
svr_model = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))

# Train the SVR model on training data
svr_model.fit(train_features, train_labels)

# Use the SVR model to make predictions on the test data
svr_predictions = svr_model.predict(test_features)

# Calculate the absolute errors
svr_errors = abs(svr_predictions - test_labels)

# Print out the mean absolute error (mae)
print('SVR Mean Absolute Error:', round(np.mean(svr_errors), 2), 'degrees.')

# Calculate mean absolute percentage error (MAPE)
svr_mape = 100 * (svr_errors / test_labels)

# Calculate and display accuracy
svr_accuracy = 100 - np.mean(svr_mape)
print('SVR Accuracy:', round(svr_accuracy, 2), '%.')

# Plotting the SVR predictions
plt.scatter(test_labels, svr_predictions)
plt.xlabel('True Values')
plt.ylabel('SVR Predictions')
plt.title('SVR Predictions vs True Values')
plt.show()

# Convert Quality values to integer class labels (adjust as needed)
predicted_classes = np.round(svr_predictions).astype(int)
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
