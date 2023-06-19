# Importing the required libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# Load the dataset
dataset = datasets.load_iris()

# Split the dataset into features and target variables
X = dataset.data
y = dataset.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a logistic regression classifier
classifier = LogisticRegression()

# Train the classifier on the training data
classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = classifier.predict(X_test)

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Calculate performance metrics
total_samples = len(y_test)
true_positive = cm[1, 1]
true_negative = cm[0, 0]
false_positive = cm[0, 1]
false_negative = cm[1, 0]

accuracy = (true_positive + true_negative) / total_samples
misclassification = (false_positive + false_negative) / total_samples
precision = true_positive / (true_positive + false_positive)
recall = true_positive / (true_positive + false_negative)
f1_score = 2 * (precision * recall) / (precision + recall)
sensitivity = recall
specificity = true_negative / (true_negative + false_positive)

print("Accuracy:", accuracy)
print("Misclassification Rate:", misclassification)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)

# Additional classification report
target_names = dataset.target_names
classification_rep = classification_report(y_test, y_pred, target_names=target_names)
print("\nClassification Report:")
print(classification_rep)
