# Machine Learning Models - Decision Tree, Confusion Matrix, Logistic Regression

This repository contains Python code examples for implementing and understanding decision tree, confusion matrix, and logistic regression.

## Decision Tree
A decision tree is a supervised learning algorithm used for classification and regression tasks. It builds a flowchart-like structure where each internal node represents a feature or attribute, each branch represents a decision rule, and each leaf node represents the outcome or class label. Decision trees are interpretable and can handle both numerical and categorical data.

In the provided code example, scikit-learn library is used to build a decision tree classifier. The Iris dataset is used as an example, where the features include sepal length, sepal width, petal length, and petal width. The code demonstrates splitting the data into training and testing sets, training the decision tree classifier, making predictions, and calculating the accuracy of the classifier using the accuracy score.

## Confusion Matrix
A confusion matrix is a table that is used to evaluate the performance of a classification model. It provides a detailed analysis of the model's predictions by comparing the predicted labels with the actual labels. The matrix includes four values: true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN). The values from the confusion matrix can be used to calculate various performance metrics such as accuracy, precision, recall, F1 score, sensitivity, and specificity.

The provided code example demonstrates the computation of a confusion matrix using scikit-learn. It uses the Iris dataset and a logistic regression classifier to make predictions. The confusion matrix is then calculated using the predicted and actual labels. Additionally, various performance metrics such as accuracy, misclassification rate, precision, recall, F1 score, sensitivity, and specificity are computed and displayed.

## Logistic Regression
Logistic regression is a popular supervised learning algorithm used for binary classification problems. It models the relationship between the independent variables (features) and the binary dependent variable (target) using a logistic function. Logistic regression is well-suited for problems where the dependent variable is categorical. It estimates the probabilities of different classes and makes predictions based on a decision boundary.

The code example for logistic regression uses the Iris dataset and scikit-learn library. It showcases splitting the data into training and testing sets, creating a logistic regression classifier, training the classifier, making predictions, and calculating the accuracy of the classifier using the accuracy score. 

Feel free to explore and modify the code examples provided to work with your own datasets and requirements.

