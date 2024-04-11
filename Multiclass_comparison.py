# -*- coding: utf-8 -*-
"""MLCOMP_BALA.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1xz154ygOS9AJmpjgTtpZ0YnQjyEwr3Vb
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(color_codes=True)
# %matplotlib inline
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import sklearn
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import sklearn.neighbors as knn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, precision_recall_curve, classification_report,  balanced_accuracy_score, ConfusionMatrixDisplay, confusion_matrix
from xgboost import XGBClassifier
from collections import Counter 

print("\nUploading dataset:\n")
df = pd.read_csv('train_10.csv')
# df = pd.read_csv("Data/day.csv")

print("First few rows of the DataFrame:")
print(df.head())
print("\nSummary statistics of the DataFrame:")
print(df.describe())
missing_values_count = df.isnull().sum().sum()
print(f"\nTotal missing values in the DataFrame: {missing_values_count}")

# Prep features and target
X = df.drop(['starttime', 'endtime', 'user', 'day', 'week', 'insider'], axis=1).values
y = df.insider.values

print(f"\nShape of feature matrix X: {X.shape}")
print("\nCounts of different classes in the 'insider' column:")
print(df['insider'].value_counts())

# Split into train & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nShape of training set (X_train, y_train):")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print("Shape of testing set (X_test, y_test):")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

"""# SMOTE"""

from imblearn.over_sampling import SMOTE

counter = Counter(y_train)
print('Before',counter)
# oversampling the train dataset using SMOTE
smt = SMOTE()
#X_train, y_train = smt.fit_resample(X_train, y_train)
X_train_sm, y_train_sm = smt.fit_resample(X_train, y_train)

counter = Counter(y_train_sm)
print('After',counter)

"""# ADASYN"""

from collections import Counter
from imblearn.over_sampling import ADASYN

counter = Counter(y_train)
print('Before',counter)
# oversampling the train dataset using ADASYN
ada = ADASYN(sampling_strategy='minority', random_state=130)
X_train_ada, y_train_ada = ada.fit_resample(X_train, y_train)

counter = Counter(y_train_ada)
print('After',counter)

"""# SMOTE + Tomek Links"""

from imblearn.combine import SMOTETomek
from collections import Counter

counter = Counter(y_train)
print('Before',counter)
# oversampling the train dataset using SMOTE + Tomek
smtom = SMOTETomek(random_state=139)
X_train_smtom, y_train_smtom = smtom.fit_resample(X_train, y_train)

counter = Counter(y_train_smtom)
print('After',counter)

"""# SMOTE + ENN"""

from imblearn.combine import SMOTEENN
from collections import Counter
counter = Counter(y_train)
print('Before',counter)
# oversampling the train dataset using SMOTE + ENN
smenn = SMOTEENN()
X_train_smenn, y_train_smenn = smenn.fit_resample(X_train, y_train)

counter = Counter(y_train_smenn)
print('After',counter)

"""# Naive Bayes Classifier"""

print("\nNaive Bayes Results:\n")
# create Gaussian Naive Bayes model object and train it with the data
nb_model = GaussianNB()
nb_model.fit(X_train, y_train.ravel())

# Predict on test set using fitted model
preds= nb_model.predict(X_test)
preds_prob = nb_model.predict_proba(X_test)

# Mean accuracy

accuracy_score(y_test,preds)

# from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# target_names=['Normal', 'Insider']
target_names = ["0","1","2","3"]
print(classification_report(preds, y_test,target_names=target_names))

# classes = ["Non-Malacious", "Malacious"]
classes = ["0","1","2","3"]
#print confusion matrix
cm = confusion_matrix(y_test,preds)

print('Confusion Matrix')
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=classes)
fig, ax = plt.subplots(figsize=(5,5))
plt.title("Confusion Matrix - Naive Bayes")
disp = disp.plot(ax=ax)
plt.savefig('Confusion Matrix - Naive Bayes.png')
plt.show()

Accuracy = metrics.accuracy_score(y_test, preds)
print('Accuracy:', Accuracy)

Precision = metrics.precision_score(y_test, preds, average=None)
print('Precision:', Precision)

Recall = metrics.recall_score(y_test, preds, average=None)
print('Recall:', Recall)

F1_score = metrics.f1_score(y_test, preds, average=None)
print('F1 Score:', F1_score)

False_positive_rate = metrics.confusion_matrix(y_test, preds)[0, 1] / (metrics.confusion_matrix(y_test, preds)[0, 1] + metrics.confusion_matrix(y_test, preds)[0, 0])
print('False Positive Rate:', False_positive_rate)

False_negative_rate = metrics.confusion_matrix(y_test, preds)[1, 0] / (metrics.confusion_matrix(y_test, preds)[1, 0] + metrics.confusion_matrix(y_test, preds)[1, 1])
print('False Negative Rate:', False_negative_rate)

True_positive_rate = metrics.confusion_matrix(y_test, preds)[1, 1] / (metrics.confusion_matrix(y_test, preds)[1, 1] + metrics.confusion_matrix(y_test, preds)[1, 0])
print('True Positive Rate:', True_positive_rate)

True_negative_rate = metrics.confusion_matrix(y_test, preds)[0, 0] /  (metrics.confusion_matrix(y_test, preds)[0, 0] + metrics.confusion_matrix(y_test, preds)[0, 1])
print('True Negative Rate:', True_negative_rate)

Balanced_accuracy = metrics.balanced_accuracy_score(y_test, preds)
print('Balanced Accuracy:', Balanced_accuracy)

TP = metrics.confusion_matrix(y_test, preds)[1, 1]
print('True Positive:', TP)

FP = metrics.confusion_matrix(y_test, preds)[0, 1]
print('False Positive:', FP)

FN = metrics.confusion_matrix(y_test, preds)[1, 0]
print('False Negative:', FN)

TN = metrics.confusion_matrix(y_test, preds)[0, 0]
print('True Negative:', TN)

Accuracy = (TP + TN) / (TP + TN + FP + FN)
print('Accuracy:', Accuracy)

Precision = TP / (TP + FP)
print('Precision:', Precision)

Recall = TP / (TP + FN)
print('Recall:', Recall)

F1_score = 2 * Precision * Recall / (Precision + Recall)
print('F1 Score:', F1_score)

True_negative_rate = TN / (TN + FN)
print('True Negative Rate:', True_negative_rate)

False_negative_rate = FN / (FN + TP)
print('False Negative Rate:', False_negative_rate)

True_positive_rate = TP / (TP + FN)
print('True Positive Rate:', True_positive_rate)

False_positive_rate = FP / (FP + TN)
print('False Positive Rate:', False_positive_rate)

roc_auc = roc_auc_score(y_test,preds_prob,multi_class='ovr')
print('ROC AUC score: %.3f' % roc_auc)
print('Mean ROC AUC: %.5f' % roc_auc.mean())


"""# Random Forest Classifier"""
print("\n Random Forest Results:\n")

# Create a RandomForestClassifier object
rf_model = RandomForestClassifier(random_state=42)

rf_model.fit(X_train, y_train.ravel())

# Predict on test set using fitted model
preds = rf_model.predict(X_test)
preds_prob = rf_model.predict_proba(X_test)

# Mean accuracy
accuracy_score(y_test,preds)

target_names=['0','1','2','3']
print(classification_report(preds, y_test,target_names=target_names))

classes = ['0','1','2','3']
#print confusion matrix
cm = confusion_matrix(y_test,preds)

print('Confusion Matrix')
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=classes)
fig, ax = plt.subplots(figsize=(5,5))
plt.title("Confusion Matrix - Random Forest")
disp = disp.plot(ax=ax)
plt.savefig('Confusion Matrix - Random Forest.png')
plt.show()

Accuracy = metrics.accuracy_score(y_test, preds)
print('Accuracy:', Accuracy)

Precision = metrics.precision_score(y_test, preds, average=None)
print('Precision:', Precision)

Recall = metrics.recall_score(y_test, preds, average=None)
print('Recall:', Recall)

F1_score = metrics.f1_score(y_test, preds, average=None)
print('F1 Score:', F1_score)

False_positive_rate = metrics.confusion_matrix(y_test, preds)[0, 1] / (metrics.confusion_matrix(y_test, preds)[0, 1] + metrics.confusion_matrix(y_test, preds)[0, 0])
print('False Positive Rate:', False_positive_rate)

False_negative_rate = metrics.confusion_matrix(y_test, preds)[1, 0] / (metrics.confusion_matrix(y_test, preds)[1, 0] + metrics.confusion_matrix(y_test, preds)[1, 1])
print('False Negative Rate:', False_negative_rate)

True_positive_rate = metrics.confusion_matrix(y_test, preds)[1, 1] / (metrics.confusion_matrix(y_test, preds)[1, 1] + metrics.confusion_matrix(y_test, preds)[1, 0])
print('True Positive Rate:', True_positive_rate)

True_negative_rate = metrics.confusion_matrix(y_test, preds)[0, 0] /  (metrics.confusion_matrix(y_test, preds)[0, 0] + metrics.confusion_matrix(y_test, preds)[0, 1])
print('True Negative Rate:', True_negative_rate)

Balanced_accuracy = metrics.balanced_accuracy_score(y_test, preds)
print('Balanced Accuracy:', Balanced_accuracy)

TP = metrics.confusion_matrix(y_test, preds)[1, 1]
print('True Positive:', TP)

FP = metrics.confusion_matrix(y_test, preds)[0, 1]
print('False Positive:', FP)

FN = metrics.confusion_matrix(y_test, preds)[1, 0]
print('False Negative:', FN)

TN = metrics.confusion_matrix(y_test, preds)[0, 0]
print('True Negative:', TN)

Accuracy = (TP + TN) / (TP + TN + FP + FN)
print('Accuracy:', Accuracy)

Precision = TP / (TP + FP)
print('Precision:', Precision)

Recall = TP / (TP + FN)
print('Recall:', Recall)

F1_score = 2 * Precision * Recall / (Precision + Recall)
print('F1 Score:', F1_score)

True_negative_rate = TN / (TN + FN)
print('True Negative Rate:', True_negative_rate)

False_negative_rate = FN / (FN + TP)
print('False Negative Rate:', False_negative_rate)

True_positive_rate = TP / (TP + FN)
print('True Positive Rate:', True_positive_rate)

False_positive_rate = FP / (FP + TN)
print('False Positive Rate:', False_positive_rate)

roc_auc = roc_auc_score(y_test,preds_prob,multi_class='ovr')
print('ROC AUC score: %.3f' % roc_auc)
print('Mean ROC AUC: %.5f' % roc_auc.mean())


"""# Decision Tree"""
print("\nDecision Tree Results:\n")

# create Gaussian Naive Bayes model object and train it with the data
dt_model = DecisionTreeClassifier(random_state=42)

dt_model.fit(X_train, y_train.ravel())

# Predict on test set using fitted model
preds = dt_model.predict(X_test)
preds_prob = dt_model.predict_proba(X_test)

# Mean accuracy
accuracy_score(y_test, preds)

target_names=['0', '1', '2', '3']
print(classification_report(preds, y_test,target_names=target_names))

classes = ['0','1','2','3']
#print confusion matrix
cm = confusion_matrix(y_test,preds)

print('Confusion Matrix')
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=classes)
fig, ax = plt.subplots(figsize=(5,5))
plt.title("Confusion Matrix - Decision Tree")
disp = disp.plot(ax=ax)
plt.savefig('Confusion Matrix - Decision Tree.png')
plt.show()

Accuracy = metrics.accuracy_score(y_test, preds)
print('Accuracy:', Accuracy)

Precision = metrics.precision_score(y_test, preds, average=None)
print('Precision:', Precision)

Recall = metrics.recall_score(y_test, preds, average=None)
print('Recall:', Recall)

F1_score = metrics.f1_score(y_test, preds, average=None)
print('F1 Score:', F1_score)

False_positive_rate = metrics.confusion_matrix(y_test, preds)[0, 1] / (metrics.confusion_matrix(y_test, preds)[0, 1] + metrics.confusion_matrix(y_test, preds)[0, 0])
print('False Positive Rate:', False_positive_rate)

False_negative_rate = metrics.confusion_matrix(y_test, preds)[1, 0] / (metrics.confusion_matrix(y_test, preds)[1, 0] + metrics.confusion_matrix(y_test, preds)[1, 1])
print('False Negative Rate:', False_negative_rate)

True_positive_rate = metrics.confusion_matrix(y_test, preds)[1, 1] / (metrics.confusion_matrix(y_test, preds)[1, 1] + metrics.confusion_matrix(y_test, preds)[1, 0])
print('True Positive Rate:', True_positive_rate)

True_negative_rate = metrics.confusion_matrix(y_test, preds)[0, 0] /  (metrics.confusion_matrix(y_test, preds)[0, 0] + metrics.confusion_matrix(y_test, preds)[0, 1])
print('True Negative Rate:', True_negative_rate)

Balanced_accuracy = metrics.balanced_accuracy_score(y_test, preds)
print('Balanced Accuracy:', Balanced_accuracy)

TP = metrics.confusion_matrix(y_test, preds)[1, 1]
print('True Positive:', TP)

FP = metrics.confusion_matrix(y_test, preds)[0, 1]
print('False Positive:', FP)

FN = metrics.confusion_matrix(y_test, preds)[1, 0]
print('False Negative:', FN)

TN = metrics.confusion_matrix(y_test, preds)[0, 0]
print('True Negative:', TN)

Accuracy = (TP + TN) / (TP + TN + FP + FN)
print('Accuracy:', Accuracy)

Precision = TP / (TP + FP)
print('Precision:', Precision)

Recall = TP / (TP + FN)
print('Recall:', Recall)

F1_score = 2 * Precision * Recall / (Precision + Recall)
print('F1 Score:', F1_score)

True_negative_rate = TN / (TN + FN)
print('True Negative Rate:', True_negative_rate)

False_negative_rate = FN / (FN + TP)
print('False Negative Rate:', False_negative_rate)

True_positive_rate = TP / (TP + FN)
print('True Positive Rate:', True_positive_rate)

False_positive_rate = FP / (FP + TN)
print('False Positive Rate:', False_positive_rate)

roc_auc = roc_auc_score(y_test,preds_prob,multi_class='ovr')
print('ROC AUC score: %.3f' % roc_auc)
print('Mean ROC AUC: %.5f' % roc_auc.mean())

"""# Support Vector Machine"""
print("\nSVM Results:\n")


svm_model = SVC(probability=True)
svm_model.fit(X_train, y_train.ravel())

preds = svm_model.predict(X_test)
preds_prob = svm_model.predict_proba(X_test)


accuracy_score(y_test, preds)

target_names=['0','1','2','3']
print(classification_report(preds, y_test,target_names=target_names))

classes = ['0','1','2','3']
cm = confusion_matrix(y_test,preds)
print('Confusion Matrix')
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=classes)
fig, ax = plt.subplots(figsize=(5,5))
plt.title("Confusion Matrix - SVM")
disp = disp.plot(ax=ax)
plt.savefig('Confusion Matrix - SVM .png')
plt.show()

Accuracy = metrics.accuracy_score(y_test, preds)
print('Accuracy:', Accuracy)

Precision = metrics.precision_score(y_test, preds, average=None)
print('Precision:', Precision)

Recall = metrics.recall_score(y_test, preds, average=None)
print('Recall:', Recall)

F1_score = metrics.f1_score(y_test, preds, average=None)
print('F1 Score:', F1_score)

False_positive_rate = metrics.confusion_matrix(y_test, preds)[0, 1] / (metrics.confusion_matrix(y_test, preds)[0, 1] + metrics.confusion_matrix(y_test, preds)[0, 0])
print('False Positive Rate:', False_positive_rate)

False_negative_rate = metrics.confusion_matrix(y_test, preds)[1, 0] / (metrics.confusion_matrix(y_test, preds)[1, 0] + metrics.confusion_matrix(y_test, preds)[1, 1])
print('False Negative Rate:', False_negative_rate)

True_positive_rate = metrics.confusion_matrix(y_test, preds)[1, 1] / (metrics.confusion_matrix(y_test, preds)[1, 1] + metrics.confusion_matrix(y_test, preds)[1, 0])
print('True Positive Rate:', True_positive_rate)

True_negative_rate = metrics.confusion_matrix(y_test, preds)[0, 0] /  (metrics.confusion_matrix(y_test, preds)[0, 0] + metrics.confusion_matrix(y_test, preds)[0, 1])
print('True Negative Rate:', True_negative_rate)

Balanced_accuracy = metrics.balanced_accuracy_score(y_test, preds)
print('Balanced Accuracy:', Balanced_accuracy)

TP = metrics.confusion_matrix(y_test, preds)[1, 1]
print('True Positive:', TP)

FP = metrics.confusion_matrix(y_test, preds)[0, 1]
print('False Positive:', FP)

FN = metrics.confusion_matrix(y_test, preds)[1, 0]
print('False Negative:', FN)

TN = metrics.confusion_matrix(y_test, preds)[0, 0]
print('True Negative:', TN)

Accuracy = (TP + TN) / (TP + TN + FP + FN)
print('Accuracy:', Accuracy)

Precision = TP / (TP + FP)
print('Precision:', Precision)

Recall = TP / (TP + FN)
print('Recall:', Recall)

F1_score = 2 * Precision * Recall / (Precision + Recall)
print('F1 Score:', F1_score)

True_negative_rate = TN / (TN + FN)
print('True Negative Rate:', True_negative_rate)

False_negative_rate = FN / (FN + TP)
print('False Negative Rate:', False_negative_rate)

True_positive_rate = TP / (TP + FN)
print('True Positive Rate:', True_positive_rate)

False_positive_rate = FP / (FP + TN)
print('False Positive Rate:', False_positive_rate)

# Calculate ROC AUC scores:

# 1. Macro-average strategy (treats all classes equally):
roc_auc_macro = roc_auc_score(y_test, preds_prob, multi_class='ovr', average='macro')
print('Macro-average ROC AUC score: %.3f' % roc_auc_macro)

# 2. Weighted-average strategy (weights scores by class proportions):
roc_auc_weighted = roc_auc_score(y_test, preds_prob, multi_class='ovr', average='weighted')
print('Weighted-average ROC AUC score: %.3f' % roc_auc_weighted)




"""XG Boost Classifier

"""
print("\nXG Boost Results:\n")

xgb_model = xgb.XGBClassifier(random_state=42)
xgb_model.fit(X_train, y_train.ravel())

preds = xgb_model.predict(X_test)

accuracy_score(y_test, preds)

target_names=['0','1','2','3']
print(classification_report(preds, y_test,target_names=target_names))

# Print confusion matrix
classes = ['0','1','2','3']
cm = confusion_matrix(y_test, preds)
print('Confusion Matrix')
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=classes)
fig, ax = plt.subplots(figsize=(5,5))
plt.title("Confusion Matrix - XGB")
disp = disp.plot(ax=ax)
plt.savefig('Confusion Matrix - XGB.png')
plt.show()
# Print accuracy metrics
Accuracy = metrics.accuracy_score(y_test, preds)
print('Accuracy:', Accuracy)

Precision = metrics.precision_score(y_test, preds, average=None)
print('Precision:', Precision)

Recall = metrics.recall_score(y_test, preds, average=None)
print('Recall:', Recall)

F1_score = metrics.f1_score(y_test, preds, average=None)
print('F1 Score:', F1_score)

# Print misclassification rates
False_positive_rate = metrics.confusion_matrix(y_test, preds)[0, 1] / (metrics.confusion_matrix(y_test, preds)[0, 1] + metrics.confusion_matrix(y_test, preds)[0, 0])
print('False Positive Rate:', False_positive_rate)

False_negative_rate = metrics.confusion_matrix(y_test, preds)[1, 0] / (metrics.confusion_matrix(y_test, preds)[1, 0] + metrics.confusion_matrix(y_test, preds)[1, 1])
print('False Negative Rate:', False_negative_rate)

True_positive_rate = metrics.confusion_matrix(y_test, preds)[1, 1] / (metrics.confusion_matrix(y_test, preds)[1, 1] + metrics.confusion_matrix(y_test, preds)[1, 0])
print('True Positive Rate:', True_positive_rate)

True_negative_rate = metrics.confusion_matrix(y_test, preds)[0, 0] /  (metrics.confusion_matrix(y_test, preds)[0, 0] + metrics.confusion_matrix(y_test, preds)[0, 1])
print('True Negative Rate:', True_negative_rate)

# Print balanced accuracy
Balanced_accuracy = metrics.balanced_accuracy_score(y_test, preds)
print('Balanced Accuracy:', Balanced_accuracy)

# Print individual counts
TP = metrics.confusion_matrix(y_test, preds)[1, 1]
print('True Positive:', TP)

FP = metrics.confusion_matrix(y_test, preds)[0, 1]
print('False Positive:', FP)

FN = metrics.confusion_matrix(y_test, preds)[1, 0]
print('False Negative:', FN)

TN = metrics.confusion_matrix(y_test, preds)[0, 0]
print('True Negative:', TN)

# Print accuracy, precision, recall, and F1 score based on individual counts
Accuracy = (TP + TN) / (TP + TN + FP + FN)
print('Accuracy:', Accuracy)

Precision = TP / (TP + FP)
print('Precision:', Precision)

Recall = TP / (TP + FN)
print('Recall:', Recall)

F1_score = 2 * Precision * Recall / (Precision + Recall)
print('F1 Score:', F1_score)

# Print true negative rate, false negative rate, true positive rate, and false positive rate based on individual counts
True_negative_rate = TN / (TN + FN)
print('True Negative Rate:', True_negative_rate)

False_negative_rate = FN / (FN + TP)
print('False Negative Rate:', False_negative_rate)

True_positive_rate = TP / (TP + FN)
print('True Positive Rate:', True_positive_rate)

False_positive_rate = FP / (FP + TN)
print('False Positive Rate:', False_positive_rate)

# Option 1: One-vs-rest with macro-average
roc_auc_ovr_macro = roc_auc_score(y_test, preds_prob, multi_class='ovr', average='macro')
print('ROC AUC score (OvR, macro): %.3f' % roc_auc_ovr_macro)

# Option 2: One-vs-one with weighted average
roc_auc_ovo_weighted = roc_auc_score(y_test, preds_prob, multi_class='ovo', average='weighted')
print('ROC AUC score (OvO, weighted): %.3f' % roc_auc_ovo_weighted)


# # Calculate and print ROC AUC score
# roc_auc = roc_auc_score(y_test, preds_prob)
# print('ROC AUC score: %.3f' % roc_auc)

# # Calculate and print mean ROC AUC score
# print('Mean ROC AUC: %.5f' % roc_auc.mean())


# prompt: give me code for knn algorithm like above one xgboost
""" KNN Algorithm

"""
print("\nKNN Results:\n")
# Create a KNeighborsClassifier object
knn_model = KNeighborsClassifier(n_neighbors=3)

# Train the model on the training data
knn_model.fit(X_train, y_train.ravel())

# Predict on test set using fitted model
preds = knn_model.predict(X_test)

# Mean accuracy
accuracy_score(y_test, preds)

# Create a classification report
target_names=['0','1','2','3']
print(classification_report(preds, y_test,target_names=target_names))

# Print confusion matrix
classes = ['0','1','2','3']
cm = confusion_matrix(y_test, preds)
print('Confusion Matrix')
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=classes)
fig, ax = plt.subplots(figsize=(5,5))
plt.title("Confusion Matrix - KNN")
disp = disp.plot(ax=ax)
plt.savefig('Confusion Matrix - KNN.png')
plt.show()
# Print accuracy metrics
Accuracy = metrics.accuracy_score(y_test, preds)
print('Accuracy:', Accuracy)

Precision = metrics.precision_score(y_test, preds, average=None)
print('Precision:', Precision)

Recall = metrics.recall_score(y_test, preds, average=None)
print('Recall:', Recall)

F1_score = metrics.f1_score(y_test, preds, average=None)
print('F1 Score:', F1_score)

# Print misclassification rates
False_positive_rate = metrics.confusion_matrix(y_test, preds)[0, 1] / (metrics.confusion_matrix(y_test, preds)[0, 1] + metrics.confusion_matrix(y_test, preds)[0, 0])
print('False Positive Rate:', False_positive_rate)

False_negative_rate = metrics.confusion_matrix(y_test, preds)[1, 0] / (metrics.confusion_matrix(y_test, preds)[1, 0] + metrics.confusion_matrix(y_test, preds)[1, 1])
print('False Negative Rate:', False_negative_rate)

True_positive_rate = metrics.confusion_matrix(y_test, preds)[1, 1] / (metrics.confusion_matrix(y_test, preds)[1, 1] + metrics.confusion_matrix(y_test, preds)[1, 0])
print('True Positive Rate:', True_positive_rate)

True_negative_rate = metrics.confusion_matrix(y_test, preds)[0, 0] /  (metrics.confusion_matrix(y_test, preds)[0, 0] + metrics.confusion_matrix(y_test, preds)[0, 1])
print('True Negative Rate:', True_negative_rate)

# Print balanced accuracy
Balanced_accuracy = metrics.balanced_accuracy_score(y_test, preds)
print('Balanced Accuracy:', Balanced_accuracy)

# Print individual counts
TP = metrics.confusion_matrix(y_test, preds)[1, 1]
print('True Positive:', TP)

FP = metrics.confusion_matrix(y_test, preds)[0, 1]
print('False Positive:', FP)

FN = metrics.confusion_matrix(y_test, preds)[1, 0]
print('False Negative:', FN)

TN = metrics.confusion_matrix(y_test, preds)[0, 0]
print('True Negative:', TN)

# Print accuracy, precision, recall, and F1 score based on individual counts
Accuracy = (TP + TN) / (TP + TN + FP + FN)
print('Accuracy:', Accuracy)

Precision = TP / (TP + FP)
print('Precision:', Precision)

Recall = TP / (TP + FN)
print('Recall:', Recall)

F1_score = 2 * Precision * Recall / (Precision + Recall)
print('F1 Score:', F1_score)

# Print true negative rate, false negative rate, true positive rate, and false positive rate based on individual counts
True_negative_rate = TN / (TN + FN)
print('True Negative Rate:', True_negative_rate)

False_negative_rate = FN / (FN + TP)
print('False Negative Rate:', False_negative_rate)

True_positive_rate = TP / (TP + FN)
print('True Positive Rate:', True_positive_rate)

False_positive_rate = FP / (FP + TN)
print('False Positive Rate:', False_positive_rate)


roc_auc_ovr_macro = roc_auc_score(y_test, preds_prob, multi_class='ovr', average='macro')
print('ROC AUC score (OvR, macro): %.3f' % roc_auc_ovr_macro)

roc_auc_ovo_weighted = roc_auc_score(y_test, preds_prob, multi_class='ovo', average='weighted')
print('ROC AUC score (OvO, weighted): %.3f' % roc_auc_ovo_weighted)