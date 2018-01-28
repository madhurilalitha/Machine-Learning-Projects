EEG Signal Analysis - A Multi-Class Classification Problem

This project is to analyze the application of various machine learning algorithms on Health Montoring Signal Data. The Dataset is taken from UCI Respository can be found in this link (https://archive.ics.uci.edu/ml/datasets/arrhythmia) 

Ground Truth:

The aim is to distinguish between the presence and absence of cardiac arrhythmia and to classify it in one of the 16 groups. Class 01 refers to 'normal' ECG classes 02 to 15 refers to different classes of arrhythmia and class 16 refers to the rest of unclassified ones. For the time being, there exists a computer program that makes such a classification. However there are differences between the cardiolog's and the programs classification. Taking the cardiolog's as a gold standard we aim to minimise this difference by means of machine learning tools

Algorithms Applied and their results:

1) Support Vector Machine Classifier - Accuracy = 65.4%

2) Logistic Regression - Accuracy = 66.17%

3) RandomForest Classifier - 72.05%

4) Multi-Layer Perceptron = Accuracy = 63.9%

5) Gradient Boosting Classifier - Accuracy = 71.3%

#### I found that the Random Forest Classifier is showing better performance if the accuacy is taken as the performance metric
