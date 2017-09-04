import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
import seaborn as sns

###Add Data to Pandas DataFrame
df = pd.read_csv('Data/breast-cancer-wisconsin.data', header=None, names=['ID', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class'])
df = df.set_index('ID')
df['Class'].replace(2, 0, inplace=True)
df['Class'].replace(4, 1, inplace=True)
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)
print('--> Added Data into Pandas Data Frame')

#Split into training and test set
y = df['Class']
df.drop('Class', axis=1, inplace=True)
X = df
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print('--> Split Train Data to Trainign and Test Sets')

# Update Status
print('--------- Now Trying Support Vector Machine Classifier ---------')

#Make Support Vector Classifier Pipeline
pipe_svc = Pipeline([('scl', StandardScaler()),
                     ('pca', PCA(n_components=9)),
                     ('clf', SVC(random_state=1))])
print('--> Made Pipeline')

#Fit Pipeline to Data
pipe_svc.fit(X_train, y_train)
print('--> Fitted Pipeline to training Data')
scores = cross_val_score(estimator=pipe_svc,
                         X=X_train,
                         y=y_train,
                         cv=10,
                         n_jobs=1)
print('--> Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

#Tune Hyperparameters
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [{'clf__C': param_range,
               'clf__kernel': ['linear']},
              {'clf__C': param_range,
               'clf__gamma': param_range,
               'clf__kernel': ['rbf']}]
gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=10,
                  n_jobs=1)
gs = gs.fit(X_train, y_train)
print('--> Tuned Parameters Best Score: ',gs.best_score_)
print('--> Best Parameters: \n',gs.best_params_)

#Use best parameters
clf_svc = gs.best_estimator_

#Get Final Scores
clf_svc.fit(X_train, y_train)
scores = cross_val_score(estimator=clf_svc,
                         X=X_train,
                         y=y_train,
                         cv=10,
                         n_jobs=1)
print('--> Final Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

print('--> Final Accuracy on Test set: %.5f' % clf_svc.score(X_test,y_test))


#################### TRY LOGISTIC REGRESSION #########################

print('--------- Now Trying Linear Regression Classifier ---------')

from sklearn.linear_model import LogisticRegression

#Make Logistic Regression Pipeline
pipe_lr = Pipeline([('scl', StandardScaler()),
                    ('pca', PCA(n_components=9)),
                    ('clf', LogisticRegression(penalty='l2', tol=0.0001, C=1.0, random_state=1, max_iter=1000, n_jobs=-1))])
print('--> Made Logistic Regression Pipeline')

#Fit Pipeline to Data
pipe_lr.fit(X_train, y_train)
print('--> Fitted Pipeline to training Data')
scores = cross_val_score(estimator=pipe_lr,
                         X=X_train,
                         y=y_train,
                         cv=10,
                         n_jobs=1)
print('--> Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

#Tune Hyperparameters
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_range_small = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
param_grid_lr = [{'clf__penalty': ['l1'],
               'clf__C': param_range,
               'clf__tol': param_range_small},
              {'clf__penalty': ['l2'],
               'clf__C': param_range,
               'clf__tol': param_range_small}]
gs_lr = GridSearchCV(estimator=pipe_lr,
                  param_grid=param_grid_lr,
                  scoring='accuracy',
                  cv=10,
                  n_jobs=1)
gs_lr = gs_lr.fit(X_train, y_train)
print('--> Tuned Parameters Best Score: ',gs_lr.best_score_)
print('--> Best Parameters: \n',gs_lr.best_params_)

#Use best parameters
clf_lr = gs_lr.best_estimator_

#Get Final Scores
clf_lr.fit(X_train, y_train)
scores_lr = cross_val_score(estimator=clf_lr,
                         X=X_train,
                         y=y_train,
                         cv=10,
                         n_jobs=1)
print('--> Final Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores_lr), np.std(scores_lr)))

print('--> Final Accuracy on Test set: %.5f' % clf_lr.score(X_test,y_test))


#################### TRY RANDOM FOREST #########################

print('--------- Now Trying Random Forest ---------')

from sklearn.ensemble import RandomForestClassifier

#Make Logistic Regression Pipeline
pipe_rf = Pipeline([('scl', StandardScaler()),
                    ('pca', PCA(n_components=9)),
                    ('clf', RandomForestClassifier(n_estimators=10000, max_depth=None, bootstrap=True,n_jobs=-1, random_state=None))])
print('--> Made Random Forest Pipeline')

#Fit Pipeline to Data
pipe_rf.fit(X_train, y_train)
print('--> Fitted Pipeline to training Data')

#Get Final Scores
print('--> Final Accuracy on Test set: %.5f' % pipe_rf.score(X_test,y_test))


#################### Combine Models ##############

print('--------- Now Combining all Models ---------')

CorrectCount = 0
WrongCount = 0

for i in range(len(X_test)):
    pred = 0
    pred += clf_svc.predict(np.array(X_test.values[i].reshape(1,-1), dtype=np.float64))
    pred += clf_lr.predict(np.array(X_test.values[i].reshape(1,-1), dtype=np.float64))
    pred += pipe_rf.predict(np.array(X_test.values[i].reshape(1,-1), dtype=np.float64))
    guess = 0 if pred <= 1 else 1
    if guess == y_test.values[i]:
        #print('Correct')
        CorrectCount += 1 
    else:
        #print('Wrong')
        WrongCount += 1

print('Combined Accuracy: ', CorrectCount/136)
print('Correct Predictions: ', CorrectCount)
print('Incorrect Predictions: ', WrongCount)

########### FEATURE IMPORTANCE ##########################

from sklearn.ensemble import RandomForestClassifier

sns.set(style='whitegrid')

feat_labels = df.columns
forest = RandomForestClassifier(n_estimators = 10000, random_state=0, n_jobs=-1)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
indicies = np.argsort(importances)[::-1]
plt.title('Feature Importancces')
plt.bar(range(X_train.shape[1]),
        importances[indicies],
        color='blue',
        align='center')
plt.xticks(range(X_train.shape[1]),
           feat_labels[indicies], rotation = 90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()
