# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 20:27:04 2021

@author: Sokkhey
"""
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
#----------------------------------------------------------------------
data = pd.read_csv('datasets/CreditScoringRevised_scale.csv', low_memory=False)
#-----------------------------------------------------------------------
data.loc[data["advance1t5"]=='10%', "advance1t5"] = 0
data.loc[data["advance1t5"]=='20%', "advance1t5"] = 1
data.loc[data["advance1t5"]=='30%', "advance1t5"] = 2
data.loc[data["advance1t5"]=='40%', "advance1t5"] = 3
data.loc[data["advance1t5"]=='>50%', "advance1t5"] = 4
data = data.drop('age_group', axis=1)
# data1 = pd.read_csv('CreditScoring.csv', low_memory=False)
print(data.shape)
print(data.head())
print(data.info())
print(data.describe())
print(data.values)
print(data.index)
print(data.columns)
# WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWwwwww -- Check missing value
print(data.isna().sum())
data.isna().sum().plot(kind="bar")
plt.show()
"""From the plot we see that there are missing values, since our data is not small,
so it is fine to drop the missing value. However, the columns children is big, 
so use imputation"""
print(data.children)
data["children"] = data["children"].fillna(data["children"].mean())
print(data.isna().sum())             # -- check again
data.isna().sum().plot(kind="bar")
plt.show()
#------------------------------- Drop NaN and Normalize data
from sklearn import preprocessing
data = data.dropna()
X = data.drop('default', axis=1)
# --

col = ['ratio', 'ratio_group', 'start', 'y2017',  'y2013', 'score1', 'term','term_st', 'term_group','debt', 'score0', 
       'score_coef', 'p1', 'p0', 'advance', 'advance1t5','temployer', 'sex', 'locaSit', 'age',  'y2015', 
       'maritalstatus', 'templ_g1','taddress','children','taddress_g1']
X = X[col]
X = pd.get_dummies(X)
y = data['default']

min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)
# WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW -- Data Balancing 
import imblearn         # -- SMOTE
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_style("whitegrid")
print(imblearn.__version__)
print(y.value_counts()) #normalize=True

unique, counts = np.unique(y, return_counts=True)
uniques = np.array(["Good Borrower", "Bad Borrrower"]).astype('str')
plt.bar(uniques, counts/counts.sum())
plt.xlabel('Classes')
plt.ylabel('Frequency')
plt.title("Defualt")
plt.show()
# WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW -- with PCA
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
X_pca = pd.DataFrame(PCA(n_components=2).fit_transform(X))
plt.scatter(X_pca.iloc[:,0], X_pca.iloc[:,1])
# WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWw -- Data Split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21)

# WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW -- Decision Tree: CART
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# -- Split data in to 70%-30%: train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1)
# -- Instantiate dt
dt = DecisionTreeClassifier()   #(max_depth=2, random_state=1)
# -- fit model to the trining set
dt.fit(X_train, y_train)
# -- predict test set label
y_pred = dt.predict(X_test)
print(confusion_matrix(y_test, y_pred))
# -- evaluate accuracy
dt_score = dt.score(X_test, y_test)
print('The accuracy of the DT classifier: {:.4f}'.format(dt_score))
dt_report = classification_report(y_test, y_pred)
print("The classificaiton report:{}".format(dt_report))
# ------- DT with entroy
# Instantiate dt_entropy, set 'entropy' as the information criterion
dt_entropy = DecisionTreeClassifier(criterion='entropy', random_state=1) # max_depth=8,
# Fit dt_entropy to the training set
dt_entropy.fit(X_train, y_train)
y_pred = dt_entropy.predict(X_test)
print(confusion_matrix(y_test, y_pred))
# -- evaluate accuracy
dt_score = dt.score(X_test, y_test)
print('The accuracy of the DT classifier: {:.4f}'.format(dt_score))
dt_entropy_report = classification_report(y_test, y_pred)
print('The accuracy of the DT_entropy classifier: {}'.format(dt_entropy_report))


def evaluation_baselinemodel(model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    model_score = model.score(X_test, y_test)
    print('The accuracy of the classifier: {:.4f}'.format(model_score))
    model_report = classification_report(y_test, y_pred)
    print("The classificaiton report:{}".format(model_report))
    auc = roc_auc_score(y_test, y_pred)
    print('AUC_score: %.4f' % auc)
# -------------
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
# -- LR
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
print("The evaluation metrics of the LOGREG:")
evaluation_baselinemodel(logreg)
# -- SVM
from sklearn import svm
svmc = svm.SVC(C=10, gamma=0.01)
print("The evaluation metrics of the SVM:")
evaluation_baselinemodel(svmc)
# --- MLP
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier()
print("The evaluation metrics of the MLP:")
evaluation_baselinemodel(mlp)
# --------- Naive Bayes
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
print("The evaluation metrics of the Naive Bayes:")
evaluation_baselinemodel(nb)
# -- LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
print("The evaluation metrics of the LDA:")
evaluation_baselinemodel(lda)
# --------------
from xgboost import XGBClassifier
xgbm = XGBClassifier()
print("The evaluation metrics of the XGBoost:")
evaluation_baselinemodel1(xgbm)