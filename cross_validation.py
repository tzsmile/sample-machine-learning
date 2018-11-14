# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 14:24:13 2018

@author: tzsmile
"""

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
df=pd.read_csv('C:/Users/tzsmile/Documents/project/GAC/daily_stats.csv')
df.head()
df.describe()
df.sort_values(by=['day'])


df1=df[df['day']<="2018-05-22"]
df1.tail()
df1.shape
df.shape
labels=pd.read_csv('C:/Users/tzsmile/Documents/project/GAC/labels.csv')
df1.sort_values(by=['vin'])
labels.sort_values(by=['vin'])
result = pd.merge(df1, labels, how='outer',on=['vin'])
y=result.loc[:,'label']
x=result.iloc[:,0:63]

#deal with NA value
from sklearn.preprocessing import Imputer
imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
imr = imr.fit(x)
x1 = imr.transform(x.values)

    
from sklearn.preprocessing import StandardScaler
stdsc=StandardScaler()
x_std=stdsc.fit_transform(x1)



from sklearn.decomposition import PCA
pca = PCA(n_components=20).fit(x_std)
x_std_pca = pca.fit_transform(x_std)



x_std_pca.shape

print(pca.explained_variance_ratio_)
print (pca.explained_variance_ratio_.cumsum())

from matplotlib import pyplot as plt
plt.semilogy(pca.explained_variance_ratio_, '--o')
plt.semilogy(pca.explained_variance_ratio_.cumsum(), '--o')

pca = PCA(n_components=16).fit(x_std_train)
x_std_pca = pca.fit_transform(x_std)



 from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
#from sklearn import svm
#from sklearn.linear_model import LogisticRegression

# on non-standardized data
k_fold=10
gnb = GaussianNB()
fit = gnb.fit(x_std_pca, y)
#y_pred_GNB = gnb.predict(x_test_std_pca)
#print('Acurracy GNB: %.2f' % accuracy_score(y_test,y_pred_GNB))
cv_gnb=cross_val_score(fit, x_std_pca, y, cv=k_fold, n_jobs=-1)
cv_gnb.mean()

        

from sklearn.linear_model import Perceptron
ppn = Perceptron(n_iter = 40, eta0 = 0.1, random_state = 0)
fit_ppn=ppn.fit(x_std_pca, y)
#y_pred = ppn.predict(x_test_std_pca)
#print('Acurracy Perceptron: %.2f' % accuracy_score(y_test,y_pred))
cv_ppn=cross_val_score(fit_ppn, x_std_pca, y, cv=k_fold, n_jobs=-1)
cv_ppn.mean()    

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1000.0, random_state = 0)
fit_lr=lr.fit(x_std_pca, y)
cv_lr=cross_val_score(fit_lr, x_std_pca, y, cv=k_fold, n_jobs=-1)
cv_lr.mean()   
#y_pred_logReg = lr.predict(x_test_std_pca)
 #Accuracy.append(['lr',input_list,accuracy_score(Y_test,Y_pred_logReg)]) 
#print('Acurracy Logistic Regression: %.2f' % accuracy_score(y_test,y_pred_logReg))

'''from sklearn.svm import SVC
svm=SVC(kernel = 'linear', C = 1.0 ,random_state = 0)
svm.fit(x_train_std_pca, y_train)
y_pred_svm = svm.predict(x_test_std_pca)
#Accuracy.append(['svm_linear',input_list,accuracy_score(y_test,y_pred_svm)]) 
print('Acurracy SVM(linear kernel): %.2f' % accuracy_score(y_test,y_pred_svm))
    
svm =SVC(kernel = 'rbf', C = 1.0 ,random_state = 0)
svm.fit(x_train_std_pca, y_train)
Y_pred_svm = svm.predict(x_test_std_pca)
#Accuracy.append(['svm_rbf',input_list,accuracy_score(Y_test,Y_pred_svm)]) 
print('Acurracy SVM(rbf kernel): %.2f' % accuracy_score(y_test,y_pred_svm))
 '''   
    
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion = 'entropy' ,max_depth = 3, random_state = 0)
fit_tree=tree.fit(x_std_pca,y)
cv_tree=cross_val_score(fit_tree, x_std_pca, y, cv=k_fold, n_jobs=-1)
cv_tree.mean() 
#y_pred_dt = tree.predict(x_test_std_pca)
#Accuracy.append(['tree1',input_list,accuracy_score(Y_test,Y_pred_svm)]) 
#print('Acurracy decision tree: %.2f' % accuracy_score(y_test,y_pred_dt))
    
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(criterion = 'entropy', n_estimators = 10, random_state = 1, n_jobs = 2)
fit_forest=forest.fit(x_std_pca, y)
#y_pred_rf = tree.predict(x_test_std_pca)
#Accuracy.append(['tree2',input_list,accuracy_score(Y_test,Y_pred_svm)]) 
#print('Acurracy random forest: %.2f' % accuracy_score(y_test,y_pred_rf))
cv_forest=cross_val_score(fit_forest, x_std_pca, y, cv=k_fold, n_jobs=-1)
cv_forest.mean() 

from sklearn.neighbors import  KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 100 ,p = 2, metric = 'minkowski')
fit_knn=knn.fit(x_std_pca, y)
# y_pred_knn = tree.predict(x_test_std_pca)
 #print('Acurracy knn: %.2f' % accuracy_score(y_test,y_pred_knn))
 #Accuracy.append(['knn',input_list,accuracy_score(Y_test,Y_pred_svm)])   
 cv_knn=cross_val_score(fit_knn, x_std_pca, y, cv=k_fold, n_jobs=-1)
 cv_knn.mean() 
    

 

 