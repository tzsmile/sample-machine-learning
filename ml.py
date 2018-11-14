# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 18:05:10 2018

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
#x_train, x_test = train_test_split(x1,test_size=0.30, random_state=12345)
x_train, x_test, y_train, y_test = \
    train_test_split(x1, y, test_size=0.3, 
                     stratify=y,
                     random_state=0)
    
from sklearn.preprocessing import StandardScaler
stdsc=StandardScaler()
x_std_train=stdsc.fit_transform(x_train)
x_std_test=stdsc.fit_transform(x_test)


from sklearn.decomposition import PCA
pca = PCA(n_components=20).fit(x_std_train)
x_train_std_pca = pca.fit_transform(x_std_train)
x_test_std_pca = pca.fit_transform(x_std_test)


x_train_std_pca.shape
x_test_std_pca.shape
print(pca.explained_variance_ratio_)
print (pca.explained_variance_ratio_.cumsum())

from matplotlib import pyplot as plt
plt.semilogy(pca.explained_variance_ratio_, '--o')
plt.semilogy(pca.explained_variance_ratio_.cumsum(), '--o')

pca = PCA(n_components=16).fit(x_std_train)
x_train_std_pca = pca.fit_transform(x_std_train)
x_test_std_pca = pca.fit_transform(x_std_test)



from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.linear_model import LogisticRegression

# on non-standardized data
gnb = GaussianNB()
fit = gnb.fit(x_train_std_pca, y_train)
y_pred_GNB = gnb.predict(x_test_std_pca)
print('Acurracy GNB: %.2f' % accuracy_score(y_test,y_pred_GNB))

from sklearn.linear_model import Perceptron
ppn = Perceptron(n_iter = 40, eta0 = 0.1, random_state = 0)
ppn.fit(x_train_std_pca, y_train)
y_pred = ppn.predict(x_test_std_pca)
print('Acurracy Perceptron: %.2f' % accuracy_score(y_test,y_pred))
    

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1000.0, random_state = 0)
lr.fit(x_train_std_pca, y_train)
y_pred_logReg = lr.predict(x_test_std_pca)
 #Accuracy.append(['lr',input_list,accuracy_score(Y_test,Y_pred_logReg)]) 
print('Acurracy Logistic Regression: %.2f' % accuracy_score(y_test,y_pred_logReg))

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
tree.fit(x_train_std_pca,y_train)
y_pred_dt = tree.predict(x_test_std_pca)
#Accuracy.append(['tree1',input_list,accuracy_score(Y_test,Y_pred_svm)]) 
print('Acurracy decision tree: %.2f' % accuracy_score(y_test,y_pred_dt))
    
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(criterion = 'entropy', n_estimators = 10, random_state = 1, n_jobs = 2)
forest.fit(x_train_std_pca, y_train)
y_pred_rf = tree.predict(x_test_std_pca)
#Accuracy.append(['tree2',input_list,accuracy_score(Y_test,Y_pred_svm)]) 
print('Acurracy random forest: %.2f' % accuracy_score(y_test,y_pred_rf))

    
 from sklearn.neighbors import  KNeighborsClassifier
 knn = KNeighborsClassifier(n_neighbors = 100 ,p = 2, metric = 'minkowski')
 knn.fit(x_train_std_pca, y_train)
 y_pred_knn = tree.predict(x_test_std_pca)
 print('Acurracy knn: %.2f' % accuracy_score(y_test,y_pred_knn))
 #Accuracy.append(['knn',input_list,accuracy_score(Y_test,Y_pred_svm)])   
 
    

 

 