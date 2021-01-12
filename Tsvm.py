import pandas
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

direktori = "heart.csv"
#var =['AGE','EDUCATION','CP','TRESTBPS','CHOL','FBS','RESTECG','THALAC','EXANG',
 #     'OLDPEAK','SLOPE','CA','THAL']

data=pandas.read_csv(direktori)

import numpy as np

array = data.values

imp = SimpleImputer(missing_values=np.nan, strategy="mean")

data_clean = imp.fit_transform(data)

x = data_clean[:, 0:12]
Y = data_clean[:,13]


from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5)

from sklearn.svm import SVC
from sklearn import metrics

array = []
array1 = []
array2 = []
array3 = []
for i in range(5):
    accuracy= 0
    precision = 0
    recall = 0
    f1 = 0
    for train, test in skf.split(x,Y):
        X_train, X_test, y_train, y_test = x[train], x[test], Y[train], Y[test]
        #model = SVC(kernel='linear')
        #model = SVC(kernel='poly')
        model = SVC()
        #model = SVC(kernel='sigmoid')
        model.fit(X_train, y_train)
    
        #print(model)
        expected = y_test
        predicted = model.predict(X_test)
    
        #print(metrics.classification_report(expected, predicted))
        #print(metrics.confusion_matrix(expected, predicted))
        accuracy = accuracy + metrics.accuracy_score(expected, predicted)
        precision = precision + metrics.precision_score(expected, predicted)
        recall = recall + metrics.recall_score(expected, predicted)
        f1 = f1 + metrics.f1_score(expected, predicted)
        #print(metrics.accuracy_score(expected, predicted))
        #print(metrics.precision_score(expected, predicted))
        #print(metrics.recall_score(expected, predicted))
        #print(metrics.f1_score(expected, predicted))
    array.append(accuracy/5)
    array1.append(precision/5)
    array2.append(recall/5)
    array3.append(f1/5)
print("rata rata akurasi : " ,array)
print("rata rata presisi : " ,array1)
print("rata rata recall : " ,array2)
print("rata rata f1 : " ,array3)


plt.plot(array, marker = "o", linewidth=1, label= "accuracy")
plt.plot(array1, marker = "o", linewidth=1, label= "precision")
plt.plot(array2, marker = "o", linewidth=1, label= "recall")
plt.plot(array3, marker = "o", linewidth=1, label= "f1")
plt.legend()
plt.show()
