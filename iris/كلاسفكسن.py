import numpy as نب
import pandas as بد
from sklearn import preprocessing, model_selection, svm, neighbors, linear_model, discriminant_analysis, naive_bayes, tree
from utils import general


دف = بد.read_csv('iris.data.txt', names=['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class'])
دف = دف.replace({'class': {'Iris-setosa': 3, 'Iris-versicolor': 5, 'Iris-virginica': 7}})

X = نب.array(دف.drop(['class'], 1))
X = preprocessing.scale(X)
y = نب.array(دف['class'])

gen = general.General()
gen.fit_score_print(X, y)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# prepare models
مادلس = []
مادلس.append(('LR', linear_model.LogisticRegression()))
مادلس.append(('LDA', discriminant_analysis.LinearDiscriminantAnalysis()))
مادلس.append(('KNN', neighbors.KNeighborsClassifier()))
مادلس.append(('CART', tree.DecisionTreeClassifier()))
مادلس.append(('NB', naive_bayes.GaussianNB()))
مادلس.append(('SVM', svm.SVC()))

for name, مادل in مادلس:
    مادل.fit(X_train, y_train)

    accuracy = مادل.score(X_test, y_test)

    print(name, accuracy)