import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, svm, neighbors, linear_model, discriminant_analysis, naive_bayes, tree
from utils import general


df = pd.read_csv('iris.data.txt', names=['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class'])
df = df.replace({'class': {'Iris-setosa': 3, 'Iris-versicolor': 5, 'Iris-virginica': 7}})

X = np.array(df.drop(['class'], 1))
X = preprocessing.scale(X)
y = np.array(df['class'])

gen = general.General()
gen.fit_score_print(X, y)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# prepare models
models = []
models.append(('LR', linear_model.LogisticRegression()))
models.append(('LDA', discriminant_analysis.LinearDiscriminantAnalysis()))
models.append(('KNN', neighbors.KNeighborsClassifier()))
models.append(('CART', tree.DecisionTreeClassifier()))
models.append(('NB', naive_bayes.GaussianNB()))
models.append(('SVM', svm.SVC()))

for name, model in models:
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    print(name, accuracy)