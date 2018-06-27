import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, svm, neighbors, linear_model, discriminant_analysis, naive_bayes, \
    tree

df = pd.read_csv('forestfires.csv')
df = df.replace({'month': {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9,
                           'oct': 10, 'nov': 11, 'dec': 12},
                 'day': {'sun': 1, 'mon': 2, 'tue': 3, 'wed': 4, 'thu': 5, 'fri': 6, 'sat': 7}})

X = np.array(df.drop(['area', 'month', 'day', 'X', 'Y'], 1))
X = preprocessing.scale(X)
y = np.array(df['area'])
y = np.heaviside(y, 0)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# prepare models
models = []
models.append(('LR', linear_model.LogisticRegression()))
models.append(('LDA', discriminant_analysis.LinearDiscriminantAnalysis()))
models.append(('KNN', neighbors.KNeighborsClassifier()))
models.append(('CART', tree.DecisionTreeClassifier()))
models.append(('NB', naive_bayes.GaussianNB()))
models.append(('SVM', svm.SVC()))

clf = svm.SVC()
# clf = svm.LinearSVC()
# clf = neighbors.KNeighborsClassifier()
# clf = linear_model.LinearRegression()

for name, model in models:

    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    print(name, accuracy)
