import numpy as नप
import pandas as पांडा
from sklearn import preprocessing, model_selection, svm, neighbors, linear_model, \
    discriminant_analysis, naive_bayes, tree

डेटा_ढांचा = पांडा.read_csv('iris.data.txt', names=['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class'])
डेटा_ढांचा = डेटा_ढांचा.replace({'class': {'Iris-setosa': 3, 'Iris-versicolor': 5, 'Iris-virginica': 7}})

एक्स = नप.array(डेटा_ढांचा.drop(['class'], 1))
एक्स = preprocessing.scale(एक्स)
वाई = नप.array(डेटा_ढांचा['class'])

एक्स_प्रशिक्षण, एक्स_परीक्षण, वाई_प्रशिक्षण, वाई_परीक्षण = model_selection.train_test_split(एक्स, वाई, test_size=0.2)

# prepare models
मॉडले = []
मॉडले.append(('रसद प्रतिगमन', linear_model.LogisticRegression()))
मॉडले.append(('रैखिक भेदभाव विश्लेषण', discriminant_analysis.LinearDiscriminantAnalysis()))
मॉडले.append(('क पड़ोस वर्गीकरण', neighbors.KNeighborsClassifier()))
मॉडले.append(('निर्णय वृक्ष वर्गीकरण', tree.DecisionTreeClassifier()))
मॉडले.append(('गॉसियन एनबी', naive_bayes.GaussianNB()))
मॉडले.append(('एसवीसी', svm.SVC()))

for name, मॉडल in मॉडले:
    मॉडल.fit(एक्स_प्रशिक्षण, वाई_प्रशिक्षण)

    शुद्धता = मॉडल.score(एक्स_परीक्षण, वाई_परीक्षण)

    print(name, शुद्धता)