from sklearn import model_selection, svm, neighbors, linear_model, discriminant_analysis, naive_bayes, tree


class General:

    def __init__(self, params=None):
        self.params = params

    def fit_score_print(self, X, y, test_size=0.2):
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=test_size)

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
