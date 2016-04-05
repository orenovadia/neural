import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.cross_validation
import sklearn.cross_validation
import sklearn.datasets
import sklearn.svm
from sklearn.learning_curve import learning_curve
from sklearn.linear_model.logistic import LogisticRegressionCV
from sklearn.linear_model import Perceptron
from sklearn.metrics import mean_squared_error,f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
data = sklearn.datasets.load_iris()


def learn_c(data, estimator, train_sizes=np.linspace(0.5, 0.9, 5)):
    for train_size in train_sizes:
        X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(
            data.data, data.target,
            train_size=train_size)
        clf = estimator()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        fa_score = f1_score(y_test, y_pred)
        print '''\
        Train size  : %s
        MSE         :%s
        F score     : %s''' % (train_size, mse, fa_score)

        print '-' * 30

    train_sizes, train_scores, test_scores = learning_curve(
        estimator(), data.data, data.target, train_sizes=train_sizes, n_jobs=1
    )
    plt.figure()
    plt.title(str(estimator))
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.grid()
    plt.legend(loc='best')


for e in GaussianNB, LogisticRegressionCV,Perceptron,DecisionTreeClassifier,SVC:
    learn_c(data, e)
plt.show()
