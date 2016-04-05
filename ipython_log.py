# IPython log file

import sklearn
import sklearn.learning_curve
get_ipython().magic(u'logstart')
get_ipython().magic(u'ls ')
get_ipython().magic(u'ls -ltr')
import sklearn
import sklearn.learning_curve
import sklearn.datasets
data = sklearn.datasets.load_iris()
dir(data)
data.train
data['DATA']
data['train']
dir(data)
data.data
data.values
help(data)
help(sklearn.datasets.load_iris)
data.target
data.data
len(data.data)
len(data.target)
get_ipython().magic(u'cat ipython_log.py')
get_ipython().magic(u'startlog')
get_ipython().magic(u'logstart')
get_ipython().magic(u'logstop')
get_ipython().magic(u'logstart')
get_ipython().magic(u'ls ')
import sklearn
import sklearn.learning_curve
import sklearn.datasets
data = sklearn.datasets.load_iris()
from sklearn.naive_bayes import GaussianNB
X_train, X_test, y_train, y_test =  sklearn.cross_validation.train_test_split(
            data.data,data.target,)
help(GaussianNB)
clf = GaussianNB()
clf.fit(X_train,y_train)
clf
print clf
clf.predict([1,2,3,4])
_
__[0]
__[1]
clf.predict(X_test)
y_test
import sklearn.cross_validation
sklearn.cross_validation
get_ipython().magic(u'pinfo sklearn.cross_validation')
get_ipython().magic(u'pinfo sklearn.cross_validation.check_scoring')
get_ipython().magic(u'pinfo sklearn.cross_validation.check_cv')
import sklearn.svm
get_ipython().magic(u'pinfo sklearn.svm.SVC')
get_ipython().magic(u'pinfo sklearn.svm.SVC')
get_ipython().magic(u'pinfo sklearn.svm.SVC')
get_ipython().magic(u'pinfo sklearn.svm.SVC')
help(sklearn.svm.SVC)
get_ipython().set_next_input(u'help(sklearn.svm.SVC');get_ipython().magic(u'pinfo sklearn.svm.SVC')
get_ipython().magic(u'pinfo sklearn.svm.SVC')
help(sklearn.svm.SVC)
get_ipython().magic(u'pinfo sklearn.svm.SVC')
X_test.max
X_test.max()
X_test.min()
from  sklearn.learning_curve import  learning_curve
learning_curve>
get_ipython().magic(u'pinfo learning_curve')
get_ipython().magic(u'pinfo learning_curve')
get_ipython().magic(u'pinfo learning_curve')
estimator = GaussianNB()
learning_curve(estimator,data.data,data,target)
learning_curve(estimator,data.data,data.target)
estimator.fit(X_train,y_train)
estimator.fit(X_train,y_train)
estimator.fit(X_train,y_train)
estimator.fit(X_train,y_train)
estimator.fit(X_train,y_train)
estimator.fit(X_train,y_train)
estimator.fit(X_train,y_train)
estimator.fit(X_train,y_train)
learning_curve(estimator,data.data,data.target)
get_ipython().magic(u'pinfo _')
get_ipython().magic(u'pinfo learning_curve')
estimator.name
estimator.__name__
print estimator
estimator = GaussianNB
learning_curve(estimator,data.data,data.target)
learning_curve(estimator(),data.data,data.target)
learning_curve(estimator(),data.data,data.target,train_sizes=np.linspace(0.1,0.5,5))
learning_curve(estimator,data.data,data.target,train_sizes=np.linspace(0.1,0.5,5))
learning_curve(estimator,data.data,data.target,train_sizes=np.linspace(0.1,0.5,5))
get_ipython().magic(u'pinfo learning_curve')
from sklearn.linear_model.logistic import LogisticRegression
get_ipython().magic(u'pinfo LogisticRegression')
t = LogisticRegression()
t.fit
t.predict
get_ipython().magic(u'pinfo GaussianNB')
get_ipython().magic(u'pinfo LogisticRegression')
from sklearn.linear_model import LogisticRegressionCV
get_ipython().magic(u'pinfo LogisticRegressionCV')
get_ipython().magic(u'pinfo LogisticRegressionCV')
get_ipython().magic(u'pinfo GaussianNB')
get_ipython().magic(u'pinfo GaussianNB')
get_ipython().magic(u'pinfo GaussianNB.__init__')
learning_curve(LogisticRegression,data.data,data.target,train_sizes=np.linspace(0.1,0.5,5))
learning_curve(LogisticRegression(),data.data,data.target,train_sizes=np.linspace(0.1,0.5,5))
learning_curve(GaussianNB(),data.data,data.target,train_sizes=np.linspace(0.1,0.5,5))
from sklearn.neural_network.rbm import BernoulliRBM
get_ipython().magic(u'pinfo BernoulliRBM')
from sklearn.neural_network import BernoulliRBM
get_ipython().magic(u'pinfo BernoulliRBM')
get_ipython().set_next_input(u'from sklearn.linear_model.Perceptron');get_ipython().magic(u'pinfo sklearn.linear_model.Perceptron')
get_ipython().magic(u'pinfo sklearn.linear_model.Perceptron')
from sklearn.tree import DecisionTreeClassifier
get_ipython().magic(u'pinfo DecisionTreeClassifier')
get_ipython().magic(u'pinfo learning_curve')
learning_curve(GaussianNB(),data.data,data.target,train_sizes=np.linspace(0.1,0.5,5))
learning_curve(GaussianNB(),data.data,data.target,train_sizes=np.linspace(0.1,0.5,5),verbose=1)
y
y_test
data.targe
data.target
y = data.target
np.mean(y)
np.std(y)
plt
plt.plot(y)
plt.show()
clf
clf.__init__.__doc__
get_ipython().magic(u'pinfo clf')
get_ipython().magic(u'pinfo LogisticRegression')
from sklearn.linear_model import 
from sklearn.linear_model import ridge_regression
get_ipython().magic(u'pinfo ridge_regression')
quit()
