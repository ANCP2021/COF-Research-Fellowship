import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import AdaBoostClassifier

dataframe = pd.read_csv("./../preprocessing/data_bin.csv")
dataframe = dataframe[dataframe.columns.drop(list(dataframe.filter(regex='Unnamed')))]

x = dataframe.drop(' Label', axis=1)
y = dataframe[' Label']

ms = MinMaxScaler()
x = ms.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.7, test_size=0.3)

Classifier_accuracy = []

# K-Nearest Neighbor Classifier
k_nearest_neighbor_classifier = KNeighborsClassifier()
k_nearest_neighbor_classifier.fit(X_train, y_train)
y_pred = k_nearest_neighbor_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
Classifier_accuracy.append(accuracy * 100)
print("Accuracy of KNN Classifier : %.2f" % (accuracy * 100))

# SVM classifier
svm_classifier = SVC()
svm_classifier.fit(X_train, y_train)
y_pred = svm_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
Classifier_accuracy.append(accuracy * 100)
print("Accuracy of SVM Classifier : %.2f" % (accuracy * 100))

# Decision Tree Classifier
decision_tree_classifier = DecisionTreeClassifier(max_depth=6)
decision_tree_classifier.fit(X_train, y_train)
y_pred = decision_tree_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
Classifier_accuracy.append(accuracy * 100)
print("Accuracy of Decision Tree Classifier : %.2f" % (accuracy * 100))

# Random Forest Classifier
random_forest_classifier = RandomForestClassifier(max_depth = 2)
random_forest_classifier.fit(X_train, y_train)
y_pred = random_forest_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
Classifier_accuracy.append(accuracy * 100)
print("Accuracy of Random Forest Classifier : %.2f" % (accuracy * 100))

# Quadraic Discriminant Analysis Classifier
quadratic_discriminant_classifier = QuadraticDiscriminantAnalysis()
quadratic_discriminant_classifier.fit(X_train, y_train)
y_pred = quadratic_discriminant_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
Classifier_accuracy.append(accuracy * 100)
print("Accuracy of QDA Classifier : %.2f" % (accuracy * 100))

# Linear Discriminant Analysis Classifier
linear_discriminant_classifier = LinearDiscriminantAnalysis()
linear_discriminant_classifier.fit(X_train, y_train)
y_pred = linear_discriminant_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
Classifier_accuracy.append(accuracy * 100)
print("Accuracy of LDA Classifier : %.2f" % (accuracy * 100))

# Stochastic Gradient Classifier
stochastic_gradient_classifier = SGDClassifier(loss='hinge', penalty='l2')
stochastic_gradient_classifier.fit(X_train, y_train)
y_pred = stochastic_gradient_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
Classifier_accuracy.append(accuracy * 100)
print("Accuracy of SGD Classifier : %.2f" % (accuracy * 100))

# Logistic Regression
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)
y_pred = logistic_regression.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
Classifier_accuracy.append(accuracy * 100)
print("Accuracy of Logistic Regression Classifier : %.2f" % (accuracy * 100))

# XGBoost Classifier
xgboost_classifier = XGBClassifier(eval_metric='error', objective='binary:logistic', max_depth=2, learning_rate=0.1)
xgboost_classifier.fit(X_train, y_train)
y_pred = xgboost_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
Classifier_accuracy.append(accuracy * 100)
print("Accuracy of XGBoost Classifier : %.2f" % (accuracy * 100))

# Adaptive Boost Classifier
adaBoost_classifier = AdaBoostClassifier(n_estimators=100, random_state=0)
adaBoost_classifier.fit(X_train, y_train)
y_pred = adaBoost_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
Classifier_accuracy.append(accuracy * 100)
print("Accuracy of AdaBoost Classifier : %.2f" % (accuracy * 100))

print(Classifier_accuracy)