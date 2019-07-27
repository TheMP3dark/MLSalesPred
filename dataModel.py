from dataPlots import data
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

x_items = ["Age", "Product_ID", "Call_Count", "genderBool"]
x = data[x_items]
y = data.saleBool
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

model = DecisionTreeClassifier(max_depth=10, random_state=3)
model.fit(x_train, y_train)
predVals = model.predict(x_test)

print("Accuracy score for decision tree classifier model (using gini index) is: ", str(100*accuracy_score(y_test, predVals)), "%")

model = DecisionTreeClassifier(criterion="entropy", random_state=3)
model.fit(x_train, y_train)
predVals = model.predict(x_test)

print("Accuracy score for decision tree classifier model (using entropy) is: ", str(100*accuracy_score(y_test, predVals)), "%")

model = LogisticRegression(random_state=3)
model.fit(x_train, y_train)
predVals = model.predict(x_test)

print("Accuracy score for logistic regression model is: ", str(100*accuracy_score(y_test, predVals)), "%")

model = RandomForestClassifier(random_state=3)
model.fit(x_train, y_train)
predVals = model.predict(x_test)

print("Accuracy score for logistic regression model is: ", str(100*accuracy_score(y_test, predVals)), "%")

# observation: logistic regressor gives best result