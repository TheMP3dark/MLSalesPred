import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data = pd.read_excel("Book1.xlsx", sheet_name="Sheet1")
data = data.dropna()

x_items = ["Age", "Product_ID", "Call_Count"]
x = data[x_items]
y = data.Sale
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)
#
# model = DecisionTreeClassifier()
# model.fit(x_train, y_train)
# predVals = model.predict(x_test)
#
# print("Accuracy score for decision tree classifier model (using gini index) is: ", str(100*accuracy_score(y_test, predVals)), "%")
#
# model = DecisionTreeClassifier(criterion="entropy")
# model.fit(x_train, y_train)
# predVals = model.predict(x_test)
#
# print("Accuracy score for decision tree classifier model (using entropy) is: ", str(100*accuracy_score(y_test, predVals)), "%")
#
# model = LogisticRegression()
# model.fit(x_train, y_train)
# predVals = model.predict(x_test)
#
# print("Accuracy score for logistic regression model is: ", str(100*accuracy_score(y_test, predVals)), "%")

model = LogisticRegression()
model.fit(x_train, y_train)
predVals = model.predict(x_test)

print("Accuracy score for logistic regression model is: ", str(100*accuracy_score(y_test, predVals)), "%")