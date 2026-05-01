from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

model = LogisticRegression()
model.fit(X_train, y_train)