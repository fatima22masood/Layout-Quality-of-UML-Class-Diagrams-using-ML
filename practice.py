import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.naive_bayes import GaussianNB 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, classification_report

uml_data= pd.read_csv(r'E:\MS\Machine learning\paper\UML class diagrams for layout quality checking\Data\features_labels.csv')
X = uml_data.iloc[:, :-1].values 
y = uml_data.iloc[:, -1].values

plt.scatter(X[:, 0], X[:, 1], c=y) 
plt.xlabel('RectOrth') 
plt.ylabel('LineCrossings') 
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = DecisionTreeClassifier() 
clf.fit(X_train, y_train)

# y_pred = clf.predict(X_test) 
# accuracy = accuracy_score(y_test, y_pred) 
# print('Accuracy:', accuracy)
