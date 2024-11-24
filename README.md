# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the Iris dataset and create a DataFrame with feature names and target labels.
2. Separate the data into features (`X`) and target (`y`).
3. Split the data into training and testing sets with an 80-20 ratio.
4. Initialize a Stochastic Gradient Descent (SGD) classifier and train it on the training data.
5. Predict the target values for the test set.
6. Calculate and display the model's accuracy.
7. Compute and display the confusion matrix for the predictions.

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Aaron I
RegisterNumber:  212223230002
*/

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

iris = load_iris()

df = pd.DataFrame(data = iris.data, columns = iris.feature_names)
df['target'] = iris.target

print(df.head())

x = df.iloc[:, :-1]
y = df['target']
# print("X :")
# print(x)
# print("Y :")
# print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/4, random_state = 42)

sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)

sgd_clf.fit(x_train, y_train)

y_pred = sgd_clf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy :",accuracy)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix : \n",cm)
```

## Output:
![image](https://github.com/user-attachments/assets/c7cee799-f4db-44b1-b23a-90ae541e1e24)



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
