import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import joblib



df = pd.read_csv('data.cvs')

df = df.drop(["PassengerId", "Name", "Ticket", "Cabin", "Fare", "Embarked"], axis=1)
df = df.replace({'male': 0, 'female': 1})
df = df.dropna()


Y = df.pop('Survived')
X = df

model = DecisionTreeClassifier()
model.fit(X, Y)

joblib.dump(model, "model.ml")
