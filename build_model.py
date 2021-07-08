import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import joblib



df = pd.read_csv('https://cdn.techroulette.xyz/projects/shipwreck/data/m3_data.csv')

df = df.drop(["PassengerId", "Name", "Ticket", "Cabin", "Fare", "Embarked"], axis=1)
df = df.replace({'male': 0, 'female': 1})
df = df.dropna()


Y = df.pop('Survived')
X = df

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15)

model = DecisionTreeClassifier()
model.fit(X_train, Y_train)

joblib.dump(model, "model.ml")
