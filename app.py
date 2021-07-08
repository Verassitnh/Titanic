import joblib

model = joblib.load("model.ml")

def predict(pclass ,age, sex, sibSp, parCh):
    return model.predict([[pclass, sex, age, sibSp, parCh]])