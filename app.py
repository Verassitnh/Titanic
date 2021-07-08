from flask import Flask
import joblib

app = Flask(__name__)
model = joblib.load("model.ml")

def predict(pclass ,age, sex, sibSp, parCh):
    return model.predict([[pclass, sex, age, sibSp, parCh]])




@app.route("/api/predict/<int:pclass>/<int:age>/<int:sex>/<int:sibsp>/<int:parch>")
def hello_world(pclass, age, sex, sibsp, parch):
    return "You will die" if predict(pclass, age, sex, sibsp, parch)[0] == 0 else "You will live"
