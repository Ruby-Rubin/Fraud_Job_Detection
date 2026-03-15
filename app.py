from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    text = request.form["job"]

    vector = vectorizer.transform([text])

    prediction = model.predict(vector)

    if prediction[0] == 1:
        result = "Fraudulent Job"
    else:
        result = "Real Job"

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run()