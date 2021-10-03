from flask import Flask, render_template, request, jsonify
import re
import predicting_heading,predicting_text

app = Flask(__name__)
# Load model and vectorizer
#new_models=keras.models.load_model("fakenews")
# Build functionalities
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')
def predict_heading(text):
    prediction=str(predicting_heading.predict([text]))
    #prediction ="FAKE"
    return prediction #return fake or real
def predict_text(text):
    prediction=str(predicting_text.predict([text]))
    #prediction ="FAKE"
    return prediction #return fake or real
@app.route('/', methods=['POST'])
def webapp():
    heading = request.form['heading']
    #text=request.form['text']
    text=request.form['text']
    predicted_header =(predict_heading(heading))
    predicted_text=predict_text(text)
    prediction=f"Heading Truth: {predicted_header} and Text Truth: {predicted_text}"
    return render_template('index.html', heading=heading,text=text, result=prediction)
@app.route('/predict/', methods=['GET','POST'])
def api():
    text = request.args.get("text")
    prediction = predict(text)
    return jsonify(prediction=prediction)
if __name__ == "__main__":
    app.run()
