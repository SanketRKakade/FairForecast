from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the pre-trained SVM model
svm_model = joblib.load('models/svm_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    features = np.array([[float(data['MA50']), float(data['RSI']), float(data['Close']), float(data['Volume'])]])

    prediction = svm_model.predict(features)

    return f"The predicted direction is: {'Up' if prediction[0] == 1 else 'Down'}"

if __name__ == '__main__':
    app.run(debug=True)
