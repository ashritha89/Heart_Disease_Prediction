from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load ML Model
model = pickle.load(open("model/heart_model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data for all 13 features
    feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                     'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    
    # Extract values from form and convert to float
    features = [float(request.form.get(feature)) for feature in feature_names]
    
    # Convert into numpy array and reshape for model
    input_data = np.array(features).reshape(1, -1)
    
    # Prediction
    prediction = model.predict(input_data)[0]
    
    # Render result page with prediction output
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run()
