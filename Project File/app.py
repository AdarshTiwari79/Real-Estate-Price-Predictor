from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Loading pre-trained model
model = joblib.load('Real_estate_price_predictor.joblib')

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Extract features from the form
        feature1 = float(request.form['CRIM'])
        feature2 = float(request.form['ZN'])
        feature3 = float(request.form['INDUS'])
        feature4 = float(request.form['CHAS'])
        feature5 = float(request.form['NOX'])
        feature6 = float(request.form['RM'])
        feature7 = float(request.form['AGE'])
        feature8 = float(request.form['DIS'])
        feature9 = float(request.form['RAD'])
        feature10 = float(request.form['TAX'])
        feature11 = float(request.form['PTRATIO'])
        feature12 = float(request.form['B'])
        feature13 = float(request.form['LSTAT'])

        # Prepare the features for prediction
        features = np.array([[feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9, feature10, feature11, feature12, feature13]])  
        # Make prediction
        prediction = model.predict(features)

        return render_template('result.html', prediction=prediction[0])
    else:
        return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
