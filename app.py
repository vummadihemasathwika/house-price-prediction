from flask import Flask, request, render_template
import pickle
import pandas as pd

# Create app
app = Flask(__name__)

# Load trained pipeline model
model = pickle.load(open("model.pkl", "rb"))

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = {
        'longitude': float(request.form['longitude']),
        'latitude': float(request.form['latitude']),
        'housing_median_age': float(request.form['housing_median_age']),
        'total_rooms': float(request.form['total_rooms']),
        'total_bedrooms': float(request.form['total_bedrooms']),
        'population': float(request.form['population']),
        'households': float(request.form['households']),
        'median_income': float(request.form['median_income']),
        'ocean_proximity': request.form['ocean_proximity']
    }

    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]

    return render_template('index.html',
                           prediction_text=f"Estimated Price: ₹ {round(prediction,2)}")

 

# Run app
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)