import pickle
from flask import Flask, request, render_template
import numpy as np

# Importing model and scalers
try:
    model = pickle.load(open('model.pkl', 'rb'))
    sc = pickle.load(open('standscaler.pkl', 'rb'))
    ms = pickle.load(open('minmaxscaler.pkl', 'rb'))
except FileNotFoundError as e:
    print(f"Error loading model or scalers: {e}")
    model, sc, ms = None, None, None

# Creating Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    try:
        # Extract input data
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosporus'])
        K = float(request.form['Potassium'])
        temp = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['Ph'])
        rainfall = float(request.form['Rainfall'])

        # Preprocess features
        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)

        # Scale features if scalers are available
        if ms and sc:
            scaled_features = ms.transform(single_pred)
            final_features = sc.transform(scaled_features)
        else:
            final_features = single_pred

        # Predict crop
        prediction = model.predict(final_features)
        crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut",
                     6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon",
                     10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                     14: "Pomegranate", 15: "Lentil", 16: "Blackgram",
                     17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas",
                     20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

        # Get result
        crop = crop_dict.get(prediction[0], "Unknown crop")
        result = f"{crop} is the best crop to be cultivated right there."
    except Exception as e:
        result = f"Error: {str(e)}"

    return render_template('index.html', result=result)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
