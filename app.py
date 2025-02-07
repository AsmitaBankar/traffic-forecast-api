from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hide TensorFlow warnings

# Load the trained LSTM model
model = tf.keras.models.load_model("traffic_forecast_model.h5")

# Initialize Flask app
app = Flask(__name__)

@app.route("/", methods=["GET"])  # Home route
def home():
    return "Smart City Traffic Forecasting API is Running!"

@app.route("/predict", methods=["POST"])  # Prediction route
def predict():
    try:
        data = request.get_json()
        if not data or "traffic_data" not in data:
            return jsonify({"error": "Invalid input"}), 400

        # Convert input data for LSTM model
        traffic_sequence = np.array(data["traffic_data"]).reshape(1, 10, 1)

        # Make prediction
        prediction = model.predict(traffic_sequence)[0][0]

        return jsonify({"predicted_traffic_count": float(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Use Render's assigned port
    app.run(host="0.0.0.0", port=port, debug=True)
