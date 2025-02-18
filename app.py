import joblib
from datetime import datetime
from flask import Flask, request, jsonify, render_template
import numpy as np

# Initialisation de Flask
app = Flask(__name__)

# Chargement du StandardScaler et du modèle
scaler = joblib.load("scaler.pkl")
model = joblib.load("best_xgboost_model.h5")

print("Modèle et scaler chargés avec succès !")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.content_type == 'application/json':
            data = request.json
            features = data["features"]
        else:
            features = [
                request.form.get("transaction_id"),
                request.form.get("timestamp"),
                float(request.form.get("amount")),
                request.form.get("merchant_id"),
                request.form.get("customer_id"),
                request.form.get("transaction_type"),
                request.form.get("country"),
                request.form.get("device_id"),
                request.form.get("ip_address"),
                request.form.get("merchant_category"),
                int(request.form.get("hour_of_day"))
            ]

        # Transformation des features (encodage et conversion)
        features[0] = int(features[0][1:])
        features[3] = int(features[3][1:])
        features[4] = int(features[4][1:])
        features[7] = int(features[7][1:])
        features[8] = int(features[8].split('.')[-1])

        transaction_type_mapping = {"payment": 0, "withdrawal": 1, "deposit": 2, "transfer": 3}
        features[5] = transaction_type_mapping.get(features[5], -1)

        country_mapping = {"FR": 0, "US": 1, "DE": 2, "UK": 3, "ES": 4, "IT": 5, "JP": 6, "CN": 7}
        features[6] = country_mapping.get(features[6], -1)

        merchant_category_mapping = {"retail": 0, "restaurant": 1, "technology": 2, "travel": 3, "entertainment": 4}
        features[9] = merchant_category_mapping.get(features[9], -1)

        # Conversion timestamp avec gestion des différents formats
        try:
            timestamp = features[1]

            # Vérifier si le format contient un 'T' (format ISO 8601)
            if "T" in timestamp:
                dt = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M")
            else:
                dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")

            # Convertir en timestamp UNIX
            features[1] = int(dt.timestamp())

        except ValueError:
            return jsonify(
                {"error": "Format de date invalide. Utilisez 'YYYY-MM-DD HH:MM:SS' ou 'YYYY-MM-DDTHH:MM'."}), 400

        # Ajout des features temporelles
        features.extend([dt.hour, dt.weekday(), dt.month, dt.isocalendar()[1]])

        # Vérification du format des données
        if len(features) != 15:
            return jsonify({"error": f"Feature shape mismatch: expected 15, got {len(features)}"}), 400

        # Normalisation des données
        features = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features)

        # Prédiction model
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]

        # débogage
        print("Données normalisées :", features_scaled)
        print("Prédiction :", prediction)
        print("Probabilité :", probability)

        return jsonify({"prediction": int(prediction), "probability": float(probability)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, host='0.0.0.0')

