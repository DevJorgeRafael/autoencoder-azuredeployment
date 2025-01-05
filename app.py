from flask import Flask, request, render_template
from keras.models import load_model
import numpy as np
import re

app = Flask(__name__)

# Cargar el modelo
model = load_model('autoencoder_model.keras')

# Umbral para clasificar las transacciones
THRESHOLD = 0.75

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Validar que el input cumple con el formato
        data = request.form['data']
        if not re.match(r'^\s*-?\d+(\.\d+)?(\s*,\s*-?\d+(\.\d+)?){28}\s*$', data):
            return render_template('result.html', prediction="Formato inválido. Ingrese 29 valores separados por comas.")
        
        # Convertir el input en un array numpy
        data = np.array([float(i) for i in data.split(',')]).reshape(1, -1)
        
        # Realizar la predicción
        prediction = model.predict(data)
        
        # Calcular el MSE
        mse = np.mean(np.power(data - prediction, 2), axis=1)
        
        # Clasificar como fraudulento o no basado en el umbral
        is_fraudulent = mse > THRESHOLD
        result = "Fraudulenta" if is_fraudulent else "Normal"
        
        return render_template('result.html', prediction=f"La transacción es: {result}")
    
    except Exception as e:
        return render_template('result.html', prediction=f"Error al procesar los datos: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
