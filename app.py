"""Módulo principal de la aplicación Flask para clasificar insectos."""
from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging
import numpy as np

#  Corrección aquí: __name__ con doble guion bajo
app = Flask(__name__)

# Configurar el registrodxd
logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo entrenado
try:
    model = joblib.load('modelo_insectos.pkl')
    app.logger.debug('Modelo cargado correctamente.')
except FileNotFoundError:
    model = None
    app.logger.error('No se pudo cargar el modelo: modelo_insectos.pkl no encontrado.')

# Definir media y desviación estándar basadas en el conjunto de entrenamiento
mean_values = np.array([4.4, 5.17])  # Media de [abdomen, antena]
std_values = np.array([3.05, 2.65])  # Desviación estándar de [abdomen, antena]

@app.route('/')
def home():
    """Renderiza la plantilla del formulario."""
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Realiza una predicción basada en los datos del formulario."""
    if model is None:
        return jsonify({'error': 'Modelo no cargado correctamente'}), 500

    try:
        # Obtener los datos enviados en el request
        abdomen = float(request.form['abdomen'])
        antenna = float(request.form['antenna'])

        # Crear un DataFrame con los datos
        data_df = pd.DataFrame([[abdomen, antenna]], columns=['abdomen', 'antenna'])
        app.logger.debug(f'DataFrame creado: {data_df}')

        # Normalizar los datos manualmente
        data_array = data_df.values
        data_scaled = (data_array - mean_values) / std_values
        app.logger.debug(f'Datos normalizados: {data_scaled}')

        # Realizar predicciones
        prediction = model.predict(data_scaled)
        app.logger.debug(f'Predicción: {prediction[0]}')

        return jsonify({'categoría': prediction[0]})
    
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

# Corrección aquí también: __name__ y __main__ con doble guion bajo
if __name__ == '__main__':
    app.run(debug=True)
