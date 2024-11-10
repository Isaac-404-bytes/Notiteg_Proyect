from flask import Flask, render_template, request, redirect, url_for
import cv2
import os
import mysql.connector
import numpy as np
from datetime import datetime

app = Flask(__name__)

# Configuración de la conexión a la base de datos
conexion_bd = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="proyecto"
)
cursor = conexion_bd.cursor()

# Cargar el clasificador frontal de Haar para la detección de caras
cascada_cara = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicializar el reconocedor facial LBPH
reconocedor = cv2.face.LBPHFaceRecognizer_create()

# Leer el modelo entrenado
if os.path.exists("reconocedor.yml"):
    reconocedor.read("reconocedor.yml")


# Función para obtener el último lugar de reconocimiento
def obtener_ultimo_reconocimiento(id_usuario):
    cursor.execute("""
        SELECT fecha_hora, lugar 
        FROM reconocimientos 
        WHERE id_usuario = %s 
        ORDER BY fecha_hora DESC LIMIT 1
    """, (id_usuario,))
    resultado = cursor.fetchone()
    if resultado:
        return resultado
    return None


# Función para detectar y reconocer una cara desde una imagen subida por el usuario
def detectar_reconocer_cara_desde_imagen(image_path):
    imagen = cv2.imread(image_path)
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    caras = cascada_cara.detectMultiScale(gris, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    if len(caras) == 0:
        return "No se detectaron caras en la imagen.", None

    for (x, y, w, h) in caras:
        roi_gris = gris[y:y+h, x:x+w]
        roi_gris = cv2.resize(roi_gris, (300, 300))
        id_, confianza = reconocedor.predict(roi_gris)

        if confianza < 50:
            cursor.execute("SELECT id, nombre FROM usuarios WHERE id = %s", (id_,))
            fila_usuario = cursor.fetchone()
            if fila_usuario:
                id_usuario = fila_usuario[0]
                nombre_usuario = fila_usuario[1]

                # Obtener el último reconocimiento
                ultimo_reconocimiento = obtener_ultimo_reconocimiento(id_usuario)

                if ultimo_reconocimiento:
                    return f"Reconocido: {nombre_usuario}. Último reconocimiento en {ultimo_reconocimiento[1]} el {ultimo_reconocimiento[0]}.", nombre_usuario
                else:
                    return f"Reconocido: {nombre_usuario}.", nombre_usuario
        else:
            return "Persona desconocida.", None

    return "No se pudo reconocer ninguna cara.", None

# Ruta para la página principal (nuevo index.html)
@app.route('/')
def home():
    return render_template('index.html')

# Ruta para la página de búsqueda (busqueda.html)
@app.route('/busqueda')
def busqueda():
    return render_template('busqueda.html', resultado=None, historial=[])

# Ruta para subir una imagen y procesarla
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return render_template('busqueda.html', resultado="No se ha subido ninguna imagen.", historial=[])

    image_file = request.files['image']
    if image_file.filename == '':
        return render_template('busqueda.html', resultado="Archivo no válido.", historial=[])

    # Guardar la imagen temporalmente
    image_path = os.path.join('static', image_file.filename)
    image_file.save(image_path)

    # Llamar a la función para detectar y reconocer la cara
    resultado, nombre_usuario = detectar_reconocer_cara_desde_imagen(image_path)

    # Si se reconoce al usuario, obtener el historial de reconocimientos
    historial = []
    if nombre_usuario:
        cursor.execute("""
            SELECT fecha_hora, lugar 
            FROM reconocimientos AS r
            JOIN usuarios AS u ON r.id_usuario = u.id
            WHERE u.nombre = %s
            ORDER BY fecha_hora DESC
        """, (nombre_usuario,))
        historial = cursor.fetchall()

    # Borrar la imagen temporal después de usarla
    os.remove(image_path)

    # Mostrar el resultado y el historial (si existe) en la página
    return render_template('busqueda.html', resultado=resultado, nombre_usuario=nombre_usuario, historial=historial)


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=os.getenv("PORT", default=5000))