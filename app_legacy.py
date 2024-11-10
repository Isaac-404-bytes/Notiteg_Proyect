import cv2
import numpy as np
import mysql.connector
import os
import requests
from datetime import datetime
from prettytable import PrettyTable



# Conectarse a la base de datos MySQL
conexion_bd = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="proyecto"
)

cursor = conexion_bd.cursor()

# Cargar el clasificador frontal de Haar para la detección de caras
cascada_cara = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicializar el reconocedor facial LBPH (Local Binary Patterns Histograms)
reconocedor = cv2.face.LBPHFaceRecognizer_create()

# Función para capturar y almacenar las imágenes faciales en la base de datos
def capturar_imagenes(nombre_usuario):
    camara = cv2.VideoCapture(0)
    
    # Capturar múltiples imágenes para mejorar la precisión
    imagenes = []
    etiquetas = []
    contador = 0
    while True:
        ret, imagen = camara.read()
        
        if not ret:  # Comprobar si la captura fue exitosa
            print("Error al capturar la imagen.")
            continue
        
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        caras = cascada_cara.detectMultiScale(gris, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
        
        for (x, y, w, h) in caras:
            roi_gris = gris[y:y+h, x:x+w]
            roi_gris = cv2.resize(roi_gris, (300, 300))  # Redimensionar la imagen
            imagenes.append(roi_gris)
            etiquetas.append(nombre_usuario)
            cv2.imwrite(f"usuario_{nombre_usuario}_{contador}.jpg", roi_gris)
            contador += 1
            cv2.rectangle(imagen, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.waitKey(100)
        
        cv2.imshow('Capturando Imagenes', imagen)
        cv2.waitKey(1)
        if contador >= 100:
            break
            
    camara.release()
    cv2.destroyAllWindows()
    return imagenes, etiquetas

# Función para entrenar el reconocedor facial con las imágenes almacenadas en la base de datos
def entrenar_reconocedor():
    caras = []
    ids = []
    
    cursor.execute("SELECT id, nombre FROM usuarios")
    resultados = cursor.fetchall()
    
    for id, nombre in resultados:
        imagenes_usuario, etiquetas_usuario = [], []
        for i in range(100):
            ruta_imagen = f"usuario_{nombre}_{i}.jpg"
            if os.path.exists(ruta_imagen):
                imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
                imagenes_usuario.append(imagen)
                etiquetas_usuario.append(int(id))
            else:
                print(f"El archivo {ruta_imagen} no existe.")
        
        caras.extend(imagenes_usuario)
        ids.extend(etiquetas_usuario)
    
    if caras and ids:
        reconocedor.train(caras, np.array(ids))
        reconocedor.save("reconocedor.yml")
    else:
        print("No se pudo entrenar el reconocedor debido a la falta de datos.")

# Función para detectar y reconocer caras en tiempo real

def obtener_lugar_por_ip():
    response = requests.get('https://ipinfo.io/json')
    datos = response.json()
    return datos.get('city')

def guardar_reconocimiento(id_usuario):
    now = datetime.now()
    fecha_hora = now.strftime('%Y-%m-%d %H:%M:%S')
    consulta = "INSERT INTO reconocimientos (id_usuario,fecha_hora,lugar) VALUES (%s,%s,%s)"
    valores = (id_usuario, fecha_hora, obtener_lugar_por_ip())
    cursor.execute(consulta, valores)
    conexion_bd.commit()
   
def detectar_reconocer_caras():
    cursor = conexion_bd.cursor()
    reconocedor.read("reconocedor.yml")
    camara = cv2.VideoCapture(0)
    detectado = 0
    while True:
        ret, imagen = camara.read()
        
        if not ret:  # Comprobar si la captura fue exitosa
            print("Error al capturar la imagen.")
            continue
        
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        caras = cascada_cara.detectMultiScale(gris, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
        
        for (x, y, w, h) in caras:
            roi_gris = gris[y:y+h, x:x+w]
            roi_gris = cv2.resize(roi_gris, (300, 300))  # Redimensionar la imagen
            id_, confianza = reconocedor.predict(roi_gris)
            
            cursor.execute("SELECT id,nombre FROM usuarios WHERE id = {}".format(id_))

            fila_usuario = cursor.fetchone()
            if fila_usuario is not None:
                id_usuario = fila_usuario[0]
                nombre_usuario = fila_usuario[1]
             
            if confianza < 50:
                cv2.putText(imagen, nombre_usuario, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                guardar_reconocimiento(id_usuario)
            else:
                cv2.putText(imagen, "Desconocido", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
            cv2.rectangle(imagen, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.imshow('Reconociendo Caras', imagen)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    camara.release()
    cv2.destroyAllWindows()

# Función para agregar un nuevo usuario a la base de datos
def agregar_usuario():
    cursor = conexion_bd.cursor()
    nombre_usuario = input("Ingrese el nombre del nuevo usuario: ")
    cursor.execute("INSERT INTO usuarios (nombre) VALUES ('{}')".format(nombre_usuario))
    imagenes, etiquetas = capturar_imagenes(nombre_usuario)
    conexion_bd.commit()
    print("Usuario {} agregado correctamente.".format(nombre_usuario))

def mostrar_registros_reconocimientos():
    try:
        # Crear un cursor para ejecutar consultas SQL
        cursor = conexion_bd.cursor()

        # Consulta SQL para obtener todos los registros de reconocimientos
        sql = """
        SELECT r.fecha_hora, r.lugar, u.nombre 
        FROM reconocimientos AS r
        JOIN usuarios AS u ON r.id_usuario = u.id
        """
        cursor.execute(sql)

        # Obtener todos los resultados de la consulta
        resultados = cursor.fetchall()

# Crear una tabla PrettyTable
        tabla = PrettyTable(["Nombre", "Fecha y hora", "Lugar"])

        # Agregar filas a la tabla
        for fecha_hora, lugar, nombre_usuario in resultados:
            tabla.add_row([nombre_usuario, fecha_hora, lugar])

        # Imprimir la tabla
        print(tabla)

    except mysql.connector.Error as error:
        print("Error al obtener los registros de reconocimientos:", error)
# Función principal

def main():
    while True:
        print("\nOpciones:")
        print("1. Agregar usuario")
        print("2. Entrenar reconocedor")
        print("3. Detectar y reconocer caras en tiempo real")
        print("4. Listar Todos Los Reconocimientos")
        print("5. Salir")
        
        opcion = input("\nSeleccione una opción: ")
        
        if opcion == "1":
            agregar_usuario()
        elif opcion == "2":
            entrenar_reconocedor()
        elif opcion == "3":
            detectar_reconocer_caras()
        elif opcion == "4":
            mostrar_registros_reconocimientos()
        elif opcion == "5":
            break
        else:
            print("Opción no válida.")

if __name__ == "__main__":
    main()
    
# Cerrar la conexión con la base de datos al finalizar
conexion_bd.close()