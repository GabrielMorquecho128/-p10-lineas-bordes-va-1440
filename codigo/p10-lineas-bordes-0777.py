# marco morquecho
# (EJEMPLO 2 — Detección de esquinas con Harris) + experimentar

import cv2
import numpy as np
import os

# Obtener la ruta base del proyecto
dir_script = os.path.dirname(os.path.abspath(__file__))
dir_proyecto = os.path.abspath(os.path.join(dir_script, '..'))

ruta_imagen = os.path.join(dir_proyecto, 'imagenes', 'avestruz.jpg')
dir_resultados = os.path.join(dir_proyecto, 'resultados')

# Crear la carpeta resultados si no existe
os.makedirs(dir_resultados, exist_ok=True)

# 1. Cargar la imagen
img_original = cv2.imread(ruta_imagen)

if img_original is None:
    print(f"Error: No se encontró la imagen en: {ruta_imagen}")
    exit()

img_harris = img_original.copy()
img_exp = img_original.copy()

# Convertir a escala de grises
gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
gray_float = np.float32(gray)

# --- EJEMPLO 2: Harris Base ---
dst = cv2.cornerHarris(gray_float, blockSize=2, ksize=3, k=0.04)
dst = cv2.dilate(dst, None)
img_harris[dst > 0.01 * dst.max()] = [0, 0, 255] # Rojo

# --- EXPERIMENTACIÓN ---
dst_exp = cv2.cornerHarris(gray_float, blockSize=5, ksize=5, k=0.04)
dst_exp = cv2.dilate(dst_exp, None)
img_exp[dst_exp > 0.05 * dst_exp.max()] = [0, 255, 0] # Verde

# Definir rutas de salida
ruta_out_base = os.path.join(dir_resultados, 'esquinas_harris_base.jpg')
ruta_out_exp = os.path.join(dir_resultados, 'esquinas_harris_experimento.jpg')

# Guardar y verificar
exito_base = cv2.imwrite(ruta_out_base, img_harris)
exito_exp = cv2.imwrite(ruta_out_exp, img_exp)

if exito_base and exito_exp:
    print(f"¡Imágenes guardadas correctamente en:\n{dir_resultados}")
else:
    print("Hubo un problema al guardar las imágenes en la carpeta de resultados.")

# Mostrar las 3 ventanas
cv2.imshow('1. Imagen Original 1440', img_original)
cv2.imshow('2. Harris Base (Rojo) 1440', img_harris)
cv2.imshow('3. Harris Experimento (Verde) 1440', img_exp)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("marco morquecho 1440")