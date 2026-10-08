# marco morquecho
# (EJEMPLO 2 — Detección de esquinas con Harris) + experimentar
import cv2
import numpy as np

# 1. Cargar imagen de la avestruz
imagen = cv2.imread("../imagenes/avestruz.jpg")

# Verificar que la imagen exista
if imagen is None:
    print("Error: no se pudo cargar la imagen.")
    exit()

# 2. Convertir a escala de grises
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Convertir a tipo float32
gris_float = np.float32(gris)

# 3. Detectar esquinas mediante Harris
esquinas = cv2.cornerHarris(
    gris_float,
    2,
    3,
    0.04
)

# Dilatar para hacer visibles las esquinas
esquinas = cv2.dilate(
    esquinas,
    None
)

# Crear copia para el resultado final
resultado = imagen.copy()

# Umbral para identificar esquinas
umbral = 0.01 * esquinas.max()

# Marcar esquinas en rojo sobre la copia
resultado[esquinas > umbral] = [0, 0, 255]

# Normalizar la matriz Harris a rango 0-255 para poder visualizarla como la 3ra imagen
esquinas_norm = cv2.normalize(
    esquinas,
    None,
    alpha=0,
    beta=255,
    norm_type=cv2.NORM_MINMAX,
    dtype=cv2.CV_8U
)

# 4. Mostrar las 3 ventanas / imágenes
cv2.imshow(
    "1. Imagen original",
    imagen
)
cv2.imshow(
    "2. Respuesta Harris (Escala de grises)",
    esquinas_norm
)
cv2.imshow(
    "3. Esquinas detectadas",
    resultado
)

# Guardar los resultados en disco
cv2.imwrite(
    "../resultados/ejemplo2_harris_gris.jpg",
    esquinas_norm
)
cv2.imwrite(
    "../resultados/ejemplo2_esquinas.jpg",
    resultado
)

# Contar esquinas aproximadas
cantidad_esquinas = np.sum(
    esquinas > umbral
)

print("Deteccion de esquinas terminada.")
print("Cantidad aproximada de puntos detectados:",
      cantidad_esquinas)
print("Resultados guardados en la carpeta /resultados/")

# Esperar una tecla
cv2.waitKey(0)

# Cerrar ventanas
cv2.destroyAllWindows()
print("marco morquecho 1440")