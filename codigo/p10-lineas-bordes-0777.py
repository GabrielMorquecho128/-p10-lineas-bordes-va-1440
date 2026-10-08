# marco morquecho 1440
# (EJEMPLO 2 — Detección de esquinas con Harris) + experimentar
import cv2
import numpy as np

# 1. Cargar imagen de la avestruz
imagen = cv2.imread("../imagenes/avestruz.jpg")

# Verificar que la imagen exista
if imagen is None:
    print("Error: no se pudo cargar la imagen.")
    exit()

# Convertir a escala de grises y a float32
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
gris_float = np.float32(gris)

# 2. Detectar esquinas mediante Harris
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

# --- DEFINIR LOS UMBRALES PARA REPARTIR LOS PUNTOS ROJOS ---
umbral_bajo = 0.01 * esquinas.max()   # Umbral bajo -> Más puntos rojos
umbral_alto = 0.05 * esquinas.max()   # Umbral alto -> Menos puntos rojos

# Crear copias para no modificar la imagen original
resultado_mas_puntos = imagen.copy()
resultado_menos_puntos = imagen.copy()

# Marcar esquinas en rojo sobre las copias
resultado_mas_puntos[esquinas > umbral_bajo] = [0, 0, 255]
resultado_menos_puntos[esquinas > umbral_alto] = [0, 0, 255]

# 3. Mostrar las 3 ventanas
cv2.imshow(
    "1. Imagen original",
    imagen
)
cv2.imshow(
    "2. Esquinas detectadas (Mas puntos - Umbral 0.01)",
    resultado_mas_puntos
)
cv2.imshow(
    "3. Esquinas detectadas (Menos puntos - Umbral 0.05)",
    resultado_menos_puntos
)

# Guardar los resultados en disco
cv2.imwrite(
    "../resultados/ejemplo2_esquinas_mas_puntos.jpg",
    resultado_mas_puntos
)
cv2.imwrite(
    "../resultados/ejemplo2_esquinas_menos_puntos.jpg",
    resultado_menos_puntos
)

# Contar esquinas aproximadas
print("Deteccion de esquinas terminada 1440.")
print("Cantidad con umbral 0.01 (mas puntos) 1440:", np.sum(esquinas > umbral_bajo))
print("Cantidad con umbral 0.05 (menos puntos) 1440 :", np.sum(esquinas > umbral_alto))
print("Resultados guardados en la carpeta /resultados/")

# Esperar una tecla y cerrar ventanas
cv2.waitKey(0)
cv2.destroyAllWindows()

print("marco morquecho 1440")