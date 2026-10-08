# marco morquecho 1440
# (EJEMPLO 2 — Detección de esquinas con Harris) + experimentar
import cv2
import numpy as np

# Cargar imagen desde la carpeta imagenes/
imagen = cv2.imread("../imagenes/avestruz.jpg")

# Verificar que la imagen exista
if imagen is None:
    print("Error: no se pudo cargar la imagen.")
    exit()

# Convertir a escala de grises
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Convertir a tipo float32
gris_float = np.float32(gris)

# Detectar esquinas mediante Harris
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

# Crear copias
resultado1 = imagen.copy()
resultado2 = imagen.copy()

# Umbral para identificar esquinas
umbral1 = 0.01 * esquinas.max()
umbral2 = 0.05 * esquinas.max()

# Marcar esquinas
resultado1[esquinas > umbral1] = [0, 0, 255]
resultado2[esquinas > umbral2] = [0, 0, 255]

# Mostrar resultados
cv2.imshow(
    "Imagen original",
    imagen
)

cv2.imshow(
    "Esquinas detectadas (Mas puntos)",
    resultado1
)

cv2.imshow(
    "Esquinas detectadas (Menos puntos)",
    resultado2
)

# Guardar resultados en la carpeta resultados/
cv2.imwrite(
    "../resultados/avestruz_esquinas1.jpg",
    resultado1
)

cv2.imwrite(
    "../resultados/avestruz_esquinas2.jpg",
    resultado2
)

# Contar esquinas aproximadas
cantidad_esquinas1 = np.sum(
    esquinas > umbral1
)

cantidad_esquinas2 = np.sum(
    esquinas > umbral2
)

print("Deteccion de esquinas terminada.")
print("Cantidad aproximada de puntos detectados (umbral 0.01):",
      cantidad_esquinas1)
print("Cantidad aproximada de puntos detectados (umbral 0.05):",
      cantidad_esquinas2)

print("Resultado guardado en:")
print("../resultados/avestruz_esquinas1.jpg")
print("../resultados/avestruz_esquinas2.jpg")

# Esperar una tecla
cv2.waitKey(0)

# Cerrar ventanas
cv2.destroyAllWindows()
print("marco morquecho 1440")