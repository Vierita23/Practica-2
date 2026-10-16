import cv2
import matplotlib.pyplot as plt


def mostrar_segmentacion(ruta_imagen):
    # separa el objeto del fondo para ver la mascara
    img = cv2.imread(ruta_imagen)
    if img is None:
        print("No se pudo abrir:", ruta_imagen)
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(thresh, cmap="gray")
    plt.title("Segmentacion")
    plt.axis("off")
    plt.show()
