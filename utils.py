import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(path):
    # Wczytanie obrazu i konwersja do float w zakresie 0-1 
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Nie znaleziono pliku: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0

def save_or_show(image, title="Wynik"):
    # Obrazy mogą mieć wartości poza zakresem [0, 1], musimy je przyciąć [cite: 9]
    image_clipped = np.clip(image, 0, 1)
    plt.imshow(image_clipped, cmap='gray' if len(image_clipped.shape) == 2 else None)
    plt.title(title)
    plt.axis('off')
    plt.show()