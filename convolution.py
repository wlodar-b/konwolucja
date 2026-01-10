import numpy as np

def apply_convolution(image, kernel):
    # Pobranie wymiarów
    img_h, img_w = image.shape[:2]
    kh, kw = kernel.shape
    
    # Obliczenie paddingu (dla zachowania rozmiaru wyjściowego) [cite: 54, 55]
    pad_h = kh // 2
    pad_w = kw // 2
    
    # Wypełnianie zerami 
    if len(image.shape) == 3: # Obraz kolorowy
        padded_img = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant', constant_values=0)
        output = np.zeros_like(image)
        channels = image.shape[2]
    else: # Obraz czarno-biały
        padded_img = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
        output = np.zeros_like(image)
        channels = 1

    # Proces konwolucji [cite: 4, 50]
    for i in range(img_h):
        for j in range(img_w):
            if channels > 1:
                for c in range(channels):
                    region = padded_img[i:i+kh, j:j+kw, c]
                    output[i, j, c] = np.sum(region * kernel)
            else:
                region = padded_img[i:i+kh, j:j+kw]
                output[i, j] = np.sum(region * kernel)
                
    return output