import numpy as np
import cv2
from kernels import SOBEL_X, SOBEL_Y, LAPLACE, GAUSSIAN, SHARPEN
from convolution import apply_convolution
from utils import load_image, save_or_show

def run_edge_detection(img_gray):
    print("Przetwarzanie: Wykrywanie krawędzi...")
    # Porównanie filtrów dla wymagania 4.0 [cite: 62]
    sobel_x = apply_convolution(img_gray, SOBEL_X) # [cite: 8, 10]
    sobel_y = apply_convolution(img_gray, SOBEL_Y) # [cite: 8, 10]
    laplace = apply_convolution(img_gray, LAPLACE) # [cite: 8, 10]
    
    save_or_show(sobel_x, "Sobel X")
    save_or_show(sobel_y, "Sobel Y")
    save_or_show(laplace, "Operator Laplace'a")

def run_blur_and_sharpen(img_rgb):
    print("Przetwarzanie: Rozmywanie i Wyostrzanie...")
    # Porównanie filtrów dla wymagania 4.0 [cite: 62]
    blurred = apply_convolution(img_rgb, GAUSSIAN) # [cite: 15, 18]
    sharpened = apply_convolution(img_rgb, SHARPEN) # [cite: 22, 24]
    
    save_or_show(blurred, "Rozmycie Gaussowskie")
    save_or_show(sharpened, "Wyostrzanie")

def run_bayer_demosaicking(img_rgb):
    print("Przetwarzanie: Demozaikowanie Bayera...")
    # 1. Tworzenie mozaiki (symulacja sensora) [cite: 27, 28]
    # Maska Bayera: [G R / B G] [cite: 36]
    h, w, _ = img_rgb.shape
    mosaic = np.zeros((h, w), dtype=np.float32)
    
    # Rozłożenie kolorów zgodnie ze wzorem Bayera [cite: 36]
    mosaic[0::2, 1::2] = img_rgb[0::2, 1::2, 0] # Red
    mosaic[0::2, 0::2] = img_rgb[0::2, 0::2, 1] # Green 1
    mosaic[1::2, 1::2] = img_rgb[1::2, 1::2, 1] # Green 2
    mosaic[1::2, 0::2] = img_rgb[1::2, 0::2, 2] # Blue
    
    # 2. Rekonstrukcja przez konwolucję [cite: 48, 61]
    # Jądra uśredniające (2x2) z odpowiednim wzmocnieniem [cite: 51, 53]
    kernel_recon = np.ones((2, 2), dtype=np.float32)
    
    # Separacja kanałów z mozaiki
    r_mos = np.zeros_like(mosaic); r_mos[0::2, 1::2] = mosaic[0::2, 1::2]
    g_mos = np.zeros_like(mosaic); g_mos[0::2, 0::2] = mosaic[0::2, 0::2]; g_mos[1::2, 1::2] = mosaic[1::2, 1::2]
    b_mos = np.zeros_like(mosaic); b_mos[1::2, 0::2] = mosaic[1::2, 0::2]
    
    # Konwolucja ze wzmocnieniem: R/B * 4, G * 2 
    r_recon = apply_convolution(r_mos, kernel_recon) * 4
    g_recon = apply_convolution(g_mos, kernel_recon) * 2
    b_recon = apply_convolution(b_mos, kernel_recon) * 4
    
    # Składanie obrazu RGB
    reconstructed = np.stack([r_recon, g_recon, b_recon], axis=2)
    save_or_show(reconstructed, "Zrekonstruowany obraz (Bayer)")

if __name__ == "__main__":
    # Ścieżka do Twojego pliku testowego
    input_path = "images/test.jpg" 
    
    try:
        image = load_image(input_path)
        img_gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        
        # Realizacja zadań
        run_edge_detection(img_gray)
        run_blur_and_sharpen(image)
        run_bayer_demosaicking(image)
        
    except FileNotFoundError as e:
        print(e)