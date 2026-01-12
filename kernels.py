import numpy as np

# Wykrywanie krawędzi - Operatory Sobela 
SOBEL_X = np.array([
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]
])

SOBEL_Y = np.array([
    [1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1]
])

# Wykrywanie krawędzi - Operator Laplace'a 
LAPLACE = np.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
])

# Rozmywanie - Jądro Gaussowskie G 
GAUSSIAN = (1/16) * np.array([
    [1, 2, 1],
    [1, 4, 1],
    [1, 2, 1]
])

# Wyostrzanie - Jądro W 
SHARPEN = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])

# Bonus: Operator Prewitta (Oś X) 
PREWITT_X = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])