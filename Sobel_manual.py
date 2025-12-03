import cv2
import numpy as np
import matplotlib.pyplot as plt

def sobel_manual(imagem_caminho):
    
    # 1. Carregar imagem em escala de cinza
    img = cv2.imread(imagem_caminho, 0)
    img_original = cv2.imread(imagem_caminho)  # Para visualização final em cor
    if img is None:
        print("Erro ao carregar imagem")
        return

    # 2. Definição Manual dos Kernels (A Máscara de Convolução)
    # Sobel X: Derivada na horizontal (detecta bordas verticais)
    # Note a coluna da esquerda negativa e a da direita positiva.
    kernel_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])

    # Sobel Y: Derivada na vertical (detecta bordas horizontais)
    # Note a linha de cima negativa e a de baixo positiva.
    kernel_y = np.array([[-1, -2, -1],
                         [ 0,  0,  0],
                         [ 1,  2,  1]])

    # 3. Aplicação da Convolução
    # filter2D realiza a operação de deslizar a matriz sobre a imagem.
    # Usamos float64 para não perder valores negativos (que indicam direção da borda).
    grad_x = cv2.filter2D(img, cv2.CV_64F, kernel_x)
    grad_y = cv2.filter2D(img, cv2.CV_64F, kernel_y)

    # 4. Cálculo da Magnitude (Teorema de Pitágoras)
    # G = raiz_quadrada(Gx^2 + Gy^2)
    magnitude = np.sqrt(np.square(grad_x) + np.square(grad_y))

    # 5. Cálculo da Direção (Arco Tangente)
    # theta = arctan(Gy / Gx)
    # np.arctan2 lida melhor com divisões por zero do que arctan simples
    direcao = np.arctan2(grad_y, grad_x)

    # 6. Normalização para visualização (0 a 255)
    # A magnitude pode ser muito alta, então normalizamos para caber em 8 bits.
    magnitude_vis = np.uint8(255 * magnitude / np.max(magnitude))
    
    # Visualizar a direção requer um mapa de cores, pois os valores são radianos (-pi a pi)
    # Vamos converter radianos para graus para facilitar o entendimento (opcional)
    direcao_graus = np.degrees(direcao)

    # --- Plotagem ---
    plt.figure(figsize=(12, 8))

    # Imagem Original
    plt.subplot(2, 2, 1)
    plt.imshow(img_original[..., ::-1])  # Convert BGR para RGB
    plt.title('Imagem Original')
    plt.axis('off')

    # Gradiente X (Bordas Verticais)
    plt.subplot(2, 2, 2)
    # Usamos valor absoluto aqui apenas para visualização
    plt.imshow(np.abs(grad_x), cmap='gray')
    plt.title('Gradiente X (Cálculo Manual)')
    plt.axis('off')

    # Gradiente Y (Bordas Horizontais)
    plt.subplot(2, 2, 3)
    plt.imshow(np.abs(grad_y), cmap='gray')
    plt.title('Gradiente Y (Cálculo Manual)')
    plt.axis('off')

    # Magnitude Final
    plt.subplot(2, 2, 4)
    plt.imshow(magnitude_vis, cmap='gray')
    plt.title('Magnitude do Gradiente (Pitágoras)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return grad_x, grad_y, magnitude, direcao

# Execute a função (substitua pelo nome da sua imagem)
# Se não tiver imagem, crie uma imagem de teste simples:
# img_teste = np.zeros((100, 100), dtype=np.uint8)
# cv2.rectangle(img_teste, (25, 25), (75, 75), 255, -1)
# cv2.imwrite('teste_sobel.png', img_teste)

sobel_manual('src/teste_sobel.png')