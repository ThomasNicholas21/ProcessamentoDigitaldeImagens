import cv2
import numpy as np

# Carregar a imagem de mamografia
imagem = cv2.imread('C:/PD/PD/analise_img/img_teste.jpg', 0)  # Carrega a imagem em escala de cinza

# Pré-processamento: Remoção de ruído com Filtro Gaussiano
imagem_suavizada = cv2.GaussianBlur(imagem, (5, 5), 0)

# Ajuste de contraste com equalização de histograma
imagem_contraste = cv2.equalizeHist(imagem_suavizada)

# Segmentação com detecção de bordas usando Canny
bordas = cv2.Canny(imagem_contraste, 100, 200)

# Destacar a área de maior intensidade (supostamente regiões anômalas)
contornos, _ = cv2.findContours(bordas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
imagem_segmentada = np.zeros_like(imagem)
cv2.drawContours(imagem_segmentada, contornos, -1, (255), 1)

# Mostrar e salva a imagem segmentada
cv2.imwrite('C:/PD/PD/analise_img/imagem_analisada/imagemsegmentada.jpg', imagem_segmentada)
cv2.imshow('Imagem Segmentada', imagem_segmentada)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Inferência básica: Se há muitos contornos (possível anomalia)
if len(contornos) > 10:  # Um número arbitrário de contornos
    print("Possível presença de câncer (análise superficial).")
else:
    print("Ausência de sinais de câncer evidentes.")
