import cv2
import numpy as np

# Carregar a imagem
imagem = cv2.imread('C:\PD\PD\sep_img\imagem.jpg')

# Separar os canais (B, G, R)
(b, g, r) = cv2.split(imagem)

# Criar imagens individuais para cada canal
# Vermelho: (r, 0, 0), Verde: (0, g, 0), Azul: (0, 0, b)
zeros = np.zeros(imagem.shape[:2], dtype="uint8")

vermelho = cv2.merge([zeros, zeros, r])
verde = cv2.merge([zeros, g, zeros])
azul = cv2.merge([b, zeros, zeros])

# Salvar os canais como imagens separadas
cv2.imwrite('C:\PD\PD\sep_img\imagem_separada\canal_vermelho.jpg', vermelho)
cv2.imwrite('C:\PD\PD\sep_img\imagem_separada\canal_verde.jpg', verde)
cv2.imwrite('C:\PD\PD\sep_img\imagem_separada\canal_azul.jpg', azul)

# Exibir as imagens dos canais separadamente (opcional)
cv2.imshow("C:\PD\PD\sep_img\imagem_separada\Canal Vermelho", vermelho)
cv2.imshow("C:\PD\PD\sep_img\imagem_separada\Canal Verde", verde)
cv2.imshow("C:\PD\PD\sep_img\imagem_separada\Canal Azul", azul)

# Esperar pressionamento de tecla para fechar as janelas
cv2.waitKey(0)
cv2.destroyAllWindows()
