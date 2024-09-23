import cv2
import numpy as np

# Leitura da imagem
imagem = cv2.imread('C:\PD\PD\pre_processamento_img\imagem.jpg')

# Redimensionamento (ajustar para 256x256, por exemplo)
imagem_redimensionada = cv2.resize(imagem, (256, 256))

# Conversão para escala de cinza
imagem_cinza = cv2.cvtColor(imagem_redimensionada, cv2.COLOR_BGR2GRAY)

# Normalização (padronizar os valores de pixel entre 0 e 1)
imagem_normalizada = imagem_cinza / 255.0

# Suavização (aplicando um filtro Gaussiano para reduzir ruído)
imagem_suavizada = cv2.GaussianBlur(imagem_normalizada, (5, 5), 0)

# Equalização do histograma (melhorar o contraste da imagem)
imagem_equalizada = cv2.equalizeHist((imagem_suavizada * 255).astype(np.uint8))

# Segmentação (por exemplo, usando limiarização)
_, imagem_segmentada = cv2.threshold(imagem_equalizada, 127, 255, cv2.THRESH_BINARY)

# Salvando Imagem
cv2.imwrite('C:\PD\PD\pre_processamento_img\imagens_processada\imagem.jpg', imagem)
cv2.imwrite('C:\PD\PD\pre_processamento_img\imagens_processada\imagempreprocessada.jpg', imagem_segmentada)


# Exibição do resultado
cv2.imshow('Imagem Original', imagem)
cv2.imshow('Imagem Pré-processada', imagem_segmentada)
cv2.waitKey(0)
cv2.destroyAllWindows()
