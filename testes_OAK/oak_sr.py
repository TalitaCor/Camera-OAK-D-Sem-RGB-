import cv2
import depthai as dai
import numpy as np

# --------------------------------------------------------------
# CONFIGURAÇÃO DO PIPELINE DepthAI
# --------------------------------------------------------------
pipeline = dai.Pipeline()

# Câmeras mono esquerda e direita
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)

# Nó de profundidade (StereoDepth)
depth = pipeline.create(dai.node.StereoDepth)

# Saída para o host
xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("disparity")

# --------------------------------------------------------------
# CONFIGURAÇÕES DAS CÂMERAS
# --------------------------------------------------------------
# Define resolução das câmeras mono (400p = 640x400)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

# Define qual é a câmera esquerda e direita fisicamente
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# --------------------------------------------------------------
# CONFIGURAÇÕES DO ALGORITMO DE PROFUNDIDADE
# --------------------------------------------------------------
# Usa perfil de alta densidade → mais detalhes em superfícies
depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)

# Filtros e refinamentos:
depth.initialConfig.setConfidenceThreshold(220)
# ↑ quanto maior, menos ruído (mas também menos pixels válidos)
# 180 a 230 é o intervalo ideal

depth.setLeftRightCheck(True)
# ↑ Corrige buracos e inconsistências de disparidade

depth.setSubpixel(True)
# ↑ Melhora a precisão de profundidade (em milímetros)
#   Pode gerar mais ruído — desative se estiver muito granulado

depth.setExtendedDisparity(False)
# ↑ Ativa profundidade muito próxima (<50 cm). 
#   Desative se houver muito ruído. Ative se o objeto estiver colado à câmera.

# Filtragem mediana (suaviza ruído fino)
depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)

# --------------------------------------------------------------
# CONEXÕES ENTRE NÓS DO PIPELINE
# --------------------------------------------------------------
monoLeft.out.link(depth.left)
monoRight.out.link(depth.right)
depth.disparity.link(xout.input)

# --------------------------------------------------------------
# EXECUÇÃO DO PIPELINE
# --------------------------------------------------------------
with dai.Device(pipeline) as device:
    print("Iniciando... Pressione 'q' para sair.")

    q = device.getOutputQueue(name="disparity", maxSize=8, blocking=False)

    # Loop principal
    while True:
        inDisparity = q.get()  # recebe frame da OAK
        frame = inDisparity.getFrame()

        # Normaliza o mapa de disparidade para 0–255
        frame = (frame * (255 / depth.initialConfig.getMaxDisparity())).astype(np.uint8)

        # Aplica colormap para visualização
        disp_color = cv2.applyColorMap(frame, cv2.COLORMAP_JET)

        # Mostra o resultado
        cv2.imshow("disparity_color", disp_color)

        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
