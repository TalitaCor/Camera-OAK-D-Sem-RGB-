import cv2
import depthai as dai
import numpy as np

# === CONFIGURAÇÃO DO PIPELINE ===
pipeline = dai.Pipeline()

# As câmeras mono da OAK-D SR são CAM_B e CAM_C
left = pipeline.createMonoCamera()
right = pipeline.createMonoCamera()
left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)

# Cria o nó de profundidade
stereo = pipeline.createStereoDepth()
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
stereo.setLeftRightCheck(True)
stereo.setSubpixel(True)
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_B)  # alinha à esquerda
stereo.initialConfig.setConfidenceThreshold(200)

# Liga câmeras
left.out.link(stereo.left)
right.out.link(stereo.right)

# Saídas
xout_disp = pipeline.createXLinkOut()
xout_disp.setStreamName("disparity")
stereo.disparity.link(xout_disp.input)

# === EXECUÇÃO ===
with dai.Device(pipeline) as device:
    qDisp = device.getOutputQueue(name="disparity", maxSize=4, blocking=False)

    print("\n[INFO] Pressione Q para sair.\n")

    while True:
        inDisp = qDisp.get()
        dispFrame = inDisp.getFrame().astype(np.float32)

        # Normaliza automaticamente com base no conteúdo da cena
        min_disp = np.percentile(dispFrame, 2)
        max_disp = np.percentile(dispFrame, 98)
        disp_norm = np.clip((dispFrame - min_disp) / (max_disp - min_disp), 0, 1)
        disp_vis = cv2.applyColorMap((disp_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)

        # Suaviza ruído mantendo bordas
        disp_vis = cv2.bilateralFilter(disp_vis, 9, 75, 75)

        cv2.imshow("Disparidade (auto-range + filtro bilateral)", disp_vis)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

