#!/usr/bin/env python3
import cv2
import depthai as dai
import numpy as np

# ===== CONFIGURAÇÕES =====
extendedDisparity = False   # Ativar se quiser enxergar mais perto (~20 cm)
subpixel = True             # Maior precisão
lrCheck = True              # Corrige oclusões
enableRectified = True

# ===== PIPELINE =====
pipeline = dai.Pipeline()

# Câmeras mono (preto e branco)
left = pipeline.create(dai.node.MonoCamera)
right = pipeline.create(dai.node.MonoCamera)

stereo = pipeline.create(dai.node.StereoDepth)
xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutDepth.setStreamName("disparity")

# ===== PROPRIEDADES =====
left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)

stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
stereo.setLeftRightCheck(lrCheck)
stereo.setExtendedDisparity(extendedDisparity)
stereo.setSubpixel(subpixel)

# ===== LIGAÇÕES =====
left.out.link(stereo.left)
right.out.link(stereo.right)

if enableRectified:
    xoutRectL = pipeline.create(dai.node.XLinkOut)
    xoutRectR = pipeline.create(dai.node.XLinkOut)
    xoutRectL.setStreamName("rectifiedLeft")
    xoutRectR.setStreamName("rectifiedRight")
    stereo.rectifiedLeft.link(xoutRectL.input)
    stereo.rectifiedRight.link(xoutRectR.input)

stereo.disparity.link(xoutDepth.input)

# ===== EXECUÇÃO =====
with dai.Device(pipeline) as device:
    print("\n[INFO] Pressione Q para sair.\n")

    maxDisp = stereo.initialConfig.getMaxDisparity()

    while True:
        queueNames = device.getQueueEvents()
        for q in queueNames:
            message = device.getOutputQueue(q).get()
            if isinstance(message, dai.ImgFrame):
                frame = message.getCvFrame()

                if 'disparity' in q:
                    disp = (frame * (255.0 / maxDisp)).astype(np.uint8)

                    # Normalização adaptativa
                    disp = cv2.equalizeHist(disp)

                    # Suavização e colormap para realçar formas
                    disp_color = cv2.applyColorMap(disp, cv2.COLORMAP_TURBO)
                    disp_color = cv2.bilateralFilter(disp_color, 7, 50, 50)

                    cv2.imshow("Disparidade (Melhorada)", disp_color)
                else:
                    cv2.imshow(q, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
