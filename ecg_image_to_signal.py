import cv2
import numpy as np

def extract_ecg_from_image(image_path):

    img = cv2.imread(image_path)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray,(5,5),0)

    edges = cv2.Canny(blur,50,150)

    height,width = edges.shape

    signal = []

    for x in range(width):

        column = edges[:,x]

        ys = np.where(column>0)[0]

        if len(ys)>0:

            signal.append(np.mean(ys))

        else:

            signal.append(signal[-1] if signal else 0)

    signal = np.array(signal)

    signal = (signal - np.min(signal)) / (np.max(signal)-np.min(signal))

    if len(signal) > 300:
        signal = signal[:300]
    else:
        signal = np.pad(signal,(0,300-len(signal)))

    return signal