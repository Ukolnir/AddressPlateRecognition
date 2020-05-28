import cv2
import matplotlib.pyplot as plt
import numpy as np

def show(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.show()

def resize_image(image, newSize):
    frame = cv2.resize(image, newSize)
    (H, W) = image.shape[:2]
    rW = W / float(newSize[0])
    rH = H / float(newSize[1])
    return (frame, rW, rH)

def apply_clahe(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(gray)
    res = np.zeros_like(image)
    res[:,:,0] = cl1
    res[:,:,1] = cl1
    res[:,:,2] = cl1
    return res

def crop_by_boxes(image, boxes):
    images = []
    for box in boxes:
        (startX, startY, endX, endY) = box
        crop = image[startY:endY, startX:endX].copy()
        images.append(crop)
    return images