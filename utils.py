import cv2
import numpy as np
import torch


def preprocess(img):
    img = cv2.resize(img, (256,256))
    img = img / 255.0
    img = torch.tensor(img).permute(2,0,1).unsqueeze(0).float()
    return img


def postprocess(tensor):
    img = tensor.squeeze().permute(1,2,0).numpy()
    img = np.clip(img*255,0,255).astype(np.uint8)
    return img


def estimate_params(img):
    mean = np.mean(img)
    var  = np.var(img)

    clip  = 2.0 + (var / 5000)
    gamma = 1.0 + (0.5 - mean/255)

    return clip, gamma


def white_balance(img):
    result = img.copy().astype(np.float32)

    for i in range(3):
        low  = np.percentile(result[:,:,i], 1)
        high = np.percentile(result[:,:,i], 99)

        result[:,:,i] = (result[:,:,i]-low)/(high-low)*255

    return np.clip(result,0,255).astype(np.uint8)


def lab_correction(img, clip):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    l,a,b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip)
    l = clahe.apply(l)

    lab = cv2.merge((l,a,b))

    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def visibility_restore(img, gamma):
    img = img/255.0
    img = np.power(img, gamma)

    return np.clip(img*255,0,255).astype(np.uint8)


def fuse(img1, img2):
    return cv2.addWeighted(img1,0.5,img2,0.5,0)
