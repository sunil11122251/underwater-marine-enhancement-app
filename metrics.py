from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np
import cv2


def resize_to(src, ref):
    h, w = ref.shape[:2]
    return cv2.resize(src, (w, h), interpolation=cv2.INTER_CUBIC)


def compute_metrics(fused, final):

    fused = resize_to(fused, final)

    fused = np.clip(fused, 0, 255).astype(np.float32)
    final = np.clip(final, 0, 255).astype(np.float32)

    psnr = peak_signal_noise_ratio(fused, final, data_range=255)

    ssim = structural_similarity(
        fused, final,
        channel_axis=2,
        data_range=255
    )

    uciqe = np.mean(final) / 255
    uiqm  = np.std(final) / 255

    return psnr, ssim, uciqe, uiqm
