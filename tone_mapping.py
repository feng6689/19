import cv2
import numpy as np


class ToneMapper:
    @staticmethod
    def drago_tone_mapping(hdr_image, gamma=2.2, saturation=1.0, bias=0.85):
        hdr_float = hdr_image.astype(np.float32) / 255.0
        
        if len(hdr_float.shape) == 3 and hdr_float.shape[2] == 3:
            hdr_lin = ToneMapper._linearize(hdr_float, gamma)
        else:
            hdr_lin = hdr_float

        drago = cv2.createTonemapDrago(gamma, saturation, bias)
        ldr = drago.process(hdr_lin)
        ldr = np.clip(ldr * 255, 0, 255).astype(np.uint8)
        
        return ldr

    @staticmethod
    def reinhard_tone_mapping(hdr_image, gamma=2.2, intensity=0.0, light_adapt=1.0, color_adapt=0.0):
        hdr_float = hdr_image.astype(np.float32) / 255.0
        
        if len(hdr_float.shape) == 3 and hdr_float.shape[2] == 3:
            hdr_lin = ToneMapper._linearize(hdr_float, gamma)
        else:
            hdr_lin = hdr_float

        reinhard = cv2.createTonemapReinhard(gamma, intensity, light_adapt, color_adapt)
        ldr = reinhard.process(hdr_lin)
        ldr = np.clip(ldr * 255, 0, 255).astype(np.uint8)
        
        return ldr

    @staticmethod
    def _linearize(img, gamma=2.2):
        return np.power(img, gamma)
