import cv2
import numpy as np


class Evaluator:
    @staticmethod
    def calculate_ssim(img1, img2, window_size=11, K1=0.01, K2=0.03):
        if len(img1.shape) == 3 and len(img2.shape) == 3:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = img1
            gray2 = img2

        gray1 = gray1.astype(np.float64)
        gray2 = gray2.astype(np.float64)

        C1 = (K1 * 255) ** 2
        C2 = (K2 * 255) ** 2

        mu1 = cv2.GaussianBlur(gray1, (window_size, window_size), 1.5)
        mu2 = cv2.GaussianBlur(gray2, (window_size, window_size), 1.5)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cv2.GaussianBlur(gray1 ** 2, (window_size, window_size), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(gray2 ** 2, (window_size, window_size), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(gray1 * gray2, (window_size, window_size), 1.5) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        ssim_value = np.mean(ssim_map)
        
        return ssim_value

    @staticmethod
    def check_difference(img1, img2, threshold=0.9):
        ssim = Evaluator.calculate_ssim(img1, img2)
        print(f"两张融合结果图的结构相似度 (SSIM): {ssim:.4f}")
        
        if ssim < threshold:
            print(f"警告: SSIM值 ({ssim:.4f}) 低于阈值 ({threshold})，两种方法差异显著")
            return False
        else:
            print(f"SSIM值 ({ssim:.4f}) 高于阈值 ({threshold})，两种方法差异较小")
            return True
