import cv2
import numpy as np


class FusionMethods:
    @staticmethod
    def mertens_fusion(images, exposure_times=None):
        merge_mertens = cv2.createMergeMertens()
        fusion = merge_mertens.process(images)
        fusion = np.clip(fusion * 255, 0, 255).astype(np.uint8)
        return fusion

    @staticmethod
    def laplacian_pyramid_fusion(images, exposure_times=None, levels=None):
        if len(images) < 2:
            return images[0]

        if levels is None:
            levels = min(4, int(np.log2(min(images[0].shape[:2]))) - 2)
            levels = max(2, levels)

        float_images = [img.astype(np.float32) / 255.0 for img in images]
        weights = FusionMethods._calculate_weights(float_images, exposure_times)
        weight_sum = np.sum(weights, axis=0)
        weight_sum[weight_sum == 0] = 1e-10
        weights = weights / weight_sum[np.newaxis, :, :, np.newaxis]

        gaussian_pyramids = []
        laplacian_pyramids = []
        weight_pyramids = []

        for i in range(len(float_images)):
            gp = FusionMethods._build_gaussian_pyramid(float_images[i], levels)
            gaussian_pyramids.append(gp)
            lp = FusionMethods._build_laplacian_pyramid(gp)
            laplacian_pyramids.append(lp)
            wp = FusionMethods._build_gaussian_pyramid(weights[i], levels)
            weight_pyramids.append(wp)

        fused_laplacian = []
        for level in range(levels + 1):
            fused_level = np.zeros_like(laplacian_pyramids[0][level])
            for i in range(len(float_images)):
                fused_level += laplacian_pyramids[i][level] * weight_pyramids[i][level]
            fused_laplacian.append(fused_level)

        fused_image = FusionMethods._reconstruct_from_laplacian(fused_laplacian)
        fused_image = np.clip(fused_image * 255, 0, 255).astype(np.uint8)

        return fused_image

    @staticmethod
    def _calculate_weights(images, exposure_times=None):
        weights = []
        for i, img in enumerate(images):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            contrast = cv2.Laplacian(gray, cv2.CV_32F, ksize=5)
            contrast = cv2.convertScaleAbs(contrast)
            contrast = contrast.astype(np.float32) / 255.0
            contrast = cv2.GaussianBlur(contrast, (5, 5), 0)

            saturation = np.std(img, axis=2)
            saturation = saturation / (saturation.max() + 1e-10)

            well_exposed = np.ones_like(gray)
            for c in range(3):
                channel = img[:, :, c]
                gauss = np.exp(-12.5 * np.power(channel - 0.5, 2))
                well_exposed *= gauss

            weight = (contrast + 0.5) * (saturation + 0.5) * (well_exposed + 0.5)
            weight = weight[:, :, np.newaxis]
            weight = np.repeat(weight, 3, axis=2)
            weight += 1e-10
            weights.append(weight)

        return np.array(weights)

    @staticmethod
    def _build_gaussian_pyramid(img, levels):
        pyramid = [img.copy()]
        current = img.copy()
        for _ in range(levels):
            current = cv2.pyrDown(current)
            pyramid.append(current)
        return pyramid

    @staticmethod
    def _build_laplacian_pyramid(gaussian_pyramid):
        laplacian = []
        levels = len(gaussian_pyramid)
        for i in range(levels - 1):
            size = (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0])
            upsampled = cv2.pyrUp(gaussian_pyramid[i + 1], dstsize=size)
            laplacian.append(gaussian_pyramid[i] - upsampled)
        laplacian.append(gaussian_pyramid[-1].copy())
        return laplacian

    @staticmethod
    def _reconstruct_from_laplacian(laplacian_pyramid):
        current = laplacian_pyramid[-1].copy()
        for i in range(len(laplacian_pyramid) - 2, -1, -1):
            size = (laplacian_pyramid[i].shape[1], laplacian_pyramid[i].shape[0])
            current = cv2.pyrUp(current, dstsize=size)
            current = current + laplacian_pyramid[i]
        return current
