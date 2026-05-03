import cv2
import numpy as np


class ImageAligner:
    def __init__(self, max_iterations=5000, epsilon=1e-10):
        self.max_iterations = max_iterations
        self.epsilon = epsilon

    def align_images(self, images):
        if len(images) < 2:
            return images, True

        aligned_images = [images[0].copy()]
        reference_gray = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
        all_aligned = True

        for i in range(1, len(images)):
            img_gray = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)

            aligned, success = self._align_with_fallback(
                reference_gray, img_gray, images[i], image_index=i+1
            )

            if not success:
                print(f"警告: 第 {i+1} 张图像对齐失败，将使用原始图像")
                aligned_images.append(images[i].copy())
                all_aligned = False
            else:
                aligned_images.append(aligned)

        if not all_aligned:
            print("警告: 部分图像对齐失败，已使用原始图像进行融合")

        return aligned_images, True

    def _align_with_fallback(self, reference_gray, img_gray, img_color, image_index):
        aligned, success = self._align_ecc(reference_gray, img_gray, img_color)

        if not success:
            print(f"第 {image_index} 张图像 ECC 对齐失败，尝试特征点对齐...")
            aligned, success = self._align_feature(reference_gray, img_gray, img_color)

            if not success:
                print(f"第 {image_index} 张图像特征点对齐也失败")
                return None, False

        return aligned, True

    def _align_ecc(self, reference_gray, img_gray, img_color):
        try:
            h, w = reference_gray.shape
            warp_matrix = np.eye(2, 3, dtype=np.float32)

            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                        self.max_iterations, self.epsilon)

            result = cv2.findTransformECC(
                reference_gray, img_gray, warp_matrix,
                cv2.MOTION_AFFINE, criteria, None, 5
            )

            if isinstance(result, tuple):
                warp_matrix = result[0]
            else:
                warp_matrix = result

            if warp_matrix is None:
                return None, False

            warp_matrix = np.array(warp_matrix, dtype=np.float32)

            aligned = cv2.warpAffine(img_color, warp_matrix, (w, h),
                                      flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)

            return aligned, True
        except cv2.error as e:
            print(f"ECC对齐错误: {e}")
            return None, False
        except Exception as e:
            print(f"ECC对齐未知错误: {e}")
            return None, False

    def _align_feature(self, reference_gray, img_gray, img_color):
        try:
            h, w = reference_gray.shape

            orb = cv2.ORB_create(5000)
            kp1, des1 = orb.detectAndCompute(reference_gray, None)
            kp2, des2 = orb.detectAndCompute(img_gray, None)

            if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
                return None, False

            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)

            if len(matches) < 4:
                return None, False

            matches = sorted(matches, key=lambda x: x.distance)
            good_matches = matches[:int(len(matches) * 0.15)]

            if len(good_matches) < 4:
                return None, False

            src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if M is None:
                return None, False

            aligned = cv2.warpPerspective(img_color, M, (w, h))

            return aligned, True
        except Exception as e:
            print(f"特征点对齐错误: {e}")
            return None, False
