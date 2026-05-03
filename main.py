import cv2
import numpy as np
import os

from align import ImageAligner
from fusion_methods import FusionMethods
from tone_mapping import ToneMapper
from evaluator import Evaluator


def load_images(image_paths):
    images = []
    for path in image_paths:
        if not os.path.exists(path):
            print(f"错误: 找不到图像文件 {path}")
            return None
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"错误: 无法读取图像文件 {path}")
            return None
        images.append(img)
        print(f"成功加载: {path}，尺寸: {img.shape}")
    return images


def main():
    print("=" * 60)
    print("多曝光图像融合程序")
    print("=" * 60)

    image_files = ["1.jpg", "2.jpg", "3.jpg"]
    print("\n[1/5] 加载图像...")
    images = load_images(image_files)
    
    if images is None or len(images) < 2:
        print("错误: 需要至少2张图像进行处理")
        return

    print("\n[2/5] 图像对齐 (先尝试 ECC，失败则使用特征点对齐)...")
    aligner = ImageAligner()
    aligned_images, success = aligner.align_images(images)

    print("\n[3/5] 执行多曝光融合...")
    
    print("\n执行 Mertens 融合...")
    mertens_result = FusionMethods.mertens_fusion(aligned_images)
    cv2.imwrite("11.jpg", mertens_result)
    print("Mertens 融合结果已保存: 11.jpg")
    
    print("\n执行拉普拉斯金字塔融合...")
    laplacian_result = FusionMethods.laplacian_pyramid_fusion(aligned_images)
    cv2.imwrite("12.jpg", laplacian_result)
    print("拉普拉斯金字塔融合结果已保存: 12.jpg")

    print("\n[4/5] 计算结构相似度 (SSIM)...")
    Evaluator.check_difference(mertens_result, laplacian_result, threshold=0.9)

    print("\n[5/5] 色调映射 (使用 Drago 算法)...")
    hdr_for_tonemap = np.float32(mertens_result) / 255.0
    tonemap_result = ToneMapper.drago_tone_mapping(mertens_result)
    cv2.imwrite("13.jpg", tonemap_result)
    print("色调映射结果已保存: 13.jpg")

    print("\n" + "=" * 60)
    print("处理完成! 输出文件:")
    print("  - 11.jpg: Mertens 融合结果")
    print("  - 12.jpg: 拉普拉斯金字塔融合结果")
    print("  - 13.jpg: 色调映射增强结果")
    print("=" * 60)


if __name__ == "__main__":
    main()
