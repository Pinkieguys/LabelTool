"""
串行，用于处理小图片
处理大图片的并行版本在实验室电脑上
"""

import numpy as np
import tifffile
from skimage.morphology import binary_erosion, ball
from skimage import exposure
from scipy.ndimage import binary_dilation
from colorcet import glasbey_dark
import spam.label
from scipy.ndimage import convolve
def hex_to_rgb(h):
    h = h.lstrip('#')
    return [int(h[i:i+2], 16) for i in (0, 2, 4)]

def is_boundary(mask):
    """
    判断边界：如果一个体素的6邻域中存在非目标体素，则它是边界。
    """
    # 定义6邻域的卷积核
    kernel = np.zeros((3, 3, 3), dtype=int)
    kernel[1, 1, 0] = 1  # 前
    kernel[1, 1, 2] = 1  # 后
    kernel[1, 0, 1] = 1  # 上
    kernel[1, 2, 1] = 1  # 下
    kernel[0, 1, 1] = 1  # 左
    kernel[2, 1, 1] = 1  # 右

    # 计算6邻域中目标体素的数量
    neighbor_count = convolve(mask.astype(int), kernel, mode='constant', cval=0)

    # 如果某个体素的6邻域中存在非目标体素，则它是边界
    boundary = (mask & (neighbor_count < 6))
    return boundary

def create_contour_image(labelled, grey_16, padding = 15):

    grey = exposure.rescale_intensity(grey_16, in_range='image', out_range='uint8').astype(np.uint8)
    result = np.stack([grey, grey, grey], axis=-1)
    contour_colored = np.zeros_like(result)
    contour = np.zeros_like(labelled)

    labels = np.unique(labelled)
    labels = labels[labels != 0]
    boundingBoxes = spam.label.boundingBoxes(labelled)


    # 1. 构建邻接表
    adj_dict = {lb: set() for lb in labels}
    for lb in labels:
        startZ = max(0, boundingBoxes[lb, 0] - padding)
        stopZ  = min(labelled.shape[0], boundingBoxes[lb, 1] + padding)
        startY = max(0, boundingBoxes[lb, 2] - padding)
        stopY  = min(labelled.shape[1], boundingBoxes[lb, 3] + padding)
        startX = max(0, boundingBoxes[lb, 4] - padding)
        stopX  = min(labelled.shape[2], boundingBoxes[lb, 5] + padding)
        neighbors = np.unique(labelled[startZ:stopZ, startY:stopY, startX:stopX])
        neighbors = neighbors[(neighbors != 0) & (neighbors != lb)]
        for nb in neighbors:
            adj_dict[lb].add(nb)
            adj_dict[nb].add(lb)

    # 2. 图着色
    color_map = {}
    cmap_len = len(glasbey_dark)
    # 按邻居多到少顺序
    sorted_labels = sorted(labels, key=lambda x: len(adj_dict[x]), reverse=True)
    for lb in sorted_labels:
        used = set(color_map[n] for n in adj_dict[lb] if n in color_map)
        for idx in range(cmap_len):
            if idx not in used:
                color_map[lb] = idx
                #print(f"Label {lb} colored with {color_map[lb]}")
                break
        else:
            print(f"Label {lb} has no color")
            raise ValueError("Too many labels")


    boundingBoxes = spam.label.boundingBoxes(labelled)
    # 3. 给每个颗粒边界上色
    for lb in labels:
        # 获取边界框坐标
        startZ = max(0, boundingBoxes[lb, 0])
        stopZ = min(labelled.shape[0], boundingBoxes[lb, 1])
        startY = max(0, boundingBoxes[lb, 2])
        stopY = min(labelled.shape[1], boundingBoxes[lb, 3])
        startX = max(0, boundingBoxes[lb, 4])
        stopX = min(labelled.shape[2], boundingBoxes[lb, 5])

        # 提取子区域
        labelled_sub = labelled[startZ:stopZ, startY:stopY, startX:stopX]
        mask = (labelled_sub == lb)
        boundary = is_boundary(mask)

        # 获取边界点在子区域中的坐标
        boundary_indices = np.where(boundary)

        if len(boundary_indices[0]) > 0:  # 确保有边界点
            # 转换到全局坐标
            global_z = boundary_indices[0] + startZ
            global_y = boundary_indices[1] + startY
            global_x = boundary_indices[2] + startX

            # 设置颜色
            c = hex_to_rgb(glasbey_dark[color_map[lb] % cmap_len])

            # 直接在全局坐标系中上色
            result[global_z, global_y, global_x] = c
            contour[global_z, global_y, global_x] = lb
            contour_colored[global_z, global_y, global_x] = c

    return contour, contour_colored, result

if __name__ == "__main__":
    import tkinter as tk
    from tkinter import filedialog

    print("label")

    """
    #labelled = tifffile.imread(r'D:\WorkSpace\scientific\scripts\archive\S60试验\S60-200-SWF1bin1-0-labelled.tif')
    #grey_16 = tifffile.imread(r'D:\WorkSpace\scientific\scripts\archive\S60试验\S60-200-.tif')
    labelled = tifffile.imread(r"D:\WorkSpace\scientific\scripts\archive\data\label.tif")
    grey_16 = tifffile.imread(r"D:\WorkSpace\scientific\scripts\archive\data\grey.tif")
    """
    print(len(glasbey_dark))

    root = tk.Tk()
    root.withdraw()

    labelled_path = filedialog.askopenfilename(
        title="选择labelled文件",
        filetypes=[("TIFF文件", "*.tif")]
    )
    grey_16_path = filedialog.askopenfilename(
        title="选择grey_16文件",
        filetypes=[("TIFF文件", "*.tif")]
    )

    labelled = tifffile.imread(labelled_path)
    grey_16 = tifffile.imread(grey_16_path)
    contour,contour_colored,result = create_contour_image(labelled, grey_16)
    tifffile.imwrite("C:\\Users\\11388\\Desktop\\scientific\\scripts\\archive\\data\\result.tif", result, photometric='rgb')
    tifffile.imwrite("C:\\Users\\11388\\Desktop\\scientific\\scripts\\archive\\data\\contour.tif", contour)
    tifffile.imwrite("C:\\Users\\11388\\Desktop\\scientific\\scripts\\archive\\data\\contour_colored.tif", contour_colored, photometric='rgb')