
import numpy as np
import tifffile
from skimage import exposure
from colorcet import glasbey_dark
import spam.label
from scipy.ndimage import generic_gradient_magnitude, generate_binary_structure
import tkinter as tk
from tkinter import filedialog

def hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i: i +2], 16) for i in (0, 2, 4))

def create_contour_image_fast(labelled, grey_16):
    """
    An optimized, faster version for creating contour images.
    This version strictly ensures that colors of adjacent particles are different.
    """
    print("Starting optimized version of contour generation...")

    # --- 1. 准备工作和全局边界检测 ---
    # 使用百分位数来确定输入范围，以增强对比度并忽略异常值
    p2, p98 = np.percentile(grey_16, (2, 98))
    grey = exposure.rescale_intensity(grey_16, in_range=(p2, p98), out_range='uint8').astype(np.uint8)
    result = np.stack([grey, grey, grey], axis=-1)
    labels = np.unique(labelled)
    labels = labels[labels != 0]

    print("正在进行全局边界检测 (兼容旧版SciPy)...")
    from scipy.ndimage import grey_dilation, grey_erosion, generate_binary_structure
    structure = generate_binary_structure(3, 1)  # 6邻域结构元素

    # 使用膨胀和腐蚀来计算形态学梯度
    dilated = grey_dilation(labelled, footprint=structure)
    eroded = grey_erosion(labelled, footprint=structure)

    # 梯度不为0的地方就是边界
    all_boundaries_mask = (dilated != eroded)

    # --- 2. 高效构建邻接表 ---
    print("正在高效构建邻接表...")
    adj_dict = {lb: set() for lb in labels}
    boundary_coords = np.where(all_boundaries_mask)

    for i in range(labelled.ndim):
        shifted_pos = np.roll(labelled, -1, axis=i)
        adj_pairs_pos = labelled[all_boundaries_mask] != shifted_pos[all_boundaries_mask]
        l1_pos = labelled[all_boundaries_mask][adj_pairs_pos]
        l2_pos = shifted_pos[all_boundaries_mask][adj_pairs_pos]
        valid_pairs_pos = (l1_pos != 0) & (l2_pos != 0)

        for l1, l2 in zip(l1_pos[valid_pairs_pos], l2_pos[valid_pairs_pos]):
            adj_dict[l1].add(l2)
            adj_dict[l2].add(l1)

    # --- 3. 图着色 (核心逻辑：保证相邻颜色不同) ---
    print("正在进行图着色...")
    color_map = {}
    cmap_len = len(glasbey_dark)
    # 按邻居多到少排序，这是一种有效的启发式策略
    sorted_labels = sorted(labels, key=lambda x: len(adj_dict[x]), reverse=True)

    for lb in sorted_labels:
        # 关键步骤1: 找出所有邻居已经使用了的颜色
        used_colors = {color_map[neighbor] for neighbor in adj_dict[lb] if neighbor in color_map}

        # 关键步骤2: 找到第一个邻居没用过的颜色并分配
        for color_index in range(cmap_len):
            if color_index not in used_colors:
                color_map[lb] = color_index
                break
        else:
            # 如果循环正常结束（没有break），说明所有颜色都用完了
            raise ValueError(f"Label {lb} 无法被着色，颜色不足。请提供一个更大的调色板。")

    # --- 4. 向量化着色 ---
    print("正在进行向量化着色...")
    contour = np.zeros_like(labelled)
    contour_colored = np.zeros_like(result)

    max_label = labels.max()
    color_lut = np.zeros((max_label + 1, 3), dtype=np.uint8)
    rgb_colors = [hex_to_rgb(c) for c in glasbey_dark]

    for lb, color_idx in color_map.items():
        color_lut[lb] = rgb_colors[color_idx % cmap_len]

    boundary_labels = labelled[all_boundaries_mask]
    colors_for_boundaries = color_lut[boundary_labels]

    result[all_boundaries_mask] = colors_for_boundaries
    contour[all_boundaries_mask] = boundary_labels
    contour_colored[all_boundaries_mask] = colors_for_boundaries

    print("处理完成！")
    return contour, contour_colored, result


if __name__ == "__main__":
    print("Glasbey 调色板中的颜色数量:", len(glasbey_dark))

    root = tk.Tk()
    root.withdraw()

    labelled_path = filedialog.askopenfilename(
        title="选择labelled文件",
        filetypes=[("TIFF文件", "*.tif")]
    )
    if not labelled_path:
        print("未选择labelled文件，程序退出。")
        exit()

    grey_16_path = filedialog.askopenfilename(
        title="选择grey_16文件",
        filetypes=[("TIFF文件", "*.tif")]
    )
    if not grey_16_path:
        print("未选择grey_16文件，程序退出。")
        exit()

    print("正在读取文件...")
    labelled = tifffile.imread(labelled_path)
    grey_16 = tifffile.imread(grey_16_path)

    print("正在调用优化后的函数...")
    contour, contour_colored, result = create_contour_image_fast(labelled, grey_16)

    print("正在保存结果...")
    output_folder = "C:\\Users\\11388\\Desktop\\scientific\\scripts\\archive\\data\\"
    tifffile.imwrite(output_folder + "result.tif", result, photometric='rgb')
    tifffile.imwrite(output_folder + "contour.tif", contour)
    tifffile.imwrite(output_folder + "contour_colored.tif", contour_colored, photometric='rgb')

    print(f"结果已保存至 {output_folder}")
