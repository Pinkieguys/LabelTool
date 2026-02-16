import sys
import numpy as np
from tifffile import imread
from PyQt5.QtWidgets import (QApplication, QMainWindow, QDockWidget, QLabel,
                             QVBoxLayout, QWidget, QSlider, QLineEdit, QPushButton, QHBoxLayout)
from PyQt5.QtCore import Qt
import pyqtgraph as pg
from pyqtgraph.graphicsItems.ImageItem import ImageItem
import spam.label
import tifffile, os, toolusing
import pyvista as pv

from . import segmentation
from . import viz_utils
import matplotlib.pyplot as plt


def manage_contact(intermediate_dir, whole_name, contactVolume, tif, contactingLabels_r, contactsTable, dilate_table,
                   labelled, local_threshold):
    npz_path = intermediate_dir / f"{whole_name}-managecontact{local_threshold}.npz"

    if os.path.exists(npz_path):
        managecontact_datas = np.load(npz_path, allow_pickle=True)
        ids = managecontact_datas['ids']
        volumes = managecontact_datas['volumes']
        greys1 = managecontact_datas['greys1']
        greys2 = managecontact_datas['greys2']
        greysori = managecontact_datas['greysori']
        surfaces = managecontact_datas['surfaces']
    else:
        idmax = contactVolume.max() + 1
        ids = []
        volumes = np.zeros(idmax)
        greys1 = np.zeros(idmax)
        greys2 = np.zeros(idmax)
        greysori = np.zeros(idmax)

        surfaces = spam.label.volumes(contactVolume)
        grey_average = np.mean(tif[contactVolume > 0])

        for lab1, lab2 in contactingLabels_r:
            lab1_contactorder = contactsTable[lab1]
            contact_id = toolusing.find_value_from_contact_order(lab1_contactorder, lab2)
            ids.append(contact_id)
            volumes[contact_id] = dilate_table[lab1][lab2] + dilate_table[lab2][lab1]
            current_grey = np.mean(tif[contactVolume == contact_id])
            greysori[contact_id] = current_grey
            greys1[contact_id] = current_grey / grey_average
            greys2[contact_id] = current_grey / np.mean(tif[(labelled == lab1) | (labelled == lab2)])

        np.savez(npz_path, ids=ids, volumes=volumes, greys1=greys1, greys2=greys2, greysori=greysori, surfaces=surfaces)

    # 取出对应位置，与ids对齐
    volumes = volumes[ids]
    greys1 = greys1[ids]
    greys2 = greys2[ids]
    greysori = greysori[ids]
    surfaces = surfaces[ids]

    return ids, volumes, greys1, greys2, greysori, surfaces


def get_contour_colors(max_id):
    fixed_colors = [
        [255, 255, 0],  # 黄色
        [255, 165, 0],  # 橙色
        [128, 0, 128],  # 紫色
        [255, 192, 203]  # 粉色
    ]
    colors = []
    for i in range(max_id):
        colors.append(fixed_colors[i % len(fixed_colors)])
    return colors


class MainWindow(QMainWindow):
    def __init__(self, gray_vol, contour_vol, label_vol, colormap, contour_colored, merge_vol, vectors=None,
                 mode='gray', vector_labels=None, reference_labelled=None, red_labels=None, segmode='None',
                 at_once=False, value=1, labels_given=None):
        super().__init__()
        self.gray_vol = gray_vol
        self.contour_vol = contour_vol
        self.label_vol = label_vol
        self.colormap = colormap
        self.contour_colored = contour_colored
        self.merge_vol = merge_vol
        self.vector3s = vectors
        self.current_z = 0
        self.selected_mask = None
        self.selected_box = None
        self.boundingBoxes = spam.label.boundingBoxes(label_vol)
        self.selected_labels = []
        self.need_to_update_box = False
        self.pad = 2
        self.render_threshold_for_grey = 80  # 百分�?
        self.mode = mode
        # mode�?label'时，默认是接触图像，默认不会一次选中多个label（即接触），渲染label时不会渲染灰度图，而是渲染构成该接触的两个颗粒
        self.vector_labels = vector_labels  # 用于label模式下的每个接触的接触向�?
        self.reference_labelled = reference_labelled  # 用于label模式下的颗粒渲染
        self.red_labels = red_labels  # 用于label模式下标红突出labels指定的接�?
        self.segmode = segmode  # 用于标记分割模式，有'fixover','fixunder',默认�?None'
        self.waiting_list_over = []  # 用于存储等待处理的过分割合并操作
        self.waiting_list_under = []  # 用于存储等待处理的欠分割分割操作
        self.waiting_mask_over = None
        self.waiting_mask_under = None
        self.current_fix_mode = 'fixover'  # 当前修复模式，默认为过分�?
        self.at_once = at_once
        self.labels_given = labels_given
        self.value = value

        # 初始化界面以�?D渲染窗口
        self.init_ui()
        self.init_3d_renderer()  # 初始化持久的3D窗口
        self.update_display()

    def init_3d_renderer(self):
        import pyvista as pv
        self.plotter = pv.Plotter(window_size=[600, 600], title="3D Viewer", off_screen=False)
        # 显示窗口，但设置auto_close=False保证后续更新时该窗口不关�?
        self.plotter.show(interactive_update=True, auto_close=False)

    def update_3d_renderer(self, contour, grey, region_labelled=None):
        import pyvista as pv
        # 创建网格数据
        grid = pv.UniformGrid()
        grid.dimensions = grey.shape  # (Z, Y, X)
        grid.spacing = (1, 1, 1)
        grid.point_data['grey'] = grey.flatten(order='F')
        grid.point_data['contour'] = contour.flatten(order='F')

        self.plotter.clear()  # 清除之前的内�?

        if self.mode == 'gray':
            # 添加灰度图体绘制
            opacity = np.linspace(0, 1, 10)
            opacity[0] = 0
            self.plotter.add_volume(grid, scalars='grey', opacity=opacity, cmap='gray', name='Grey Volume')

        # 提取轮廓：只保留非零标签
        unique_ids = np.unique(contour)
        unique_ids = unique_ids[unique_ids != 0]
        max_id = unique_ids.max() if unique_ids.size > 0 else 0
        colors = get_contour_colors(max_id)

        for idx, id in enumerate(unique_ids):
            mask = (contour == id)
            grid.point_data['mask'] = mask.flatten(order='F')
            surface = grid.contour(isosurfaces=[True], scalars='mask', compute_normals=False)
            if surface.n_points != 0:
                color = colors[idx]
                self.plotter.add_mesh(surface, color=color, name=f'Contour {id}', opacity=self.highlight_item.opacity())

                if self.mode == 'label':
                    lab1, lab2 = self.vector_labels[id]
                    mask1 = region_labelled == lab1
                    mask2 = region_labelled == lab2
                    grid.point_data['mask1'] = mask1.flatten(order='F')
                    grid.point_data['mask2'] = mask2.flatten(order='F')
                    surface1 = grid.contour(isosurfaces=[True], scalars='mask1', compute_normals=False)
                    surface2 = grid.contour(isosurfaces=[True], scalars='mask2', compute_normals=False)
                    if surface1.n_points != 0:
                        self.plotter.add_mesh(surface1, color=(255, 0, 0), name=f'Contour {lab1}',
                                              opacity=self.render_threshold_for_grey / 100)
                    if surface2.n_points != 0:
                        self.plotter.add_mesh(surface2, color=(0, 255, 0), name=f'Contour {lab2}',
                                              opacity=self.render_threshold_for_grey / 100)

                if self.vector3s is not None:
                    # 计算体数据的中心。注意：grid.dimensions 的顺序为 (Z, Y, X)，需要转换为 (X, Y, Z)
                    vector3 = self.vector3s[id]
                    # print(f"Label {id} vector: {vector3}")
                    # 体数据的中心是mask中非0体积的中�?
                    indices = np.argwhere(mask)
                    # print(f"Label {id} indices: {indices}")

                    nz, ny, nx = indices.mean(axis=0)

                    center = (nx, ny, nz)

                    axes_length = np.max(grey.shape) * 0.5
                    origin = center
                    tip_length = 0.2
                    tip_radius = 0.02 * axes_length
                    shaft_radius = 0.01 * axes_length

                    arrow_x = pv.Arrow(start=origin, direction=(axes_length, 0, 0),
                                       tip_length=tip_length, tip_radius=tip_radius, shaft_radius=shaft_radius, scale=3)
                    arrow_y = pv.Arrow(start=origin, direction=(0, axes_length, 0),
                                       tip_length=tip_length, tip_radius=tip_radius, shaft_radius=shaft_radius, scale=3)
                    arrow_z = pv.Arrow(start=origin, direction=(0, 0, axes_length),
                                       tip_length=tip_length, tip_radius=tip_radius, shaft_radius=shaft_radius, scale=3)
                    arrow = pv.Arrow(start=center, direction=vector3, tip_length=tip_length, tip_radius=tip_radius,
                                     shaft_radius=shaft_radius, scale=3)

                    self.plotter.add_mesh(arrow_x, color='red', name=f'X-Axis{id}')
                    self.plotter.add_mesh(arrow_y, color='green', name=f'Y-Axis{id}')
                    self.plotter.add_mesh(arrow_z, color='blue', name=f'Z-Axis{id}')
                    self.plotter.add_mesh(arrow, color="black", name=f"Arrow{id}")
                    print(f"Label {id} vector: {vector3}")

        self.plotter.render()
        # 初始化的时候reset一下视�?
        self.plotter.reset_camera()

    def init_ui(self):
        # 创建主显示部�?
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        layout = QVBoxLayout(self.main_widget)

        # 创建图像显示视图
        self.graphics_view = pg.GraphicsView()
        self.view = pg.ViewBox()
        self.view.setAspectLocked(True, 1)
        self.graphics_view.setCentralItem(self.view)
        layout.addWidget(self.graphics_view)

        # 创建图像�?
        self.img_item = ImageItem()
        self.view.addItem(self.img_item)

        # 创建高亮�?
        self.highlight_item = ImageItem()
        self.highlight_item.setOpacity(0.8)
        self.view.addItem(self.highlight_item)

        # 添加三个并列的滑动条（不透明度、pad和threshold�?
        slider_layout = QHBoxLayout()

        # 不透明度滑动条
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(int(self.highlight_item.opacity() * 100))
        slider_layout.addWidget(self.opacity_slider)

        # pad滑动�?
        self.pad_slider = QSlider(Qt.Horizontal)
        self.pad_slider.setRange(0, 30)
        self.pad_slider.setValue(self.pad)
        slider_layout.addWidget(self.pad_slider)

        # threshold滑动�?
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setValue(self.render_threshold_for_grey)
        slider_layout.addWidget(self.threshold_slider)

        layout.addLayout(slider_layout)

        # 创建坐标显示窗口
        self.coord_dock = QDockWidget("Coordinates", self)
        self.coord_dock.setFeatures(QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetMovable)
        self.coord_dock.setFloating(True)

        # 创建显示坐标的标�?
        self.coord_label = QLabel()

        # 创建搜索输入框和按钮
        self.search_input = QLineEdit()
        self.search_button = QPushButton("Search")

        # 布局容器
        dock_container = QWidget()
        v_layout = QVBoxLayout(dock_container)
        v_layout.addWidget(self.coord_label)

        # 将搜索输入框和按钮放到一�?
        h_layout = QHBoxLayout()
        h_layout.addWidget(self.search_input)
        h_layout.addWidget(self.search_button)
        v_layout.addLayout(h_layout)

        # 添加模式切换按钮（仅在分割模式下显示�?
        if self.segmode == 'fix':
            self.mode_switch_button = QPushButton(f"Mode: {self.current_fix_mode}")
            v_layout.addWidget(self.mode_switch_button)
            self.mode_switch_button.clicked.connect(self.switch_fix_mode)

        # 添加execute按钮
        self.execute_button = QPushButton("Execute")
        v_layout.addWidget(self.execute_button)

        # 设置容器�?dock widget 的子控件
        self.coord_dock.setWidget(dock_container)
        self.addDockWidget(Qt.RightDockWidgetArea, self.coord_dock)

        # 连接事件处理
        self.graphics_view.scene().sigMouseMoved.connect(self.update_coords)
        self.graphics_view.scene().sigMouseClicked.connect(self.handle_click)
        self.view.setMouseMode(pg.ViewBox.PanMode)
        self.search_button.clicked.connect(self.search_label)
        self.execute_button.clicked.connect(self.execute_action)
        self.opacity_slider.valueChanged.connect(self.change_opacity)
        self.pad_slider.valueChanged.connect(self.change_pad)
        self.threshold_slider.valueChanged.connect(self.change_threshold)

    def switch_fix_mode(self):
        """切换过分�?欠分割修复模�?""
        if self.current_fix_mode == 'fixover':
            self.current_fix_mode = 'fixunder'
        else:
            self.current_fix_mode = 'fixover'

        self.mode_switch_button.setText(f"当前模式: {self.current_fix_mode}")

        # 更新当前等待遮罩显示
        self.update_waiting_mask_display()

    def update_waiting_mask_display(self):
        """更新等待遮罩的显�?""
        if self.current_fix_mode == 'fixover':
            self.waiting_mask = self.waiting_mask_over
        else:
            self.waiting_mask = self.waiting_mask_under

        # 如果有选中的mask，需要重新合�?
        if self.selected_mask is not None and self.waiting_mask is not None:
            self.selected_mask = self.selected_mask | self.waiting_mask

        self.update_display()

    def execute_action(self):

        draw = True  # 是否绘制直方�?

        if not self.at_once:

            if self.waiting_list_over is not None and len(self.waiting_list_over) > 0:

                if draw:

                    overSegCoeff, touchingLabels = segmentation.detect_over_segmentation(
                        self.label_vol)
                    filtered_coeff = overSegCoeff[overSegCoeff != 0]
                    values = [overSegCoeff[idx] for pair in self.waiting_list_over for idx in pair]
                    # Plot histogram
                    plt.hist(filtered_coeff, bins=30, edgecolor='black', alpha=0.7)
                    plt.title('Frequency Distribution Histogram')
                    plt.xlabel('Value')
                    plt.ylabel('Frequency')
                    # Highlight the values in the histogram
                    for value in values:
                        plt.axvline(value, color='red', linestyle='dashed', linewidth=1)
                    plt.show()
                    print(values)

                result = segmentation.fix_over_segmentation(self.label_vol, self.waiting_list_over,
                                                                          verbose=True, imShowProgress=True)


            if self.waiting_list_under is not None and len(self.waiting_list_under) > 0:

                underSegCoeff = segmentation.detect_under_segmentation(self.label_vol)
                filter_under = underSegCoeff[underSegCoeff != 0]

                gray = (self.gray_vol - self.gray_vol.min()) / (self.gray_vol.max() - self.gray_vol.min())

                result = segmentation.fix_under_segmentation_simplified(self.label_vol, gray,
                                                                                               self.waiting_list_under,
                                                                                               underSegCoeff,
                                                                                               imShowProgress=True,
                                                                                               verbose=True,
                                                                                               disableCoeffCheck=False)

                if draw:

                    values = [underSegCoeff[idx] for idx in self.waiting_list_under]

                    # Plot histogram
                    plt.hist(filter_under, edgecolor='black', alpha=0.7)
                    plt.title('Frequency Distribution Histogram')
                    plt.xlabel('Value')
                    plt.ylabel('Frequency')
                    # Highlight the values in the histogram
                    for value in values:
                        plt.axvline(value, color='red', linestyle='dashed', linewidth=1)
                    plt.show()
                    print("values:")
                    print(values)

            result = spam.label.makeLabelsSequential(result)

        else:
            # At once mode
            if self.current_fix_mode == 'fixover':
                if self.labels_given is None:
                    overSegCoeff, touchingLabels = segmentation.detect_over_segmentation(
                        self.label_vol)
                    filtered_coeff = overSegCoeff[overSegCoeff != 0]
                    value = self.value
                    labels = np.where(overSegCoeff > value)[0]

                    import modifiedlabel_pool
                    result = modifiedlabel_pool.fix_over_segmentation(self.label_vol, labels, touchingLabels,
                                                                    verbose=True, imShowProgress=True)

                    # Plot histogram
                    plt.hist(filtered_coeff, bins=30, edgecolor='black', alpha=0.7)
                    plt.title('Frequency Distribution Histogram')
                    plt.xlabel('Value')
                    plt.ylabel('Frequency')
                    # Highlight the values in the histogram
                    plt.axvline(value, color='red', linestyle='dashed', linewidth=1)
                    plt.show()

                else:
                    labels = self.labels_given
                    result = segmentation.fix_over_segmentation(self.label_vol, labels, verbose=True,
                                                                              imShowProgress=True)
                    if draw:
                        overSegCoeff, touchingLabels = segmentation.detect_over_segmentation(
                            self.label_vol)
                        filtered_coeff = overSegCoeff[overSegCoeff != 0]
                        values = [overSegCoeff[idx] for pair in labels for idx in pair]

                        # Plot histogram
                        plt.hist(filtered_coeff, bins=30, edgecolor='black', alpha=0.7)
                        plt.title('Frequency Distribution Histogram')
                        plt.xlabel('Value')
                        plt.ylabel('Frequency')
                        # Highlight the values in the histogram
                        for value in values:
                            plt.axvline(value, color='red', linestyle='dashed', linewidth=1)
                        plt.show()


            elif self.current_fix_mode == 'fixunder':
                underSegCoeff = segmentation.detect_under_segmentation(self.label_vol)
                filter_under = underSegCoeff[underSegCoeff != 0]

                if self.labels_given is None:
                    value = self.value
                    labels = np.where(underSegCoeff > value)[0]
                    # Plot histogram
                    plt.hist(filter_under, edgecolor='black', alpha=0.7)
                    plt.title('Frequency Distribution Histogram')
                    plt.xlabel('Value')
                    plt.ylabel('Frequency')
                    # Highlight the values in the histogram
                    plt.axvline(value, color='red', linestyle='dashed', linewidth=1)
                    plt.show()
                else:
                    labels = self.labels_given
                    print(labels)

                    if draw:
                        values = underSegCoeff[labels]
                        # Plot histogram
                        plt.hist(filter_under, edgecolor='black', alpha=0.7)
                        plt.title('Frequency Distribution Histogram')
                        plt.xlabel('Value')
                        plt.ylabel('Frequency')
                        for value in values:
                            plt.axvline(value, color='red', linestyle='dashed', linewidth=1)
                        plt.show()

                gray = (self.gray_vol - self.gray_vol.min()) / (self.gray_vol.max() - self.gray_vol.min())

                result = segmentation.fix_under_segmentation_simplified(self.label_vol, gray,
                                                                                               labels, underSegCoeff,
                                                                                               imShowProgress=True,
                                                                                               verbose=True,
                                                                                               disableCoeffCheck=False)


            result = spam.label.makeLabelsSequential(result)


        print("finish")
        label_path = f"C:\\Users\\11388\\Desktop\\scientific\\scripts\\archive\\data\\label.tif"
        contour_path = f"C:\\Users\\11388\\Desktop\\scientific\\scripts\\archive\\data\\contour.tif"
        contour_color_path = f"C:\\Users\\11388\\Desktop\\scientific\\scripts\\archive\\data\\contour_colored.tif"
        merge_path = f"C:\\Users\\11388\\Desktop\\scientific\\scripts\\archive\\data\\result.tif"

        # 先将原先位于这四个位置的文件备份，存到data\\backup目录�?
        backup_dir = "C:\\Users\\11388\\Desktop\\scientific\\scripts\\archive\\data\\backup"
        os.makedirs(backup_dir, exist_ok=True)
        from tifffile import imread
        if os.path.exists(label_path):
            tifffile.imwrite(os.path.join(backup_dir, "label_backup.tif"), imread(label_path))
        if os.path.exists(contour_path):
            tifffile.imwrite(os.path.join(backup_dir, "contour_backup.tif"), imread(contour_path))
        if os.path.exists(contour_color_path):
            tifffile.imwrite(os.path.join(backup_dir, "contour_color_backup.tif"), imread(contour_color_path))
        if os.path.exists(merge_path):
            tifffile.imwrite(os.path.join(backup_dir, "merge_backup.tif"), imread(merge_path))

        tifffile.imwrite(label_path, result)
        print("labelled save")

        contour, contour_colored, mergeresult = viz_utils.create_contour_image_fast(result, self.gray_vol)

        tifffile.imwrite(merge_path, mergeresult, photometric='rgb')
        tifffile.imwrite(contour_path, contour)
        tifffile.imwrite(contour_color_path, contour_colored, photometric='rgb')
        print("contour save")

        self.label_vol = result
        self.contour_colored = contour_colored
        self.contour_vol = contour
        self.merge_vol = mergeresult

        # 清除所有选中
        self.selected_mask = None
        self.selected_labels = []

        # 清除对应模式的等待列表和遮罩
        if self.current_fix_mode == 'fixover':
            self.waiting_list_over = []
            self.waiting_mask_over = None
        else:
            self.waiting_list_under = []
            self.waiting_mask_under = None
        self.waiting_mask = None

        self.boundingBoxes = spam.label.boundingBoxes(self.label_vol)
        self.update_display()
        # 如果�?D窗口，也清除3D显示
        if hasattr(self, 'plotter'):
            self.plotter.clear()
            self.plotter.render()

    def change_opacity(self, value):
        self.highlight_item.setOpacity(value / 100.0)
        self.update_3d_display()
        self.update_display()

    def change_pad(self, value):
        self.pad = value
        self.update_3d_display()

    def change_threshold(self, value):
        self.render_threshold_for_grey = value
        self.update_3d_display()

    def search_label(self):
        label = self.search_input.text()
        if label.isdigit():
            label = int(label)
            mask = self.label_vol == label
            self.selected_mask = mask
            self.selected_labels = [label]
            try:
                self.current_z = np.where(mask)[0][0]
                self.update_display()
            except:
                self.search_label.setText("Label not found.")

    def update_display(self):
        # 生成当前层的叠加图像
        gray_slice = self.gray_vol[self.current_z]
        contour_slice = self.contour_vol[self.current_z]
        """
        # 创建RGB图像
        img = np.stack([gray_slice]*3, axis=-1).astype(np.uint8)

        # 添加轮廓颜色
        for label, color in self.colormap.items():
            mask = contour_slice == label
            img[mask] = color
        """
        img = self.merge_vol[self.current_z]
        self.img_item.setImage(img.swapaxes(0, 1))  # 转置以适应显示坐标�?
        label_img = self.label_vol[self.current_z]
        # 清除之前绘制的箭�?
        if hasattr(self, 'arrowItems'):
            for arrow in self.arrowItems:
                self.view.removeItem(arrow)
        else:
            self.arrowItems = []
        arrowItems = []

        # 对当前层的每个非零label，计算其中心并绘制箭�?
        unique_labels = np.unique(label_img)
        for label in unique_labels:
            if label == 0:
                continue
            mask = label_img == label
            if np.any(mask):
                # 计算label区域的中心（注意：label_img的索引顺序为(y, x)�?
                indices = np.argwhere(mask)
                center = indices.mean(axis=0)
                # 如果存在对应的向量，则取其在平面上的投影（忽略z分量�?
                if self.vector3s is not None and label in self.vector3s:
                    vec = self.vector3s[label]
                    proj = vec[:2]  # 投影到XY平面
                    # print(f"Label {label} vector: {vec}, projected: {proj}")
                    norm = np.linalg.norm(proj)
                    if norm > 0:
                        proj = proj / norm * 20  # 缩放箭头长度�?0像素
                    else:
                        proj = np.zeros_like(proj)
                    # 计算箭头角度（pyqtgraph的ArrowItem�?度指向右边，角度正方向为逆时针）
                    angle = np.degrees(np.arctan2(-proj[1], proj[0]))

                    if self.red_labels is not None and label in self.red_labels:
                        arrow = pg.ArrowItem(pos=(center[1], center[0]), angle=angle,
                                             headLen=10, tipAngle=30, baseAngle=20, brush='r')
                    else:
                        arrow = pg.ArrowItem(pos=(center[1], center[0]), angle=angle,
                                             headLen=10, tipAngle=30, baseAngle=20, brush='y')

                    self.view.addItem(arrow)
                    arrowItems.append(arrow)

        self.arrowItems = arrowItems

        # 更新高亮显示
        if self.selected_mask is not None:
            highlight = np.zeros(img.shape[:2], dtype=bool)
            if self.current_z < self.selected_mask.shape[0]:
                highlight = self.selected_mask[self.current_z]
            self.highlight_item.setImage(
                highlight.T.astype(np.uint8) * 255,  # 转置并转换为图像
                levels=(0, 255),
                opacity=self.highlight_item.opacity(),
                compositionMode=pg.QtGui.QPainter.CompositionMode_Plus
            )

    def update_3d_display(self):
        if self.selected_mask is not None:
            # 更新3D渲染内容
            padding = self.pad
            startZ = min(self.boundingBoxes[label, 0] for label in self.selected_labels) - padding
            stopZ = max(self.boundingBoxes[label, 1] for label in self.selected_labels) + padding
            startY = min(self.boundingBoxes[label, 2] for label in self.selected_labels) - padding
            stopY = max(self.boundingBoxes[label, 3] for label in self.selected_labels) + padding
            startX = min(self.boundingBoxes[label, 4] for label in self.selected_labels) - padding
            stopX = max(self.boundingBoxes[label, 5] for label in self.selected_labels) + padding

            depth, height, width = self.contour_vol.shape

            startZ = max(0, startZ)
            stopZ = min(stopZ, depth)
            startY = max(0, startY)
            stopY = min(stopY, height)
            startX = max(0, startX)
            stopX = min(stopX, width)

            # 提取区域
            # region_to_3D_contour = self.contour_vol[startZ:stopZ, startY:stopY, startX:stopX]
            region_to_3D_grey = self.gray_vol[startZ:stopZ, startY:stopY, startX:stopX]
            region_to_3D_label = self.label_vol[startZ:stopZ, startY:stopY, startX:stopX]
            if self.mode == 'label':
                region_labelled = self.reference_labelled[startZ:stopZ, startY:stopY, startX:stopX]
            else:
                region_labelled = None

            # region_to_3D_contour只保留selected_labels的轮廓，但轮廓值不�?
            # region_to_3D_contour = np.where(np.isin(region_to_3D_contour, self.selected_labels), region_to_3D_contour, 0)
            # region_to_3D_label只保留selected_labels的标签，但标签值不�?
            region_to_3D_label = np.where(np.isin(region_to_3D_label, self.selected_labels), region_to_3D_label, 0)
            # region_to_3D_grey删去region_to_3D_contour对应的位�?
            # region_to_3D_grey = np.where(region_to_3D_contour == 0, region_to_3D_grey, 0)

            # grey中小于最大值的render_threshold_for_grey%的值设�?
            max_grey = region_to_3D_grey.max()
            threshold = max_grey * self.render_threshold_for_grey / 100
            region_to_3D_grey = np.where(region_to_3D_grey > threshold, region_to_3D_grey, 0)

            # 调用持久�?D渲染窗口更新内容
            self.update_3d_renderer(region_to_3D_label, region_to_3D_grey, region_labelled)

    def update_coords(self, pos):
        # 将鼠标位置转换为图像坐标
        img_coords = self.img_item.mapFromScene(pos)
        x, y = img_coords.x(), img_coords.y()
        if 0 <= x < self.gray_vol.shape[2] and 0 <= y < self.gray_vol.shape[1]:
            self.coord_label.setText(
                f"""Z: {self.current_z}, Y: {int(y)}, X: {int(x)}
                value: {self.gray_vol[self.current_z, int(y), int(x)]}, label: {self.label_vol[self.current_z, int(y), int(x)]}
                opacity: {self.highlight_item.opacity()}"""
            )

    def handle_click(self, event):
        pos = event.pos()
        img_coords = self.img_item.mapFromScene(pos)
        x, y = int(img_coords.x()), int(img_coords.y())

        if self.current_fix_mode == 'fixover':
            waiting_mask = self.waiting_mask_over
        else:
            waiting_mask = self.waiting_mask_under

        if 0 <= x < self.gray_vol.shape[2] and 0 <= y < self.gray_vol.shape[1]:
            label = self.label_vol[self.current_z, y, x]
            if label != 0:
                label_mask = self.label_vol == label
                # 如果按住Shift键，则累加选择的label，否则单独选择
                if event.modifiers() & Qt.ShiftModifier:
                    if hasattr(self, 'selected_mask') and self.selected_mask is not None:
                        self.selected_mask = self.selected_mask | label_mask
                        self.selected_labels.append(label)
                    else:
                        self.selected_mask = label_mask
                        self.selected_labels = [label]
                else:
                    self.selected_mask = label_mask
                    self.selected_labels = [label]
                if waiting_mask is not None:
                    self.selected_mask = self.selected_mask | waiting_mask
                self.update_display()
                self.update_3d_display()

    def wheelEvent(self, event):
        # 滚轮切换Z�?
        delta = event.angleDelta().y()
        if delta > 0 and self.current_z < self.gray_vol.shape[0] - 1:
            self.current_z += 1
        elif delta < 0 and self.current_z > 0:
            self.current_z -= 1
        self.coord_label.setText(
            f"""Z: {self.current_z}"""
        )
        self.update_display()

    def keyPressEvent(self, event):
        # 处理回车事件
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            if self.selected_labels:
                print(f"当前选中的颗粒ID: {self.selected_labels}")
                if self.current_fix_mode == 'fixover':
                    if len(self.selected_labels) == 2:
                        labela, labelb = self.selected_labels
                        self.waiting_list_over.append((labela, labelb))
                        if self.waiting_mask_over is None:
                            self.waiting_mask_over = (self.label_vol == labela) | (self.label_vol == labelb)
                        else:
                            self.waiting_mask_over = (self.waiting_mask_over) | (self.label_vol == labela) | (
                                    self.label_vol == labelb)
                        print("当前等待处理的过分割合并操作:", self.waiting_list_over)
                    else:
                        print("请选中两个颗粒进行合并")
                elif self.current_fix_mode == 'fixunder':
                    if len(self.selected_labels) == 1:
                        label = self.selected_labels[0]
                        self.waiting_list_under.append(label)
                        if self.waiting_mask_under is None:
                            self.waiting_mask_under = self.label_vol == label
                        else:
                            self.waiting_mask_under = self.waiting_mask_under | (self.label_vol == label)
                        print("当前等待处理的欠分割分割操作:", self.waiting_list_under)
                    else:
                        print("请选中一个颗粒进行分�?)

                # 清除所有选中
                self.selected_mask = None
                self.selected_labels = []
                self.update_display()
                # 如果�?D窗口，也清除3D显示
                if hasattr(self, 'plotter'):
                    self.plotter.clear()
                    self.plotter.render()
            else:
                print("当前没有选中的颗�?)
        else:
            # 调用父类的事件处理器
            super().keyPressEvent(event)


def main(appendname="", contactORI="", reference_image=None, deal_contact=False, check_vector_tendancy=False,
         deal_orientation=False, fix=False, at_once=False, value=1, labels_given=None):
    # 示例数据路径和颜色映�?
    label_path = f"C:\\Users\\11388\\Desktop\\scientific\\scripts\\archive\\data\\label{appendname}.tif"
    gray_path = f"C:\\Users\\11388\\Desktop\\scientific\\scripts\\archive\\data\\grey{appendname}.tif"
    colormap = {1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255)}

    # 加载数据（假设数据为ZYX顺序�?
    label_vol = imread(label_path)
    gray_vol = imread(gray_path)

    # from 三维图叠�?import create_contour_image
    # contour_vol,contour_colored,merge_vol = create_contour_image(label_vol, gray_vol)
    contour_path = f"C:\\Users\\11388\\Desktop\\scientific\\scripts\\archive\\data\\contour{appendname}.tif"
    contour_color_path = f"C:\\Users\\11388\\Desktop\\scientific\\scripts\\archive\\data\\contour_colored{appendname}.tif"
    merge_path = f"C:\\Users\\11388\\Desktop\\scientific\\scripts\\archive\\data\\result{appendname}.tif"

    contour_vol = imread(contour_path)
    contour_colored = imread(contour_color_path)
    merge_vol = imread(merge_path)

    # 归一化灰度图�?
    gray_vol = (gray_vol - gray_vol.min()) / (gray_vol.max() - gray_vol.min()) * 255
    gray_vol = gray_vol.astype(np.uint8)

    if fix:
        app = QApplication(sys.argv)
        window = MainWindow(gray_vol, contour_vol, label_vol, colormap, contour_colored, merge_vol, mode="gray",
                            segmode='fix', at_once=at_once, value=value, labels_given=labels_given)
        window.setWindowTitle("3D Image Viewer")
        window.resize(800, 600)
        window.show()
        sys.exit(app.exec_())

    else:
        if appendname == "S60Glabel" or appendname == "":
            app = QApplication(sys.argv)
            window = MainWindow(gray_vol, contour_vol, label_vol, colormap, contour_colored, merge_vol, mode="gray")
            window.setWindowTitle("3D Image Viewer")
            window.resize(800, 600)
            window.show()
            sys.exit(app.exec_())

        else:
            if deal_contact:
                contact_datas = np.load(r"D:\WorkSpace\scientific\scripts\archive\S60尺寸200金标准f-contactORI.npz",
                                        allow_pickle=True)
                contactVolume = contact_datas['contactVolume']
                contactOrientations_ORI = contact_datas['contactOrientations']
                reference_image = tifffile.imread(r"D:\WorkSpace\scientific\scripts\archive\S60尺寸200金标准f.tif")

                labels = np.unique(label_vol)
                labels = labels[labels != 0]
                vectors = {}
                vector_labels = {}
                contact_grays, bothgreys, ratios3 = [], [], []
                sizes, bothsizes, retiosizes = [], [], []
                xs, ys, zs = [], [], []
                for label in labels:
                    print("当前处理的标�?", label)
                    mask = label_vol == label
                    reference_mask = reference_image[mask]
                    print(reference_mask)
                    values, counts = np.unique(reference_mask, return_counts=True)
                    print(values)
                    lab1, lab2 = values[counts.argsort()[-2:]]
                    row_mask = ((contactOrientations_ORI[:, 0] == lab1) & (contactOrientations_ORI[:, 1] == lab2)) | \
                               ((contactOrientations_ORI[:, 0] == lab2) & (contactOrientations_ORI[:, 1] == lab1))
                    matching_rows = contactOrientations_ORI[row_mask]
                    if matching_rows.size > 0:
                        xyz = matching_rows[0, 2:5]
                        print("匹配�?xyz �?", xyz)
                        vectors[label] = xyz
                        vector_labels[label] = (lab1, lab2)

                    else:
                        print("未找到匹配的行�?)
                        return

                    if label == 32:
                        padding = 2
                        boundingBoxes = spam.label.boundingBoxes(reference_image)
                        startZ = min(boundingBoxes[lab1, 0], boundingBoxes[lab2, 0]) - padding
                        stopZ = max(boundingBoxes[lab1, 1], boundingBoxes[lab2, 1]) + padding
                        startY = min(boundingBoxes[lab1, 2], boundingBoxes[lab2, 2]) - padding
                        stopY = max(boundingBoxes[lab1, 3], boundingBoxes[lab2, 3]) + padding
                        startX = min(boundingBoxes[lab1, 4], boundingBoxes[lab2, 4]) - padding
                        stopX = max(boundingBoxes[lab1, 5], boundingBoxes[lab2, 5]) + padding

                        sub = reference_image[startZ:stopZ + 1, startY:stopY + 1, startX:stopX + 1]
                        subgray = gray_vol[startZ:stopZ + 1, startY:stopY + 1, startX:stopX + 1]

                        sub_mask = (sub == lab1) | (sub == lab2)
                        subgray[~sub_mask] = 0
                        sub[~sub_mask] = 0
                        tifffile.imwrite(f"D:\\WorkSpace\\scientific\\scripts\\archive\\{appendname}Mlabel-32.tif", sub)
                        tifffile.imwrite(f"D:\\WorkSpace\\scientific\\scripts\\archive\\{appendname}Mgrey-32.tif",
                                         subgray)
                        return

                    contact_grey = np.mean(gray_vol[mask])
                    bothgrey = np.mean(gray_vol[(reference_image == lab1) | (reference_image == lab2)])
                    ratio3 = contact_grey / bothgrey
                    contact_grays.append(contact_grey)
                    bothgreys.append(bothgrey)
                    ratios3.append(ratio3)
                    sizes.append(np.sum(mask))
                    bothsizes.append(np.sum((reference_image == lab1) | (reference_image == lab2)))
                    retiosizes.append(np.sum(mask) / np.sum((reference_image == lab1) | (reference_image == lab2)))
                    xs.append(xyz[0])
                    ys.append(xyz[1])
                    zs.append(xyz[2])

                import pandas as pd
                df = pd.DataFrame({
                    'label': labels,
                    'contact_grey': contact_grays,
                    'bothgrey': bothgreys,
                    'ratio3': ratios3,
                    'size': sizes,
                    'bothsize': bothsizes,
                    'retiosize': retiosizes,
                    'x': xs,
                    'y': ys,
                    'z': zs
                })
                df.to_csv(f"D:\\WorkSpace\\scientific\\scripts\\archive\\{appendname}M-contact.csv", index=False)
                np.savez(f"D:\\WorkSpace\\scientific\\scripts\\archive\\{appendname}M-vectors.npz", vectors=vectors)
                np.savez(f"D:\\WorkSpace\\scientific\\scripts\\archive\\{appendname}M-vectors_label.npz",
                         vector_labels=vector_labels)
                return

            if check_vector_tendancy:
                vectors_data = np.load(
                    f"C:\\Users\\11388\\Desktop\\scientific\\scripts\\archive\\{appendname}M-vectors.npz",
                    allow_pickle=True)
                vectors_label_data = np.load(
                    f"C:\\Users\\11388\\Desktop\\scientific\\scripts\\archive\\{appendname}M-vectors_label.npz",
                    allow_pickle=True)
                vectors = vectors_data['vectors'].item()
                vectors_label = vectors_label_data['vector_labels'].item()

                vectors[13] = np.array([0.37713, 0.00000, 0.92616])

                refer_vector = np.array([0, 1, 0])
                threshold = 0.99
                # 如果vectors[label]与refer_vector的内积小于threshold，就记录下label和vectors[label]
                vectors_selected = {label: vectors[label] for label in vectors if
                                    np.abs(np.dot(vectors[label], refer_vector)) >= threshold}

                vectors_selected_values = np.array(list(vectors_selected.values()))
                vectors_selected_labels = np.array(list(vectors_selected.keys()))
                print(vectors_selected_labels)

            if deal_orientation:
                # MOIeigenValues, MOIeigenVectors = spam.label.momentOfInertia(label_vol)
                # vectors = MOIeigenVectors[:,6:9]
                # 转换为字�?
                # vectors = {i:vectors[i] for i in range(vectors.shape[0])}
                if False:
                    import matplotlib
                    import modifiedlabel_pool
                    maxval = segmentation.plot_spherical_histogram(vectors_selected_values,
                                                                                           title="",
                                                                                           color=matplotlib.pyplot.cm.viridis_r)
                    print(maxval)
                    # >>�?
                    return

            # Load reference image (Update with your local path)
            # reference_labelled = tifffile.imread("path/to/reference.tif")
            reference_labelled = None 
            app = QApplication(sys.argv)
            window = MainWindow(gray_vol, contour_vol, label_vol, colormap, contour_colored, merge_vol, vectors,
                                mode="label", vector_labels=vectors_label, reference_labelled=reference_labelled,
                                red_labels=vectors_selected_labels)
            window.setWindowTitle("3D Image Viewer")
            window.resize(800, 600)
            window.show()
            sys.exit(app.exec_())


if __name__ == "__main__":
    # main("S60G局部阈�?00差量")
    # main("S60Glabel")
    # main("长方体实�?)
    # main("S60G局部阈�?00")
    # main(fix=True,at_once=False,value=1)
    main(fix=True, at_once=False, value=1.2)
