import cv2
import numpy as np
import pandas as pd
import h5py
import math
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import detect_sources
from astropy.convolution import convolve
from photutils.segmentation import make_2dgaussian_kernel
from photutils.segmentation import SourceCatalog
from photutils.segmentation import deblend_sources
import to_rgb


def is_blended_improved(file):
    image = to_rgb.dr2_rgb(file, ['g', 'r', 'z'])
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    data = img
    bkg_estimator = MedianBackground()
    bkg = Background2D(data, (50, 50), filter_size=(3, 3),
                       bkg_estimator=bkg_estimator)
    data -= bkg.background  # subtract the background
    threshold = 1.5 * bkg.background_rms
    kernel = make_2dgaussian_kernel(3.0, size=5)  # FWHM = 3.0
    convolved_data = convolve(data, kernel)
    half_x = int(data.shape[0] / 2)
    half_y = int(data.shape[1] / 2)
    # 检测到的源x
    segment_map = detect_sources(convolved_data, threshold, npixels=10)
    pre_label = segment_map.data[half_x][half_y]

    pre_cat = SourceCatalog(data, segment_map, convolved_data=convolved_data)
    pre_object = pre_cat.get_labels(pre_label).to_table()

    segm_deblend = deblend_sources(convolved_data, segment_map,
                                   npixels=10, nlevels=90, contrast=0.000001,  # 改参数
                                   progress_bar=False)
    cat = SourceCatalog(data, segm_deblend, convolved_data=convolved_data)
    tbl = cat.to_table()
    label = segm_deblend.data[half_x][half_y]
    object = cat.get_labels(label).to_table()

    if len(tbl) == 1:
        return "OK"

    else:
        return "NO"  # 重叠


def iterative_threshold(image):
    # 获取图像的最大和最小灰度值
    cmax = np.max(image)
    cmin = np.min(image)
    # print(cmin)
    # 初始化阈值为 (cmax + cmin) / 2
    T_prev = (cmax + cmin) / 2
    T_next = 0

    while True:# 开始一个无限循环
        # 根据当前阈值将图像分为前景和背景两类
        foreground = image >= T_prev
        background = image < T_prev

        # 计算前景和背景的平均灰度值
        cobj = np.mean(image[foreground])
        cbkg = np.mean(image[background])

        # 更新阈值
        T_next = (cobj + cbkg) / 2
        # 如果下一个阈值与当前阈值的差值小于0.1，或者两者相等，则退出循环
        if T_next == T_prev:
            T_next= T_next# 如果条件满足，将当前阈值设置为下一个阈值并退出循环
            break
        # 更新当前阈值为下一个阈值，以进行下一轮迭代
        T_prev = T_next
        # print("T_next:",T_next)

    return T_next# 返回最佳阈值

def find_enclosing_circle(image):
    try:
        # 使用 connectedComponentsWithStats 获取连通区域信息
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image)
        # 选择要处理的目标区域，这里假设选择面积最大的非背景区域
        target_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        # 使用 np.where 找到目标连通区域的所有像素坐标
        target_pixels = np.where(labels == target_label)
        # 获取目标连通区域的坐标
        target_coordinates = np.column_stack(target_pixels)
        # 使用 cv2.minEnclosingCircle 计算最小外接圆的圆心坐标和半径
        center, radius = cv2.minEnclosingCircle(target_coordinates)
        # 将圆心坐标和半径返回
        center = (int(center[0]), int(center[1]))
        radius = int(radius)
        return center, radius
    except Exception as e:
        print("图片处理异常:", e)
        center = (int(75) ,int(75))
        radius = int(1000)
        return center, radius



def overlap_detection(processed_image):
    img_rgb = np.transpose(processed_image, (1, 2, 0))
    accurate = True  # 结果是否准确

    inputMat = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

    # 图像二值化 ：
    threshold_value = iterative_threshold(inputMat)

    _, thresholded = cv2.threshold(inputMat, threshold_value, 255, cv2.THRESH_BINARY)
    thresholded = thresholded.astype(np.uint8)


    # 求连通域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresholded)

    if num_labels > 2:
        accurate = False

    return accurate


def mask_ellipse_center(image, Expansion_scale=1.4, accuracy=5):
    img_rgb = to_rgb.dr2_rgb(image, ['g', 'r', 'z'])
    accurate = True  # 结果是否准确
    reason = None  # 不准确的原因
    end_image = np.transpose(image, (1, 2, 0))

    # 1.转化成灰度图像：
    inputMat = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

    # 2.图像二值化 ：

    threshold_value = iterative_threshold(inputMat)
    # print("threshold_value:",threshold_value )
    _, thresholded = cv2.threshold(inputMat, threshold_value, 255, cv2.THRESH_BINARY)

    # 3.边缘检测(连通域的边缘)：
    edges = cv2.Canny(thresholded.astype(np.uint8), 30, 90)
    # 轮廓提取（每个连通域的轮廓为一个contours）：
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # 在图像上绘制轮廓：
    # contour_image = cv2.drawContours(rgb, contours, -1, (0, 255, 0), 2)

    end_ellipse = None  # 存储拟合到的中心椭圆
    fit_rate = None  # 存储拟合与实际面积比值
    center_x = inputMat.shape[1] / 2
    center_y = inputMat.shape[0] / 2
    min_distance = 9999

    # 4.将每个星体拟合成一个椭圆，并保存图像中心的椭圆：
    for contour in contours:

        # 轮廓包含的点数小于5则无法拟合：
        if len(contour) >= 5:

            # 拟合最小椭圆
            ellipse = cv2.fitEllipse(contour)

            # 椭圆参数
            center_e_x = ellipse[0][0]
            center_e_y = ellipse[0][1]
            a = ellipse[1][0] / 2  # 长轴
            b = ellipse[1][1] / 2
            angle = ellipse[2]

            # 检查长轴和短轴是否为零
            if a > 3 and b > 3:
                distance = math.sqrt((center_e_x - center_x) ** 2 + (center_e_y - center_y) ** 2)
                operator = ((center_x - center_e_x) ** 2 / a ** 2) + ((center_y - center_e_y) ** 2 / b ** 2)

                # 判断给定点是否在椭圆内
                if operator <= 1:
                    # 判断给定点距离原点距离
                    if distance <= min_distance:
                        min_distance = distance
                        end_ellipse = ellipse

                        # 计算拟合与实际面积比值：
                        area_ellipse = math.pi * a * b  # 计算椭圆的面积
                        area_contour = cv2.contourArea(contour)  # 计算轮廓包含的面积
                        fit_rate = area_contour / area_ellipse

    # 5.遮盖其余图像
    # 定义掩码
    mask = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
    center = None
    radius = None
    # 判断有无找到中心星体：
    if min_distance < accuracy:

        # 定义椭圆参数
        center = (int(end_ellipse[0][0]), int(end_ellipse[0][1]))
        axes = (int(end_ellipse[1][0] / 2 * Expansion_scale), int(end_ellipse[1][1] / 2 * Expansion_scale))
        angle = int(end_ellipse[2])

        # 在掩码上绘制椭圆区域（白色部分）
        cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)

        # 黑色遮盖椭圆范围外的部分
        end_image = cv2.bitwise_and(end_image, end_image, mask=mask)
        # print(thresholded.shape)
        # print(thresholded)
        # print(mask.shape)
        thresholded = thresholded.astype(np.uint8)
        thresholded_mask = cv2.bitwise_and(thresholded, mask)
        center, radius = find_enclosing_circle(thresholded_mask)
        # print(center, radius)

    else:
        # 在图像中心点没有找到星体
        accurate = False
        reason = f"Minimum distance greater than {accuracy}:{min_distance}"

    result = {
        'image': np.transpose(end_image, (2, 0, 1)),
        'accurate': accurate,
        'reason': reason,
        'fit_rates': fit_rate,
        'ellipses': end_ellipse,
        'distance': min_distance,
        'center': center,
        'radius': radius
    }
    return result

def process_and_save_data(h5_file_path, radius_Limit=[0, 23], new_h5_file_path=None, new_csv_file_path=None,
                          save_old=True):
    # 打开H5旧文件，指定读模式 'r'，允许对文件进行读取
    with h5py.File(h5_file_path, 'r') as h5_file:
        # 检查'image' key是否存在
        if 'images' in h5_file:
            # 获取'image' key对应的数据集
            dataset = h5_file['images']
            # 获取所有的key用于将已有参数复制到新图像
            keys = list(h5_file.keys())

            processed_images = []  # 保存处理过的图像
            distances = []  # 保存拟合的椭圆的准确度，越小越好
            circles_radius = []  # 保存外接圆半径（像素点）
            circles_arc = []  # 角秒
            selected_indices = []  # 保存符合的索引
            data_dict = {}

            # 计数
            num_tol = 0
            num_success = 0
            num_six = 0
            num_conform = 0

            for i, image in enumerate(dataset):
                # 调用函数得到新图像等数据
                result = mask_ellipse_center(image)
                # result2 = is_blended(image)
                num_tol += 1

                # 成功识别
                # if result['accurate'] and result2 == 'OK':
                if result['accurate']:
                    num_success += 1
                    processed_image = result['image']  # 图像数据
                    distance = result['distance']  # 椭圆圆心距离原点距离
                    circle_radius = result['radius']  # 外接圆半径
                    circle_arc = circle_radius * 0.262
                    # 符合6角秒
                    if circle_radius > radius_Limit[0] and circle_radius < radius_Limit[1]:
                        # if oval_size >radius_Limit[0]*2 and oval_size<radius_Limit[1]*2:
                        num_six += 1
                        if is_blended_improved(result['image']) == 'OK' and overlap_detection(processed_image):
                            if new_h5_file_path:
                                processed_images.append(processed_image)
                            selected_indices.append(i)
                            num_conform += 1
                            distances.append(distance)
                            circles_radius.append(circle_radius)
                            circles_arc.append(circle_arc)
                if i % 1000 == 0:
                    print(i)
            print("总数：", num_tol)
            print("成功检测：", num_success)
            print("小于6角秒：", num_six)
            print("符合总数：", num_conform)

            if new_h5_file_path:
                # 创建新H5文件
                with h5py.File(new_h5_file_path, 'w') as new_h5_file:

                    # new_h5_file.create_dataset('processed_images', data=processed_images)
                    new_h5_file.create_dataset('processed_images', shape=(len(processed_images), 3, 152, 152))
                    for i in range(len(processed_images)):
                        new_h5_file['processed_images'][i] = processed_images[i]

                    new_h5_file.create_dataset('circle_radius', data=circles_radius)
                    new_h5_file.create_dataset('oval_distances', data=distances)
                    new_h5_file.create_dataset('circle_arc', data=circles_arc)

                    for key in keys:

                        if key != 'images':
                            # 读取原始H5文件中的数据
                            data = h5_file[key][:]
                            # 根据索引列表获取要保留的数据
                            selected_data = data[selected_indices]
                            data_dict[key] = selected_data[:]
                            # 将数据保存到新的H5文件中
                            new_h5_file.create_dataset(key, data=selected_data)
                        else:
                            if save_old:
                                data = h5_file['images']
                                selected_data = data[selected_indices]
                                new_h5_file.create_dataset('images', shape=(len(selected_data), 3, 152, 152))
                                for i in range(len(selected_data)):
                                    new_h5_file['images'][i] = selected_data[i]

                print("新H5已生成")
            else:
                print("缺少H5路径")
                for key in keys:
                    if key != 'images':
                        # 读取原始H5文件中的数据
                        data = h5_file[key][:]
                        # 根据索引列表获取要保留的数据
                        selected_data = data[selected_indices]
                        data_dict[key] = selected_data[:]

            if new_csv_file_path:
                df = pd.DataFrame(data_dict)
                df['circles_radius'] = circles_radius
                df['oval_distances'] = distances
                df['circle_arc'] = circles_arc
                df.to_csv(new_csv_file_path, index=False)
                print("csv已生成")
            else:
                print("缺少csv路径")
        else:
            print("Key 'image' 不存在于H5文件中。")

#
for i in range(0,42):
    h5_file_path = 'project/experiment/desi/image_h5/south_'+f"{i:02d}"+'_'+f"{i+1:02d}"+'.h5'
    print(h5_file_path)
    new_h5_file_path = 'project/experiment/desi/image_h5/new_south_'+f"{i:02d}"+'_'+f"{i+1:02d}"+'.h5'
    print(new_h5_file_path)
    new_csv_file_path = 'project/experiment/desi/image_h5/new_south_'+f"{i:02d}"+'_'+f"{i+1:02d}"+'.csv'
    print(new_csv_file_path)

    process_and_save_data(h5_file_path=h5_file_path, new_h5_file_path=new_h5_file_path,
                          new_csv_file_path=new_csv_file_path, save_old=False)

for i in range(0,14):
    h5_file_path = 'project/experiment/desi/image_h5/north_'+f"{i:02d}"+'_'+f"{i+1:02d}"+'.h5'
    print(h5_file_path)
    new_h5_file_path ='project/experiment/desi/image_h5/new_north_'+f"{i:02d}"+'_'+f"{i+1:02d}"+'.h5'
    print(new_h5_file_path)
    new_csv_file_path = 'project/experiment/desi/image_h5/new_north_'+f"{i:02d}"+'_'+f"{i+1:02d}"+'.csv'
    print(new_csv_file_path)

    process_and_save_data(h5_file_path=h5_file_path, new_h5_file_path=new_h5_file_path,
                          new_csv_file_path=new_csv_file_path, save_old=False)



h5_file_path = 'project/experiment/desi/image_h5/north_14_14174203.h5'
print(h5_file_path)
new_h5_file_path = 'project/experiment/desi/image_h5/new_north_14_14174203.h5'
print(new_h5_file_path)
new_csv_file_path = 'project/experiment/desi/image_h5/new_north_14_14174203.csv'
save_old = False
process_and_save_data(h5_file_path=h5_file_path, new_h5_file_path=new_h5_file_path,
                          new_csv_file_path=new_csv_file_path, save_old=False)


