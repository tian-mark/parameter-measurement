
#单波段
import h5py
import numpy as np
import to_rgb
def process_images_in_h5(input_h5_file, output_h5_file, channel_index=2):
    # 打开 h5 文件进行读取
    with h5py.File(input_h5_file, 'r') as file:
        processed_images = file['processed_images'][:]  # 读取图片数据
        oh_p50_values = file['oh_p50'][:]  # 读取相应的oh_p50值
    img_rgb1 = []
    for image in processed_images:
        img_rgb = to_rgb.dr2_rgb(image, ['g', 'r', 'z'])
        img_rgb1.append(img_rgb)
    img_rgb1 = np.array(img_rgb1)
    img_rgb1_0 = [np.repeat(i[:, :, channel_index][..., np.newaxis], 3, axis=-1) for i in img_rgb1]
    img_rgb1_0 = [np.transpose(img, (2, 0, 1)) for img in img_rgb1_0]
    img_rgb1_0 = np.array(img_rgb1_0)
    with h5py.File(output_h5_file, 'w') as file:
        file.create_dataset('processed_images', data=img_rgb1_0)
        file.create_dataset('oh_p50', data=oh_p50_values)

# 使用输入和输出H5文件路径调用函数
input_h5_file_path = 'project/experiment/desi/image_h5_distribution_test/merged.h5'
output_h5_file_path = 'project/experiment/desi_test/band/g_test.h5'
process_images_in_h5(input_h5_file_path, output_h5_file_path)
print("处理单波段文件处理完成并保存到新的H5文件中。")

#双波段
#gr
import h5py
import numpy as np
import to_rgb
import cv2
def process_images_in_h5(input_h5_file, output_h5_file):
    with h5py.File(input_h5_file, 'r') as file:
        processed_images = file['processed_images'][:]  # 读取图片数据
        oh_p50_values = file['oh_p50'][:]  # 读取相应的oh_p50值
    img_bgr_list = []
    for image in processed_images:
        img_rgb = to_rgb.dr2_rgb(image, ['g', 'r', 'z'])
        channel_g = img_rgb[:, :, 2]
        channel_r = img_rgb[:, :, 1]
        channel_0 = (channel_g + channel_r) / 2
        img_bgr = cv2.merge([channel_g, channel_0, channel_r])
        img_bgr_list.append(img_bgr)
    img_bgr_list = [np.transpose(img, (2, 0, 1)) for img in  img_bgr_list]
    img_bgr_array = np.array(img_bgr_list)
    with h5py.File(output_h5_file, 'w') as file:
        file.create_dataset('processed_images', data=img_bgr_array)
        file.create_dataset('oh_p50', data=oh_p50_values)
input_h5_file_path = 'project/experiment/desi/image_h5_distribution/merged.h5'
output_h5_file_path = '/project/experiment/desi_test/band/gr.h5'
process_images_in_h5(input_h5_file_path, output_h5_file_path)
print("处理gr文件处理完成并保存到新的H5文件中。")

#gz
import h5py
import numpy as np
import to_rgb
import cv2

def process_images_in_h5(input_h5_file, output_h5_file):
    # 打开h5文件进行读取
    with h5py.File(input_h5_file, 'r') as file:
        processed_images = file['processed_images'][:]
        oh_p50_values = file['oh_p50'][:]
    img_bgr_list = []
    for image in processed_images:
        img_rgb = to_rgb.dr2_rgb(image, ['g', 'r', 'z'])
        channel_g = img_rgb[:, :, 2]
        channel_z = img_rgb[:, :, 0]
        channel_0 = (channel_g + channel_z) / 2
        img_bgr = cv2.merge([channel_z, channel_0, channel_g])
        img_bgr_list.append(img_bgr)
    img_bgr_list = [np.transpose(img, (2, 0, 1)) for img in  img_bgr_list]
    img_bgr_array = np.array(img_bgr_list)
    with h5py.File(output_h5_file, 'w') as file:
        file.create_dataset('processed_images', data=img_bgr_array)
        file.create_dataset('oh_p50', data=oh_p50_values)
input_h5_file_path = 'project/experiment/desi/image_h5_distribution/merged.h5'
output_h5_file_path = 'project/experiment/desi_test/band/gz.h5'
process_images_in_h5(input_h5_file_path, output_h5_file_path)
print("处理gz文件处理完成并保存到新的H5文件中。")

#rz
import h5py
import numpy as np
import to_rgb
import cv2
def process_images_in_h5(input_h5_file, output_h5_file):
    with h5py.File(input_h5_file, 'r') as file:
        processed_images = file['processed_images'][:]  # 读取图片数据
        oh_p50_values = file['oh_p50'][:]  # 读取相应的oh_p50值
    img_bgr_list = []
    for image in processed_images:
        img_rgb = to_rgb.dr2_rgb(image, ['g', 'r', 'z'])
        channel_r = img_rgb[:, :, 1]
        channel_z = img_rgb[:, :, 0]
        channel_0 = (channel_r + channel_z) / 2
        img_bgr = cv2.merge([channel_r, channel_0, channel_z])
        img_bgr_list.append(img_bgr)
    img_bgr_list = [np.transpose(img, (2, 0, 1)) for img in  img_bgr_list]
    img_bgr_array = np.array(img_bgr_list)
    with h5py.File(output_h5_file, 'w') as file:
        file.create_dataset('processed_images', data=img_bgr_array)
        file.create_dataset('oh_p50', data=oh_p50_values)
input_h5_file_path = 'project/experiment/desi/image_h5_distribution/merged.h5'
output_h5_file_path = 'project/experiment/desi_test/band/rz.h5'
process_images_in_h5(input_h5_file_path, output_h5_file_path)
print("处理rz文件处理完成并保存到新的H5文件中。")