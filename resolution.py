import h5py
import cv2

original_file_path = r"project/experiment/desi/image_h5_distribution/merged.h5"
new_file_path = r"project/experiment/desi_test/fixel/16x16.h5"
with h5py.File(original_file_path, 'r') as f:
    images = f['processed_images'][:]
    num_images, num_channels, height, width = images.shape
    with h5py.File(new_file_path, 'w') as new_f:
        for key, value in f.items():
            if key != 'processed_images':
                f.copy(value, new_f)
        new_images = new_f.create_dataset('processed_images', shape=(num_images, num_channels, 16, 16),
                                          dtype=images.dtype)
        for i in range(num_images):
            original_image = images[i]
            resized_image = cv2.resize(original_image.transpose((1, 2, 0)), (16, 16))
            resized_image = resized_image.transpose((2, 0, 1))
            new_images[i] = resized_image
print(f"处理完成并保存到新的H5文件中：{new_file_path}")