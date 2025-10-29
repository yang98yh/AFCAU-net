import glob
import numpy as np
import segyio
import random
import os


random.seed(2024)
np.random.seed(2024)


def read_segy_data(filename):
    print("### Reading SEGY-formatted Seismic Data:")
    print("Data file-->[%s]" % (filename))
    with segyio.open(filename, "r", ignore_geometry=True) as f:
        f.mmap()
        data = np.asarray([np.copy(x) for x in f.trace[:]]).T
    f.close()
    return data

def gen_patches(file_path, patch_size, stride_x, stride_y):
    shot_data = np.load(file_path)
    time_sample, trace_number = shot_data.shape
    patches = []

    strides_x = []
    if stride_x < 0:
        raise ValueError("步长stride_x不能为负数。")
    else:
        if trace_number >= patch_size:
            num_x = ((trace_number - patch_size) // stride_x) + 1
            for i in range(num_x):
                strides_x.append(i * stride_x)
            if trace_number - strides_x[-1] - patch_size >= 0:
                strides_x.append(trace_number - patch_size)
            elif trace_number - strides_x[-1] > 0:
                strides_x.append(trace_number - patch_size)
        else:
            print("切片大小超出单炮数据尺寸！strides_x为空")

    strides_y = []
    if stride_y < 0:
        raise ValueError("步长stride_y不能为负数。")
    else:
        if time_sample >= patch_size:
            num_y = ((time_sample - patch_size) // stride_y) + 1
            for i in range(num_y):
                strides_y.append(i * stride_y)
            if time_sample - strides_y[-1] - patch_size >= 0:
                strides_y.append(time_sample - patch_size)
            elif time_sample - strides_y[-1] > 0:
                strides_y.append(time_sample - patch_size)
        else:
            print("切片大小超出单炮数据尺寸！strides_y为空")

    for index_x in strides_x:
        for index_y in strides_y:
            if index_y + patch_size <= time_sample and index_x + patch_size <= trace_number:
                patch = shot_data[index_y: index_y + patch_size, index_x: index_x + patch_size]
                if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                    patches.append(patch)

    return patches

def data_generator(data_dir, patch_size, stride_x, stride_y):
    file_list = glob.glob(os.path.join(data_dir, '*npy'))
    data = []
    for i in range(len(file_list)):
        patches = gen_patches(file_list[i], patch_size, stride_x, stride_y)
        for patch in patches:
            data.append(patch)
    return data


if __name__ == "__main__":

    data_dir1 = '..\\data\\npy_shot\\ori_shot'
    data_dir2 = '..\\data\\npy_shot\\sam_shot'
    patch_size = 128
    xs = data_generator(data_dir2, patch_size, 32, 64)
    ys = data_generator(data_dir1, patch_size, 32, 64)

    train_sam_dir = '..\\data\\train_patch\\sam\\'
    train_ori_dir = '..\\data\\train_patch\\ori\\'
    valid_sam_dir = '..\\data\\valid_patch\\sam\\'
    valid_ori_dir = '..\\data\\valid_patch\\ori\\'
    test_sam_dir = '..\\data\\test_patch\\sam\\'
    test_ori_dir = '..\\data\\test_patch\\ori\\'

    os.makedirs(train_sam_dir, exist_ok=True)
    os.makedirs(train_ori_dir, exist_ok=True)
    os.makedirs(valid_sam_dir, exist_ok=True)
    os.makedirs(valid_ori_dir, exist_ok=True)
    os.makedirs(test_sam_dir, exist_ok=True)
    os.makedirs(test_ori_dir, exist_ok=True)

    random.seed(2024)
    np.random.seed(2024)

    len_x1 = len(xs)
    patches_size = [k for k in range(len_x1)]
    random.shuffle(patches_size)

    train_count = 0
    valid_count = 0
    test_count = 0

    for num, j in enumerate(patches_size):
        sam_patch = xs[j]
        ori_patch = ys[j]
        sam_name = f'sam{num + 1}'
        ori_name = f'ori{num + 1}'

        if num <= int(0.8 * len_x1):
            np.save(train_sam_dir + sam_name, sam_patch)
            np.save(train_ori_dir + ori_name, ori_patch)
            train_count += 1

        elif int(0.8 * len_x1) < num <= int(0.9 * len_x1):
            np.save(valid_sam_dir + sam_name, sam_patch)
            np.save(valid_ori_dir + ori_name, ori_patch)
            valid_count += 1

        else:
            np.save(test_sam_dir + sam_name, sam_patch)
            np.save(test_ori_dir + ori_name, ori_patch)
            test_count += 1

    print(f'训练集样本数: {train_count} 个')
    print(f'验证集样本数: {valid_count} 个')
    print(f'测试集样本数: {test_count} 个')
    print(f'总样本数: {len(xs)} 个')
