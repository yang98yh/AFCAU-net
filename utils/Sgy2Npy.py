import numpy as np
from GetPatches import read_segy_data
import random
import os

file_path1 = os.path.join("..\\data\\", "BP2007.sgy")


sgy_data = read_segy_data(file_path1)
shot_num = 50
traces = 800
shot_index = [k for k in range(shot_num)]
random.shuffle(shot_index)
random.seed(2024)
np.random.seed(2024)

save_dir1 = '..\\data\\npy_shot\\ori_shot'
save_dir2 = '..\\data\\npy_shot\\sam_shot'

if not os.path.exists(save_dir1):
    os.makedirs(save_dir1)
if not os.path.exists(save_dir2):
    os.makedirs(save_dir2)
for num, k in enumerate(shot_index):
    ori_data = sgy_data[:, k * int(traces):(k + 1) * int(traces)]
    max_data = abs(ori_data.max())
    ori_data = ori_data / max_data
    mask = np.ones(traces) 
    rates = [0.7] 
    rate = random.sample(rates, 1)  
    mask[:int(rate[0] * len(mask))] = 0  
    np.random.shuffle(mask)  
    mask.reshape(-1, len(mask))  
    sam_data = mask * ori_data
    ori_name = f'ori{num+1}.npy'  
    sam_name = f'sam{num+1}.npy'

    np.save(os.path.join(save_dir1, ori_name), ori_data)  
    np.save(os.path.join(save_dir2, sam_name), sam_data) 
