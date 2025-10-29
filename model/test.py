import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
from utils.dataset import MyDataset
from utils.SignalProcessing import batch_snr, psnr
from model.AFCAUnet import Unet


batch_size = 16
test_path_x = "..\\data\\test_patch\\sam\\"
test_path_y = "..\\data\\test_patch\\ori\\"
test_dataset = MyDataset(test_path_x, test_path_y)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model = Unet()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
temp_sets = []  
psnr_sets = [] 
file_list = glob.glob(os.path.join("save_dir2\\", '*pth'))
snr_set1 = 0.0
psnr_set1 = 0.0

for i in range(len(file_list)):
    state_dict = torch.load(file_list[i])
    model.load_state_dict(state_dict)
    model.to(device=device) 
    model.eval() 

    snr_set2 = 0.0
    psnr_set2 = 0.0
    for batch_idx, (test_x, test_y) in enumerate(test_loader, 0):
        test_x = test_x.to(device=device, dtype=torch.float32)
        test_y = test_y.to(device=device, dtype=torch.float32)

        with torch.no_grad():  
            out = model(test_x)  

            test_x_cpu = test_x.cpu()
            test_y_cpu = test_y.cpu()
            out_cpu = out.cpu()

            if i < 1: 
                SNR1 = batch_snr(test_x_cpu, test_y_cpu) 
                PSNR1 = psnr(test_x_cpu, test_y_cpu) 
                snr_set1 += SNR1
                psnr_set1 += PSNR1

            SNR2 = batch_snr(out_cpu, test_y_cpu)  
            PSNR2 = psnr(out_cpu, test_y_cpu) 
            snr_set2 += SNR2
            psnr_set2 += PSNR2

    if i < 1: 
        snr_set1 = snr_set1 / (batch_idx + 1)
        psnr_set1 = psnr_set1 / (batch_idx + 1)
    snr_set2 = snr_set2 / (batch_idx + 1)
    psnr_set2 = psnr_set2 / (batch_idx + 1)

    temp_sets.append(snr_set2)
    psnr_sets.append(psnr_set2)

    print("epoch={}，去噪前的平均信噪比(SNR)：{:.4f} dB，去噪后的平均信噪比(SNR)：{:.4f} dB".format(i + 1, snr_set1, snr_set2))
    print("epoch={}，去噪前的平均PSNR：{:.4f} dB，去噪后的平均PSNR：{:.4f} dB".format(i + 1, psnr_set1, psnr_set2))

np.savetxt('result2/snr_sets.txt', temp_sets, fmt='%.4f')
np.savetxt('result2/psnr_sets.txt', psnr_sets, fmt='%.4f')

res_snr = np.loadtxt('result2/snr_sets.txt')
x = range(len(res_snr))
fig1 = plt.figure()
plt.plot(x, res_snr, label='Denoise SNR', color='b')
plt.xlabel('Epoch')
plt.ylabel('SNR (dB)')
plt.legend()
plt.title('SNR')
plt.savefig('./result2/snr_plot.png', bbox_inches='tight')
plt.tight_layout()
plt.show()

res_psnr = np.loadtxt('result2/psnr_sets.txt')
fig2 = plt.figure()
plt.plot(x, res_psnr, label='Denoise PSNR', color='r')
plt.xlabel('Epoch')
plt.ylabel('PSNR (dB)')
plt.legend()
plt.title('PSNR')
plt.savefig('./result2/psnr_plot.png', bbox_inches='tight')
plt.tight_layout()
plt.show()
