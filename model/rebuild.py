import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from datetime import datetime
from utils.SignalProcessing import compare_SNR, psnr
from model.AFCAUnet import Unet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    print(f"Using GPU: {torch.cuda.current_device()}")
    print(f"GPU Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("Using CPU")


path_x = "..\\data\\test_patch\\sam\\sam7060.npy"  
path_y = "..\\data\\test_patch\\ori\\ori7060.npy"  

test_x = np.load(path_x) 
test_y = np.load(path_y) 

model = Unet1()
model.load_state_dict(torch.load('bm/best_model2.pth')) 
model.to(device=device) 
model.eval()  

patch = test_x
patch = patch.reshape(1, 1, patch.shape[0], patch.shape[1]) 
patch = torch.from_numpy(patch) 
patch = patch.to(device=device, dtype=torch.float32) 

out_data = model(patch) 
out_data = out_data.data.cpu().numpy()
out_data = out_data.squeeze() 

snr_set1 = compare_SNR(test_y, test_x)
psnr_set1 = psnr(test_x, test_y) 

snr_set2 = compare_SNR(test_y, out_data) 
psnr_set2 = psnr(out_data, test_y)


print(f"SNR before denoising : {snr_set1:.4f} dB")
print(f"SNR after denoising : {snr_set2:.4f} dB")
print(f"PSNR before denoising : {psnr_set1:.4f} dB")
print(f"PSNR after denoising : {psnr_set2:.4f} dB")

residual = out_data - test_y
residual_max = residual.max()
residual_min = residual.min()
print(f"Residual Max: {residual_max:.4f}, Residual Min: {residual_min:.4f}")

os.makedirs('result2', exist_ok=True) 
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
save_path = f'result2/Reconstruction_{timestamp}.npy'
np.save(save_path, out_data)
print(f"Reconstructed data saved at {save_path}")

text_save_path = f'result2/SNR_PSNR_Residual_{timestamp}.txt'
with open(text_save_path, 'w') as f:
    f.write("### Denoising Results ###\n")
    f.write(f"SNR before denoising : {snr_set1:.4f} dB\n")
    f.write(f"SNR after denoising : {snr_set2:.4f} dB\n")
    f.write(f"PSNR before denoising : {psnr_set1:.4f} dB\n")
    f.write(f"PSNR after denoising : {psnr_set2:.4f} dB\n")
    f.write("\n### Residual Information ###\n")
    f.write(f"Residual Max: {residual_max:.4f}\n")
    f.write(f"Residual Min: {residual_min:.4f}\n")

plt.figure(figsize=(8, 6))
plt.imshow(test_x, cmap='seismic', interpolation='nearest', aspect=1, vmin=-0.1, vmax=0.1)
plt.title('Sample Signal')
plt.savefig('./result2/sample_signal.png', bbox_inches='tight')
plt.show()

plt.figure(figsize=(8, 6))
plt.imshow(out_data, cmap='seismic', interpolation='nearest', aspect=1, vmin=-0.1, vmax=0.1)
plt.title('Rebuild Signal')
plt.savefig('./result2/rebuild_signal.png', bbox_inches='tight')
plt.show()

plt.figure(figsize=(8, 6))
plt.imshow(test_y, cmap='seismic', interpolation='nearest', aspect=1, vmin=-0.1, vmax=0.1)
plt.title('Original Signal')
plt.savefig('./result2/original_signal.png', bbox_inches='tight')
plt.show()

plt.figure(figsize=(8, 6))
plt.imshow(residual, cmap='seismic', interpolation='nearest', aspect=1, vmin=-0.1, vmax=0.1)
plt.title('Residual')
plt.savefig('./result2/residual_signal.png', bbox_inches='tight')
plt.show()
