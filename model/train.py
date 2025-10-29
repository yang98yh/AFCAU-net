from model.AFCAUnet import Unet
from utils.dataset import MyDataset
from torch import optim
import torch.nn as nn
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    print(f"Using GPU: {torch.cuda.current_device()}")
    print(f"GPU Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("Using CPU")


my_net = Unet()
my_net.to(device=device)
train_path_x = "..\\data\\train_patch\\sam\\"
train_path_y = "..\\data\\train_patch\\ori\\"
valida_path_x = "..\\data\\valid_patch\\sam\\"
valida_path_y = "..\\data\\valid_patch\\ori\\"


batch_size = 16
train_dataset = MyDataset(train_path_x, train_path_y)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
valida_dataset = MyDataset(valida_path_x, valida_path_y)
valida_loader = torch.utils.data.DataLoader(dataset=valida_dataset, batch_size=batch_size, shuffle=True)

epochs = 100 
LR = 0.001  
optimizer = optim.Adam(my_net.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)  
criterion = nn.MSELoss(reduction='sum') 
temp_sets = [] 
start_time = time.strftime("1. %Y-%m-%d %H:%M:%S", time.localtime())  

best_val_loss = float('inf')
best_epoch = 0  

for epoch in range(epochs):
    train_loss = 0.0
    my_net.train() 
    for batch_idx1, (batch_x, batch_y) in enumerate(train_loader, 0): 
        batch_x = batch_x.to(device=device, dtype=torch.float32)
        batch_y = batch_y.to(device=device, dtype=torch.float32)
        optimizer.zero_grad()  
        out1 = my_net(batch_x) 
        loss1 = criterion(out1, batch_y)
        train_loss += loss1.item()  
        loss1.backward()
        optimizer.step()
    train_loss = train_loss / (batch_idx1 + 1)
    scheduler.step()

    my_net.eval() 
    val_loss = 0.0
    for batch_idx2, (val_x, val_y) in enumerate(valida_loader, 0):
        val_x = val_x.to(device=device, dtype=torch.float32)
        val_y = val_y.to(device=device, dtype=torch.float32)
        with torch.no_grad():  
            out2 = my_net(val_x) 
            loss2 = criterion(out2, val_y)
            val_loss += loss2.item() 
    val_loss = val_loss / (batch_idx2 + 1)

    loss_set = [train_loss, val_loss]
    temp_sets.append(loss_set)
    print(f"epoch={epoch + 1}，训练集loss：{train_loss:.4f}，验证集loss：{val_loss:.4f}")

    model_name = f'model_epoch{epoch + 1}' 
    torch.save(my_net.state_dict(), os.path.join('save_dir2', model_name + '.pth')) 

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        best_model_name = 'best_model2.pth'
        torch.save(my_net.state_dict(), os.path.join('bm', best_model_name)) 

print(f"最优模型出现在第 {best_epoch + 1} epoch，验证集loss为 {best_val_loss:.4f}")

end_time = time.strftime("1. %Y-%m-%d %H:%M:%S", time.localtime()) 
with open('result2/训练时间.txt', 'w', encoding='utf-8') as f:
    f.write(start_time)
    f.write(end_time)
    f.close()

with open('result2/训练时间.txt', 'a', encoding='utf-8') as f:
    f.write(f"最优模型出现在第 {best_epoch + 1} epoch，验证集loss为 {best_val_loss:.4f}\n")

print(f"训练开始时间 {start_time} >>>>>>>>>>>>>> 训练结束时间 {end_time}") 

loss_sets = []
for sets in temp_sets:
    for i in range(2):
        loss_sets.append(sets[i])
loss_sets = np.array(loss_sets).reshape(-1, 2) 
np.savetxt('result2/loss_sets.txt', loss_sets, fmt='%.4f')

loss_lines = np.loadtxt('result2/loss_sets.txt')
train_line = loss_lines[:, 0] / batch_size
valida_line = loss_lines[:, 1] / batch_size
x1 = range(len(train_line))
fig1 = plt.figure()
plt.plot(x1, train_line, x1, valida_line)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'valida'])
plt.savefig('./result2/loss_plot.png', bbox_inches='tight')
plt.tight_layout()
plt.show()
