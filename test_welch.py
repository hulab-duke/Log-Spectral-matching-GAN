from loss_functions import _torch_welch
from scipy.signal import welch
from PPGDataset_40hz import PPGDataloader
import torch
import matplotlib.pyplot as plt
import numpy as np

def read_csv(file_path):
    f = open(file_path)
    content = f.readline().split(',')
    output = []
    for item in content:
        output.append(float(item))
    f.close()
    output = np.array(output)
    output.reshape((1,-1))
    return output



data_path = 'path-to-file'
data = read_csv(data_path)[0:-3]
data_torch = torch.tensor(data, dtype = torch.float)
data_torch = data_torch.view(1200,)

tmp_welch = _torch_welch(data_torch.view(1,1200),nperseg=400,fs=40)
scipy_welch = welch(data,fs=40,nperseg=400)[1]
f, a = plt.subplots(3, 1)
a[0].plot(data)
a[1].plot(10*torch.log(tmp_welch.view(201,)))
a[2].plot(10*torch.log(torch.tensor(scipy_welch)))
plt.show()
