import torch
import matplotlib.pyplot as plt
import pickle as pk
import numpy as np
from tqdm import tqdm
from scipy.signal import butter, lfilter
import h5py

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

nz = 1200  # length of noise
ngpu = 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device )
epoch = 35
final = []
netG = torch.load('model_dir_af/netG_af_35.pkl')

for tms in tqdm(range(400)):
    fixed_noise = torch.randn(1000, nz, 1, device=device)
    fake = netG(fixed_noise).detach().cpu().numpy()
    fake = fake.reshape((fake.shape[0],fake.shape[2]))
    new_fake = []
    for sig in fake:
        # new_sig = butter_bandpass_filter(sig,lowcut=0.9,highcut=5,fs=40,order=2)
        new_sig = sig
        new_sig = new_sig - min(new_sig)/(max(new_sig) - min(new_sig))
        new_fake.append(new_sig)
    new_fake = np.array(new_fake)
    final.append(new_fake)

final = np.vstack(final)

with h5py.File('AF'+str(epoch)+'_40w.pk', "w") as hf:
    hf.create_dataset("dataset_1", data=final)
# f, a = plt.subplots(4, 4, figsize=(35, 8))
# for i in range(4):
#     for j in range(4):
#         a[i][j].plot(fake[i * 4 + j].view(-1))
#         a[i][j].set_xticks(())
#         a[i][j].set_yticks(())
# plt.show()