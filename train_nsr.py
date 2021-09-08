import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
# from models import Discriminator, Generator, weights_init
from models_1200 import Discriminator,Generator,weights_init
from PPGDataset_40hz import PPGDataloader
from scipy.signal import butter, lfilter,welch
import torch
from torch.autograd.variable import Variable
import os
from loss_functions import PSDShift,AutoCorrelationLoss,autocorrelation_function
import torch.nn.functional as F
from loss_functions import _torch_welch
import warnings
import numpy as np
from PPGDataset_40hz import PPGDataloader
from preprocessing import Dataset

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

lr = 5e-4
weight_decay = 0
beta1 = 0.5
epoch_num = 200
batch_size = 1000
nz = 1200  # length of noise
ngpu = 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")
adversarial_loss = nn.BCELoss()
PSDloss = PSDShift(device=device)#window_length=1200,noverlap=0,
MsEloss = nn.MSELoss()
lag_sec = 1.5
fs = 40
ACLoss = AutoCorrelationLoss(int(lag_sec*fs),device=device)

def main(delta):

    model_dir = './model_dir_nsr_test_plot/'
    image_dir = './image_dir_nsr_test_plot/'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if not os.path.exists(image_dir):
        os.mkdir(image_dir)
    # load training data

    trainset = Dataset()

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )
    # init netD and netG
    netD = Discriminator().to(device)
    netD.apply(weights_init)

    netG = Generator().to(device)
    netG.apply(weights_init)

    # used for visualzing training process
    fixed_noise = torch.randn(16, nz, 1, device=device)

    if torch.cuda.is_available():
        cuda = True
        print('Using: ' + str(torch.cuda.get_device_name(device)))
    else:
        cuda = False
        print('Using: CPU')

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    optimizerD = optim.Adam(netD.parameters(), lr=lr,weight_decay=weight_decay, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr,weight_decay=weight_decay,betas=(beta1, 0.999))

    for epoch in range(epoch_num):
        BCE_loss = 0.0
        PSD_loss = 0.0
        for step, (data, _) in enumerate(trainloader):
            # print(step)
            perm = torch.randperm(data.size(0))
            idx = perm[:16]
            sample_data = data[idx]

            # Adversarial GT
            label = Variable(Tensor(data.size(0), 2).fill_(1.0), requires_grad=False)
            fake_label = Variable(Tensor(data.size(0), 2).fill_(0.0), requires_grad=False)

            real_cpu = data.to(device)
            b_size = real_cpu.size(0)

            # train netD

            netD.zero_grad()
            output = netD(real_cpu)
            errD_real = adversarial_loss(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # train netG
            noise = torch.randn(b_size, nz, 1, device=device)
            fake = netG(noise)
            output = netD(fake.detach())
            errD_fake = adversarial_loss(output, fake_label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            psd_loss = PSDloss(fake, real_cpu)
            # ac_loss = ACLoss(fake,real_cpu)
            netG.zero_grad()
            output = netD(fake)
            ad_loss = adversarial_loss(output, label)
            errG = ad_loss + delta*psd_loss
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            BCE_loss += ad_loss.item()
            PSD_loss += psd_loss.item()

        print('[%d/%d]\tCE_Loss: %.4f\tPSD_Loss: %.4f\t'%(epoch,epoch_num,BCE_loss/len(trainloader),PSD_loss/len(trainloader)))

        perm = torch.randperm(data.size(0))
        idx = perm[:4]
        real_data = data[idx].detach().cpu()

        fake_data = fake[idx].detach().cpu()
        plot_figure(real_data,fake_data,save_dir= image_dir + '/nsr_epoch_%d.png' % epoch)
        # save models
        torch.save(netG, model_dir+'/netG_nsr_%d.pkl' % epoch)
        torch.save(netD, model_dir+'/netD_nsr_%d.pkl' % epoch)


def plot_figure(real,fake,save_dir):
    # save training process
        f, a = plt.subplots(4, 6, figsize=(80, 15))
        a[0][0].plot(fake[0].view(1200,))
        a[0][1].plot(fake[1].view(1200,))
        a[0][2].plot(10*torch.log(_torch_welch(fake[0].view(1,1200),fs=40,nperseg=400)[0]))
        a[0][3].plot(10*torch.log(_torch_welch(fake[1].view(1,1200),fs=40,nperseg=400)[0]))
        a[0][4].plot(autocorrelation_function(fake[0].view(1200,),200))
        a[0][5].plot(autocorrelation_function(fake[1].view(1200,),200))


        a[1][0].plot(fake[2].view(1200,))
        a[1][1].plot(fake[3].view(1200,))
        a[1][2].plot(10 * torch.log(_torch_welch(fake[2].view(1,1200), fs=40, nperseg=400)[0]))
        a[1][3].plot(10 * torch.log(_torch_welch(fake[3].view(1,1200), fs=40, nperseg=400)[0]))
        a[1][4].plot(autocorrelation_function(fake[2].view(1200,), 200))
        a[1][5].plot(autocorrelation_function(fake[3].view(1200,), 200))


        a[2][0].plot(real[0].view(1200,))
        a[2][1].plot(real[1].view(1200,))
        a[2][2].plot(10 * torch.log(_torch_welch(real[0].view(1,1200), fs=40, nperseg=400)[0]))
        a[2][3].plot(10 * torch.log(_torch_welch(real[1].view(1,1200), fs=40, nperseg=400)[0]))
        a[2][4].plot(autocorrelation_function(real[0].view(1200,), 200))
        a[2][5].plot(autocorrelation_function(real[1].view(1200,), 200))

        a[3][0].plot(real[2].view(1200,))
        a[3][1].plot(real[3].view(1200,))
        a[3][2].plot(10 * torch.log(_torch_welch(real[2].view(1,1200), fs=40, nperseg=400)[0]))
        a[3][3].plot(10 * torch.log(_torch_welch(real[3].view(1,1200), fs=40, nperseg=400)[0]))
        a[3][4].plot(autocorrelation_function(real[2].view(1200,), 200))
        a[3][5].plot(autocorrelation_function(real[3].view(1200,), 200))
        plt.savefig(save_dir)
        plt.close()

if __name__ == '__main__':
    main(1)
    # for delta in np.linspace(3.1, 5.0, num=20, endpoint=True):
    #     print(delta)
    #     main(delta)
    #     print('-----------------------------------------------')
