import numpy as np
from torch.autograd.variable import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
def _torch_welch(data, fs=40.0, nperseg=400, noverlap=None, average='mean',
                 device='cpu',return_list = False):
    """ Compute PSD using Welch's method.
    """
    if len(data.shape) > 2:
        data = data.view(data.shape[0], -1)
    N, nsample = data.shape

    # Get parameters
    if noverlap is None:
        noverlap = nperseg // 2
    nstride = nperseg - noverlap
    nseg = int(np.ceil((nsample - nperseg) / nstride)) + 1
    nfreq = nperseg // 2 + 1
    T = nsample * fs

    # Calculate the PSD
    psd = torch.zeros((nseg, N, nfreq)).to(device)
    window = torch.hann_window(nperseg).to(device) * 2

    # calculate the FFT amplitude of each segment
    for i in range(nseg):
        seg_ts = data[:, i * nstride:i * nstride + nperseg] * window
        seg_fd = torch.rfft(seg_ts, 1)
        seg_fd_abs = (seg_fd[:, :, 0] ** 2 + seg_fd[:, :, 1] ** 2)
        psd[i] = seg_fd_abs

    if return_list:
        # for i in range(nseg):
        #     psd[i]/=T
        return psd
    else:
        # taking the average
        if average == 'mean':
            psd = torch.sum(psd, 0)
        elif average == 'median':
            psd = torch.median(psd, 0)[0] * nseg
        else:
            raise ValueError(f'average must be "mean" or "median", got {average} instead')

        # Normalize
        psd /= T
        return psd

def js_div(p_logits, q_logits, get_softmax=True):
    """
    Function that measures JS divergence between target and output logits:
    """
    KLDivLoss = nn.KLDivLoss(reduction='batchmean')
    if get_softmax:
        p_output = F.softmax(p_logits)
        q_output = F.softmax(q_logits)
    log_mean_output = ((p_logits + q_logits )/2).log()
    return (KLDivLoss(log_mean_output, p_logits) + KLDivLoss(log_mean_output, q_logits))/2

class PSDShift(nn.Module):
    def __init__(self,fs=40,window_length = 100,noverlap = 0,device = 'cpu',intergration='mean'):
        super().__init__()
        self.device = device
        self.fs = fs
        self.welch = lambda x: _torch_welch(
            x, fs=fs, nperseg=window_length, noverlap=noverlap, device=device,return_list=False)
        self.psd_list = lambda x: _torch_welch(
            x, fs=fs, nperseg=window_length, noverlap=noverlap, device=device, return_list=True)
        self.Tensor = torch.FloatTensor if device=='cpu' else torch.cuda.FloatTensor
        self.intergration = intergration
        if self.intergration not in ['mean','max']:
            raise ValueError(f'average must be "mean" or "max", got {self.intergration} instead')
        # self.kernel = Variable(Tensor(torch.rand(1, 1, 3,device = device)), requires_grad=True)

    def forward(self,fake,real):
        min_value = 1e-6
        b_size = fake.shape[0]
        fake = torch.clamp(fake.view(b_size,-1),min = min_value)
        real = torch.clamp(real.view(b_size,-1),min = min_value)

        fake_psd_list = torch.log(self.psd_list(fake))
        real_psd_list = torch.log(self.psd_list(real))

        sign = fake_psd_list.sign()
        fake_psd_list = fake_psd_list.abs_().clamp_(min=min_value)
        fake_psd_list *= sign
        sign = real_psd_list.sign()
        real_psd_list = real_psd_list.abs_().clamp_(min=min_value)
        real_psd_list *= sign

        for index,sig in enumerate(fake_psd_list):
            sig = sig.view(-1,b_size)
            tmp = (sig - torch.min(sig,dim=0)[0])/(torch.max(sig,dim=0)[0] - torch.min(sig,dim=0)[0])
            fake_psd_list[index] = tmp.view(b_size,-1)

        for index,sig in enumerate(real_psd_list):
            sig = sig.view(-1, b_size)
            tmp = (sig - torch.min(sig, dim=0)[0]) / (torch.max(sig, dim=0)[0] - torch.min(sig, dim=0)[0])
            real_psd_list[index] = tmp.view(b_size, -1)

        l2_loss = nn.MSELoss()
        l1_loss = nn.L1Loss()
        psd_loss = torch.zeros((len(fake_psd_list)**2, 1),requires_grad=True).to(self.device)
        count = 0
        count_self = 0
        m = len(fake_psd_list)
        m = int(m*(m-1)/2)
        psd_loss_fake = torch.zeros((m, 1),requires_grad=True).to(self.device)

        for index_fake in range(1,len(fake_psd_list)+1):
            for index_real in range(1,len(real_psd_list)+1):
                fake_seg = fake_psd_list[index_fake - 1]
                real_seg = real_psd_list[index_real - 1]
                # fake_seg = torch.clamp(fake_seg, min=min_value)
                # real_seg = torch.clamp(real_seg, min=min_value)# to avoid getting inf loss
                psd_loss[count] = l2_loss(fake_seg,real_seg)
                if torch.isinf(psd_loss[count]) or torch.isnan(psd_loss[count]):
                    print('bug found')
                count+=1

            for index_fake_self in range(1,len(fake_psd_list)+1):
                if index_fake < index_fake_self:
                    fake_seg_self = fake_psd_list[index_fake_self-1]
                    # fake_seg_self = torch.clamp(fake_seg_self, min=min_value)
                    psd_loss_fake[count_self] = l2_loss(fake_seg,fake_seg_self)
                    count_self+=1


        if self.intergration == 'max':
            psd_loss = torch.max(psd_loss)
            psd_loss_fake_self = torch.max(psd_loss_fake)
        else:
            psd_loss = torch.mean(psd_loss,0)[0]
            psd_loss_fake_self = torch.mean(psd_loss_fake,0)[0]
        return psd_loss,psd_loss_fake_self

# Compute covariance for a  lag
def cov(x, y):
    """
    Compute covariance for a lag
    :param x: Timeseries tensor
    :param y: Timeseries tensor
    :return: The covariance coefficients
    """
    # Average x and y
    x_mu = torch.mean(x, dim=0)
    y_mu = torch.mean(y, dim=0)

    # Average covariance over length
    return torch.mean(torch.mul(x - x_mu, y - y_mu))

def autocorrelation_function(x: torch.Tensor, n_lags: int):
    """
    AutoCorrelation coefficients function for a time series
    @param x: The 1-D timeseries
    @param n_lags: Number of lags
    @return: A 1-D tensor with n_lags+1 components
    """
    # Store coefs
    autocov_coefs = torch.zeros(n_lags+1)

    # Time length for comparison
    com_time_length = x.size(0) - n_lags

    # The time length for comparison must
    # be superior (or equal) to the number of lags required
    if com_time_length < n_lags:
        raise ValueError(
            "Time time length for comparison must "
            "be superior (or equal) to the number of lags required (series of length "
            "{}, {} lags, comparison length of {})".format(x.size(0), n_lags, com_time_length)
        )
    # end if

    # Covariance t to t
    autocov_coefs[0] = cov(x[:com_time_length], x[:com_time_length])

    # For each lag
    for lag_i in range(1, n_lags+1):
        autocov_coefs[lag_i] = cov(
            x[:com_time_length],
            x[lag_i:lag_i + com_time_length]
        )
    # end for

    # Co
    c0 = autocov_coefs[0].item()

    # Normalize with first coef
    autocov_coefs /= c0

    return autocov_coefs

# AutoCorrelation Coefficients for a time series
def autocorrelation_coefs(x: torch.Tensor, n_coefs: int):
    """
    AutoCorrelation Coefficients for a time series
    @param x: A 2D tensor (no batch) or 3D tensor (with batch) -> (batch,signal,channel)
    @param n_coefs: Number of coefficients for each dimension
    @return: A 2D tensor (n. channels x n. coefs) if no batch, 3D tensor (n. batch x n.channels x n. coefs) if batched
    """
    # Has batch?
    use_batch = x.ndim == 3

    # Add batch dim if necessary
    if not use_batch:
        x = torch.unsqueeze(x, dim=0)
    # end if

    # Sizes
    batch_size, time_length, n_channels = x.size()

    # Result collector
    result_collector = torch.zeros(batch_size, n_channels, n_coefs+1)

    # For each batch
    for batch_i in range(batch_size):
        # For each channel
        for channel_i in range(n_channels):
            result_collector[batch_i, channel_i] = autocorrelation_function(x[batch_i, :, channel_i], n_lags=n_coefs)
        # end for
    # end for

    # Return result
    if not use_batch:
        return torch.squeeze(result_collector, dim=0)
    # end if
    return result_collector

class AutoCorrelationLoss(nn.Module):
    def __init__(self,n_coefs,device = 'cpu'):
        super().__init__()
        self.n_coefs = n_coefs

    def forward(self,fake,real):
        b_size = fake.shape[0]
        signal_length = fake.shape[2]
        fake = fake.view(b_size,signal_length,1)
        real = real.view(b_size,signal_length,1)
        fake_ac = autocorrelation_coefs(fake,self.n_coefs)
        real_ac = autocorrelation_coefs(real,self.n_coefs)

        l1_loss = nn.L1Loss()
        ac_loss = l1_loss(fake_ac, real_ac)
        return ac_loss