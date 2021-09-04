import numpy as np
from torch.autograd.variable import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F

def _torch_welch(data, fs=40.0, nperseg=400, noverlap=None, average='mean',
                 device='cpu'):
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
    def __init__(self,fs=40,window_length = 400,noverlap = None,device = 'cpu'):
        super().__init__()
        self.fs = fs
        self.welch = lambda x: _torch_welch(
            x, fs=fs, nperseg=window_length, noverlap=noverlap, device=device)
        Tensor = torch.FloatTensor if device=='cpu' else torch.cuda.FloatTensor
        self.kernel = Variable(Tensor(torch.rand(1, 1, 3,device = device)), requires_grad=True)

    def forward(self,fake,real):

        b_size = fake.shape[0]
        fake = fake.view(b_size,-1)
        real = real.view(b_size,-1)
        fake_freq = self.welch(fake)
        real_freq = self.welch(real)
        fake_freq = torch.log(self.welch(fake))
        real_freq = torch.log(self.welch(real))

        min_real = torch.min(real_freq, dim=1)[0].view(b_size, 1)
        range_real = torch.max(real_freq, dim=1)[0].view(b_size, 1) - min_real
        normalised_real = (real_freq - min_real) / range_real
        real_freq = normalised_real.view(b_size, -1)

        # fake_freq = fake_freq.view(b_size,1,-1)

        # fake_pool = F.max_pool1d(F.conv1d(fake_freq,self.kernel),2)
        # real_pool = F.max_pool1d(F.conv1d(real_freq,self.kernel),2)
        # fake_freq= F.upsample(fake_pool,size=real_freq.shape[1])

        fake_freq = fake_freq.view(b_size, -1)
        min_fake = torch.min(fake_freq, dim=1)[0].view(b_size, 1)
        range_fake = torch.max(fake_freq, dim=1)[0].view(b_size, 1) - min_fake
        normalised_fake = (fake_freq - min_fake) / range_fake
        fake_freq = normalised_fake.view(b_size, -1)

        l2_loss = nn.MSELoss()
        psd_loss = l2_loss(fake_freq,real_freq)
        return psd_loss


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