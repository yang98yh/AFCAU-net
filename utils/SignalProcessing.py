import numpy as np
import math
from scipy import signal
import segyio

def read_sgy(filename):
    print("### Reading SEGY-formatted Seismic Data:")
    print("Data file-->[%s]" %(filename))
    with segyio.open(filename, "r", ignore_geometry=True)as f:
        f.mmap()
        data = np.asarray([np.copy(x) for x in f.trace[:]]).T
    f.close()
    return data

def compare_SNR(recov_img, real_img):
    real_mean = np.mean(real_img)
    tmp1 = real_img - real_mean
    real_var = sum(sum(tmp1*tmp1))

    noise = real_img - recov_img
    noise_mean = np.mean(noise)
    tmp2 = noise - noise_mean
    noise_var = sum(sum(tmp2*tmp2))

    if noise_var == 0 or real_var==0:
      s = 999.99
    else:
      s = 10*math.log(real_var/noise_var, 10)
    return s

def batch_snr(de_data, clean_data):
    De_data = de_data.data.cpu().numpy()
    Clean_data = clean_data.data.cpu().numpy()
    SNR = 0
    for i in range(De_data.shape[0]):
        De = De_data[i, :, :, :].squeeze()
        Clean = Clean_data[i, :, :, :].squeeze()
        SNR += compare_SNR(De, Clean)
    return SNR / De_data.shape[0]

def mse(signal, noise_data):
    signal = np.array(signal)
    noise_data = np.array(noise_data)
    m = np.sum((signal - noise_data) ** 2)
    m = m / m.size
    return m

def psnr(signal, noise_data):
    signal = np.array(signal)
    noise_data = np.array(noise_data)
    psnr = 2 * 10 * math.log10(abs(signal.max()) / np.sqrt(np.sum((signal - noise_data) ** 2) / noise_data.size))
    return psnr

def fft_spectrum(Signal, SampleRate):
    fft_len = Signal.size
    SignalFFT = np.fft.rfft(Signal) / fft_len
    SignalFreqs = np.linspace(0, SampleRate/2, int(fft_len/2)+1)
    SignalAmplitude = np.abs(SignalFFT) * 2
    return SignalFreqs, SignalAmplitude

def butter_lowpass(cutoff, sample_rate, order=4):
    rate = sample_rate * 0.5
    normal_cutoff = cutoff / rate
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(noise_data, cutoff, sample_rate, order=4):
    b, a = butter_lowpass(cutoff, sample_rate, order=order)
    clear_data = signal.filtfilt(b, a, noise_data)
    return clear_data

def butter_bandpass(lowcut, highcut, sample_rate, order=4):
    rate = sample_rate * 0.5
    low = lowcut / rate
    high = highcut / rate
    b, a = signal.butter(order, [low, high], btype='bandpass', analog=False)
    return b, a

def bandpass_filter(noise_data, lowcut, highcut, sample_rate, order=4):
    b, a = butter_bandpass(lowcut, highcut, sample_rate, order=order)
    clear_data = signal.filtfilt(b, a, noise_data)
    return clear_data

def butter_highpass(cutup, sample_rate, order=4):
    rate = sample_rate * 0.5
    normal_cutup = cutup / rate
    b, a = signal.butter(order, normal_cutup, btype='high', analog=False)
    return b, a

def highpass_filter(noise_data, cutup, sample_rate, order=4):
    b, a = butter_highpass(cutup, sample_rate, order=order)
    clear_data = signal.filtfilt(b, a, noise_data)
    return clear_data

def mide_filter(x,kernel_size=5):
    x1 = x.reshape(x.size)
    y = signal.medfilt(x1, kernel_size=kernel_size)
    return y

def fk_spectra(data, dt, dx, L=6):
    data = np.array(data)
    [nt, nx] = data.shape
    i = 0
    while (2 ** i) <= nx:
        i = i + 1
    nk = 4 * 2 ** i
    j = 0
    while (2 ** j) <= nt:
        j = j + 1
    nf = 4 * 2 ** j
    S = np.fft.fftshift(abs(np.fft.fft2(data, (nf, nk))))
    H1 = np.hamming(L)
    H = (H1.reshape(L, -1)) * (H1.reshape(1, L))
    S = signal.convolve2d(S, H, boundary='symm', mode='same')
    S = S[nf // 2:nf, :]
    f = np.arange(0, nf / 2, 1)
    f = f / nf / dt
    k = np.arange(-nk / 2, nk / 2, 1)
    k = k / nk / dx
    return S, k, f

