import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import torch

# --- Load files ---
user = sio.loadmat("user1-1-1-5-7-r5.mat")
pred = sio.loadmat("0-0.mat")
fs = 40e6

user_data = torch.from_numpy(user['feature']).to(torch.complex64)
pred_data = torch.from_numpy(pred['pred']).to(torch.complex64)

# --- FFT ---
def fft_spectrum(x, sampling_rate):
    fft_result = np.fft.fft(x)
    frequencies = np.fft.fftfreq(len(x), d=1/sampling_rate)
    shifted_fft_result = np.fft.fftshift(fft_result)
    shifted_frequencies = np.fft.fftshift(frequencies)

    return shifted_frequencies, shifted_fft_result

f_user_shifted, f_user_shifted_result = fft_spectrum(user_data[0, :], fs)
f_pred_shifted, f_pred_shifted_result = fft_spectrum(pred_data[0, 0, :], fs)

plt.figure(figsize=(10, 6))
plt.plot(f_user_shifted, np.abs(f_user_shifted_result))
plt.title('FFT Magnitude Spectrum Data')
plt.xlabel('Frequency (MHz)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(f_pred_shifted, np.abs(f_pred_shifted_result))
plt.title('FFT Magnitude Spectrum Prediction')
plt.xlabel('Frequency (MHz)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.show()


power_db = 20 * np.log10(np.abs(pred_data[0, :, :]) + 1e-6)

plt.figure(figsize=(10,6))
plt.imshow(power_db, aspect='auto', origin='lower',
           extent=[0, 40, 0, pred_data.shape[1]/100])  # 0–40 MHz, time in s
plt.xlabel('Frequency (MHz)')
plt.ylabel('Time (s)')
plt.title('CSI Waterfall Plot')
plt.colorbar(label='Magnitude (dB)')
plt.show()