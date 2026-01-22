import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.signal import spectrogram


# ============================================================
# 1. LOAD IQ DATA
# ============================================================

def load_iq(path, filetype="npy"):
    """
    Load IQ samples from a file.
    Supported: npy, mat, bin
    """

    if filetype == "npy":
        iq = np.load(path)

    elif filetype == "mat":
        mat = sio.loadmat(path)
        # Guess the key
        for k in mat.keys():
            if "iq" in k.lower() or "data" in k.lower():
                iq = mat[k]
                break

    elif filetype == "bin":
        # complex64 raw binary
        raw = np.fromfile(path, dtype=np.complex64)
        iq = raw

    else:
        raise ValueError("Unsupported filetype")

    return np.asarray(iq).flatten()


# ============================================================
# 2. PLOT FUNCTIONS
# ============================================================


def plot_constellation(iq, downsample=10):
    # Downsampling avoids clutter
    real = np.real(iq[::downsample])
    imag = np.imag(iq[::downsample])

    plt.figure(figsize=(5, 5))
    plt.scatter(real, imag, s=3, alpha=0.5)
    plt.title("Constellation Diagram")
    plt.xlabel("In-Phase (I)")
    plt.ylabel("Quadrature (Q)")
    plt.grid()
    plt.axis("equal")
    plt.tight_layout()


def plot_spectrogram(iq, fs=20e6):
    f, t, Sxx = spectrogram(iq, fs=fs, nperseg=256, noverlap=200)

    plt.figure(figsize=(10, 4))
    plt.pcolormesh(t, f/1e6, 10 * np.log10(Sxx), shading="auto")
    plt.ylabel("Frequency [MHz]")
    plt.xlabel("Time [s]")
    plt.title("Spectrogram (OFDM Structure / Preamble Visible)")
    plt.colorbar(label="Power [dB]")
    plt.tight_layout()


# ============================================================
# 4. MAIN
# ============================================================

if __name__ == "__main__":
    # ----------- EDIT THIS SECTION --------------
    path = "./sig80211GenGr0.bin"     # <-- put your IQ file here
    filetype = "bin"           # "npy", "mat", or "bin"
    sample_rate = 40e6         # depends on your GRC setup
    # --------------------------------------------

    iq = load_iq(path, filetype=filetype)

    print("Loaded IQ samples:", len(iq))

    # ---- CALL PLOTTING FUNCTIONS ----
    plot_constellation(iq)


    plt.show()
